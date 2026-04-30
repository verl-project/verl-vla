# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os

import torch
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.config import HFModelConfig, TrainingWorkerConfig
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.base import BaseRollout, get_rollout_class

from verl_vla.models.register_vla_models import register_vla_models
from verl_vla.workers.config import ActorConfig, RolloutConfig, SFTActorConfig
from verl_vla.workers.engine.sac import SACTrainingWorker
from verl_vla.workers.engine.sft import SFTTrainingWorker
from verl_vla.workers.rollout import register_vla_rollouts

from .fsdp import FSDPEngineWithActionHEAD

register_vla_rollouts()

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


ACTOR_WORKER_REGISTRY = {
    "verl_vla.workers.config.ActorConfig": (ActorConfig, SACTrainingWorker),
    "verl_vla.workers.config.SFTActorConfig": (SFTActorConfig, SFTTrainingWorker),
}


class VLAActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    - __init__
    - init_model (override)
    - compute_ref_log_prob (not used)
    - compute_log_prob (not used)
    - update_actor (override)
    - load_checkpoint
    - save_checkpoint
    - update_weights
    - execute_checkpoint_engine
    """

    def _require_fsdp_rollout_engine(self) -> FSDPEngineWithActionHEAD:
        if self.config.actor.strategy not in {"fsdp", "fsdp2"}:
            raise RuntimeError(
                "switch_to_rollout/switch_to_train are only supported when actor.strategy is fsdp or fsdp2."
            )
        if self.actor is None or not isinstance(self.actor.engine, FSDPEngineWithActionHEAD):
            raise RuntimeError("VLA rollout switching requires a FSDPEngineWithActionHEAD-backed actor engine.")
        return self.actor.engine

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)
        self.tokenizer = getattr(self, "tokenizer", None) or model_config.tokenizer

        # 1. build reference model
        if "ref" in self.role:
            # VLA training currently does not require ref.
            ...

        # 2. build actor model
        if "actor" in self.role:
            actor_target = OmegaConf.select(self.config.actor, "_target_")
            if actor_target not in ACTOR_WORKER_REGISTRY:
                supported = ", ".join(sorted(ACTOR_WORKER_REGISTRY))
                raise ValueError(f"Unsupported actor config target: {actor_target}. Supported values: {supported}")
            actor_config_cls, worker_cls = ACTOR_WORKER_REGISTRY[actor_target]
            actor_config = omega_conf_to_dataclass(self.config.actor, dataclass_type=actor_config_cls)
            actor_config.model_config = model_config
            actor_training_config = TrainingWorkerConfig(
                model_type="vla_model",
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )
            self.actor = worker_cls(
                config=actor_training_config,
                actor_config=actor_config,
                tokenizer=self.tokenizer,
            )
            self.actor.reset()
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=True
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config,
                model_config=model_config,
                device_mesh=rollout_device_mesh,
                engine=self.actor.engine if "actor" in self.role else None,
                tokenizer=self.tokenizer,
            )

            # used for LoRA
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role or "rollout" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            is_master = False
            if "actor" in self.role:
                is_master = torch.distributed.get_rank() == 0
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, 
                is_master=is_master, 
                bucket_size=bucket_size, 
                **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto) -> DataProto:
        assert self._is_actor
        output = self.actor.train_mini_batch(data=data)
        return output.to("cpu") if output is not None else None

    # The interface reserved for VLA

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_rollout(self):
        self._require_fsdp_rollout_engine().switch_to_rollout()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_train(self):
        self._require_fsdp_rollout_engine().switch_to_train()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        assert self._is_rollout
        prompts = prompts.to(get_device_id())
        output = self.rollout.generate_sequences(prompts=prompts)
        return output.to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        # Rollout-only worker: receive weights from checkpoint engine
        if self._is_rollout and not self._is_actor:
            weights = self.checkpoint_engine.receive_weights()
            await self.rollout.update_weights(weights, global_steps=global_steps)
            return

        assert self.checkpoint_engine is not None

        # Actor-only worker: send weights via checkpoint engine
        if self._is_actor and not self._is_rollout:
            if self.config.rollout.checkpoint_engine.backend != "naive":
                per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
                await self.checkpoint_engine.send_weights(per_tensor_param)
            return

        # Colocated worker (actor + rollout)
        if self.config.rollout.checkpoint_engine.backend != "naive":
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(per_tensor_param)
            return

        from verl.utils.device import set_expandable_segments

        set_expandable_segments(False)

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=True)

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done or self.rollout.sleep_level != 1

        if do_lora_base_sync:
            per_tensor_base_params, _ = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(per_tensor_base_params, peft_config=peft_config, base_sync_done=False)

        log_gpu_memory_usage("After update_weights", logger=logger)

        self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

class VLAActorWorker(VLAActorRolloutRefWorker):
    def __init__(self, config, role=None):
        super().__init__(config, role="actor")
        
class VLARolloutWorker(VLAActorRolloutRefWorker):
    def __init__(self, config, role=None):
        super().__init__(config, role="rollout")

register_vla_models()
