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
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.config import HFModelConfig, RolloutConfig, TrainingWorkerConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
from verl.workers.rollout.base import BaseRollout, get_rollout_class

from verl_vla.models.register_vla_models import register_vla_models
from verl_vla.utils.data import get_dataproto_from_prefix, split_nested_dicts_or_tuples, valid_mean
from verl_vla.utils.replay_pool import SACReplayPool
from verl_vla.workers.config import ActorConfig
from verl_vla.workers.rollout import register_vla_rollouts

from .fsdp import FSDPEngineWithActionHEAD

register_vla_rollouts()

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VLATrainingWorker(TrainingWorker):
    """
    VLATrainingWorker extends the TrainingWorker used in the LLM pipeline.

    It overrides several methods to adapt the worker to the VLA setting,
    including differences in input/output structure, training behavior,
    and metric computation.

    Inherited from TrainingWorker:
    - __init__
    - to
    - reset
    - train_mini_batch (override)
    - save_checkpoint
    - load_checkpoint
    - set_loss_fn (not used)
    - infer_batch (not used)
    - train_batch (not used)
    """

    def __init__(self, config: TrainingWorkerConfig, actor_config: ActorConfig, tokenizer=None):
        super().__init__(config=config)
        self.actor_config = actor_config
        self.sac_config = actor_config.sac
        self.tokenizer = tokenizer or self.model_config.tokenizer
        self._sac_initialized = False

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: DataProto) -> DataProto:
        self._ensure_sac_initialized()
        with self.engine.train_mode():
            metrics = self._update_sac_policy(data)
        return DataProto(meta_info={"metrics": metrics})

    def _ensure_sac_initialized(self):
        if self._sac_initialized:
            return

        self.engine.module.sac_init()
        self.replay_pool = SACReplayPool(
            single_pool_capacity=self.actor_config.replay_pool_single_size,
            sample_device=get_device_name(),
        )
        self.replay_pool.load(self.actor_config.replay_pool_save_dir)

        self.critic_optimizer = torch.optim.Adam(
            self.engine.module.sac_get_critic_parameters(),
            lr=self.actor_config.critic_lr,
            weight_decay=self.actor_config.critic_weight_decay,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ConstantLR(self.critic_optimizer, factor=1.0)

        self.auto_entropy = self.sac_config.get("auto_entropy", False)
        if self.auto_entropy:
            self.target_entropy = torch.tensor(
                float(self.sac_config.get("target_entropy", -32.0)), device=get_device_name()
            )
            self.alpha_type = self.sac_config.get("alpha_type", "softplus")
            initial_alpha = float(self.sac_config.get("initial_alpha", 1 if self.alpha_type == "exp" else 0.01))
            if self.alpha_type == "exp":
                raw_alpha = np.log(np.exp(initial_alpha))
            elif self.alpha_type == "softplus":
                raw_alpha = np.log(np.exp(initial_alpha) - 1)
            else:
                raise NotImplementedError(f"Unsupported alpha_type: {self.alpha_type}")
            self.raw_alpha = torch.nn.Parameter(torch.full((1,), raw_alpha, device=get_device_name()))
            self.alpha_optimizer = torch.optim.Adam([self.raw_alpha], lr=self.sac_config.get("alpha_lr", 3e-4))
            self.alpha_scheduler = torch.optim.lr_scheduler.ConstantLR(self.alpha_optimizer, factor=1.0)

        self.actor_ema_enabled = bool(self.actor_config.get("actor_ema_enabled", True))
        self.actor_ema_decay = float(self.actor_config.get("actor_ema_decay", 0.995))
        self.actor_ema_shadow: dict[str, torch.Tensor] = {}
        self.actor_ema_initialized = False
        self.bc_loss_coef = float(self.sac_config.get("bc_loss_coef", 0.5))
        self._sac_initialized = True

    @property
    def _grad_clip(self) -> float:
        return self.actor_config.optim.clip_grad

    def _init_actor_ema(self):
        if self.actor_ema_initialized:
            return
        self.actor_ema_shadow = {}
        if not self.actor_ema_enabled:
            self.actor_ema_initialized = True
            return
        for name, param in self.engine.module.sac_get_named_actor_parameters():
            self.actor_ema_shadow[name] = param.detach().clone().to(dtype=torch.float32)
        self.actor_ema_initialized = True

    @torch.no_grad()
    def _update_actor_ema(self):
        if not self.actor_ema_enabled:
            return
        one_minus_decay = 1.0 - self.actor_ema_decay
        for name, param in self.engine.module.sac_get_named_actor_parameters():
            self.actor_ema_shadow[name].mul_(self.actor_ema_decay).add_(
                param.detach().to(dtype=torch.float32), alpha=one_minus_decay
            )

    @torch.no_grad()
    def _apply_actor_ema_to_actor_module(self):
        if not self.actor_ema_enabled:
            return
        for name, param in self.engine.module.sac_get_named_actor_parameters():
            param.copy_(self.actor_ema_shadow[name].to(device=param.device, dtype=param.dtype))

    def _get_alpha(self) -> torch.Tensor:
        if self.auto_entropy:
            if self.alpha_type == "exp":
                return self.raw_alpha.exp()
            if self.alpha_type == "softplus":
                return F.softplus(self.raw_alpha)
            raise NotImplementedError(f"Unsupported alpha_type: {self.alpha_type}")
        return torch.tensor(float(self.sac_config.get("initial_alpha", 0.2)), device=get_device_name())

    def _calculate_actor_loss(
        self,
        log_probs: Optional[torch.Tensor],
        q_values: torch.Tensor,
        valids: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._get_alpha()
        loss = -q_values if log_probs is None else alpha * log_probs - q_values
        return (loss * valids).sum() / valids.sum().clamp_min(1.0)

    def _calculate_alpha_loss(self, log_probs: Optional[torch.Tensor], valids: torch.Tensor) -> torch.Tensor:
        if log_probs is None:
            return torch.tensor(0.0, device=valids.device)
        alpha_loss = -self._get_alpha() * (log_probs.detach() + self.target_entropy)
        return (alpha_loss * valids).sum() / valids.sum().clamp_min(1.0)

    def _calculate_critic_loss(
        self,
        q_predict: torch.Tensor,
        q_target: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_log_prob: Optional[torch.Tensor],
        valids: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self.sac_config.gamma
        alpha = self._get_alpha()
        with torch.no_grad():
            y = (
                rewards + gamma * (1.0 - dones) * q_target
                if next_log_prob is None
                else rewards + gamma * (1.0 - dones) * (q_target - alpha * next_log_prob)
            )

        y = y.unsqueeze(1).expand_as(q_predict)
        valid_mask = valids.unsqueeze(1)
        mse = F.mse_loss(q_predict, y, reduction="none")
        per_critic = (mse * valid_mask).sum(dim=0) / valid_mask.sum().clamp_min(1.0)
        return per_critic.sum()

    def _forward_critic(self, micro_batch: DataProto, resample=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s0 = get_dataproto_from_prefix(micro_batch, "t0.obs.")
        s1 = get_dataproto_from_prefix(micro_batch, "t1.obs.")
        a0 = get_dataproto_from_prefix(micro_batch, "t0.action.").batch
        a1 = get_dataproto_from_prefix(micro_batch, "t1.action.").batch

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            with torch.no_grad():
                state_features = self.engine.module.sac_forward_state_features(
                    DataProto.concat([s0, s1]), self.tokenizer
                )
                s0_state_features, s1_state_features = split_nested_dicts_or_tuples(state_features, 2)
                if resample:
                    a1_actions, log_probs_1, _ = self.engine.module.sac_forward_actor(
                        s1_state_features,
                        is_first_micro_batch=False,
                    )
                    a1 = {"action": a1_actions}
                else:
                    log_probs_1 = None

            q_values_0 = self.engine.module.sac_forward_critic(
                a0,
                s0_state_features,
                task_ids=micro_batch.batch["info.task_ids"],
                use_target_network=False,
                method="cat",
                requires_grad=True,
            )
            q_values_1 = self.engine.module.sac_forward_critic(
                a1,
                s1_state_features,
                task_ids=micro_batch.batch["info.task_ids"],
                use_target_network=True,
                method="min",
                requires_grad=False,
            )

            critic_loss = self._calculate_critic_loss(
                q_predict=q_values_0,
                q_target=q_values_1,
                rewards=micro_batch.batch["info.rewards"],
                dones=micro_batch.batch["info.dones"],
                next_log_prob=log_probs_1,
                valids=micro_batch.batch["info.valids"],
            )
        return critic_loss, q_values_0, q_values_1

    def _forward_actor(
        self,
        micro_batch: DataProto,
        is_first_micro_batch: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, dict[str, float]]:
        s0 = get_dataproto_from_prefix(micro_batch, "t0.obs.")

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            s0_state_features = self.engine.module.sac_forward_state_features(s0, self.tokenizer)
            a0_actions, log_probs_0, actor_forward_metrics = self.engine.module.sac_forward_actor(
                s0_state_features,
                is_first_micro_batch=is_first_micro_batch,
            )
            q_values_0 = self.engine.module.sac_forward_critic(
                {"action": a0_actions},
                s0_state_features,
                task_ids=micro_batch.batch["info.task_ids"],
                use_target_network=False,
                method="min",
                requires_grad=False,
            )

            sac_loss = self._calculate_actor_loss(
                log_probs=log_probs_0,
                q_values=q_values_0,
                valids=micro_batch.batch["info.valids"],
            )
            if self.bc_loss_coef > 0:
                bc_loss = self.engine.module.bc_loss(
                    obs=s0,
                    tokenizer=self.tokenizer,
                    actions={"full_action": a0_actions},
                    valids=micro_batch.batch["info.valids"],
                )
                actor_loss = sac_loss + self.bc_loss_coef * bc_loss
            else:
                actor_loss = sac_loss
        return actor_loss, log_probs_0, q_values_0, actor_forward_metrics

    @staticmethod
    def _force_set_lr(opt: torch.optim.Optimizer, lr: float):
        for pg in opt.param_groups:
            pg["lr"] = lr

    def _update_sac_policy(self, data: DataProto) -> dict:
        if not self.actor_ema_initialized:
            self._init_actor_ema()

        self._force_set_lr(self.engine.optimizer, 5e-6)
        self._force_set_lr(self.critic_optimizer, 1e-4)

        if "empty_batch" not in data.meta_info:
            replay_batch_keys = [key for key in data.batch.keys() if key.startswith(("t0.", "t1.", "info."))]
            replay_non_tensor_batch_keys = [
                key for key in data.non_tensor_batch.keys() if key.startswith(("t0.", "t1.", "info."))
            ]
            self.replay_pool.add_batch(
                data.select(batch_keys=replay_batch_keys, non_tensor_batch_keys=replay_non_tensor_batch_keys),
                task_ids=data.batch["info.task_ids"],
            )

        critic_batch, critic_replay_sample_info = self.replay_pool.sample_batch(
            self.actor_config.sac_mini_batch_size,
            positive_sample_ratio=float(self.sac_config.get("critic_replay_positive_sample_ratio", 0.5)),
            return_sample_info=True,
        )
        micro_batches = critic_batch.split(self.actor_config.sac_micro_batch_size_per_gpu)
        grad_accum_steps = len(micro_batches) * torch.distributed.get_world_size()
        global_steps = data.meta_info["global_steps"]

        actor_logprobs_list, actor_qvalues_list = [], []
        critic_qvalues_0_list, critic_qvalues_1_list = [], []
        actor_loss_list, critic_loss_list, alpha_loss_list = [], [], []
        actor_forward_metrics: dict[str, float] = {}

        self.critic_optimizer.zero_grad()
        for batch_idx, micro_batch in enumerate(micro_batches):
            logger.info(f"[{batch_idx + 1}/{len(micro_batches)}] critic micro batch ")
            micro_batch = micro_batch.to(get_device_id())
            raw_critic_loss, q_values_0, q_values_1 = self._forward_critic(micro_batch, resample=True)
            (raw_critic_loss / grad_accum_steps).backward()
            critic_loss_list.append(raw_critic_loss.detach().item())
            critic_qvalues_0_list.append(q_values_0.mean(dim=-1).detach())
            critic_qvalues_1_list.append(q_values_1.detach())
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.engine.module.sac_get_critic_parameters(), max_norm=self._grad_clip
        )
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        update_actor = (
            global_steps >= self.actor_config.critic_warmup_steps
            and global_steps % self.actor_config.actor_update_interval == 0
        )
        actor_replay_sample_info = {"actual_positive_sample_ratio": 0.0}
        if update_actor:
            actor_batch, actor_replay_sample_info = self.replay_pool.sample_batch(
                self.actor_config.sac_mini_batch_size,
                positive_sample_ratio=float(self.sac_config.get("actor_replay_positive_sample_ratio", 0.5)),
                return_sample_info=True,
            )
            micro_batches = actor_batch.split(self.actor_config.sac_micro_batch_size_per_gpu)

            self.engine.optimizer_zero_grad()
            for batch_idx, micro_batch in enumerate(micro_batches):
                logger.info(f"[{batch_idx + 1}/{len(micro_batches)}] actor micro batch ")
                micro_batch = micro_batch.to(get_device_id())
                raw_actor_loss, log_probs, q_values, actor_forward_metrics_mb = self._forward_actor(
                    micro_batch,
                    is_first_micro_batch=(batch_idx == 0),
                )
                (raw_actor_loss / grad_accum_steps).backward()
                actor_loss_list.append(raw_actor_loss.detach().item())
                if log_probs is not None:
                    actor_logprobs_list.append(log_probs.detach())
                actor_qvalues_list.append(q_values.detach())
                actor_forward_metrics.update(actor_forward_metrics_mb)
            actor_grad_norm = self.engine.optimizer_step()
            self._update_actor_ema()
            self._apply_actor_ema_to_actor_module()

            if self.auto_entropy and actor_logprobs_list:
                self.alpha_optimizer.zero_grad()
                for micro_batch, log_probs in zip(micro_batches, actor_logprobs_list, strict=False):
                    micro_batch = micro_batch.to(get_device_id())
                    raw_alpha_loss = self._calculate_alpha_loss(log_probs, micro_batch.batch["info.valids"])
                    (raw_alpha_loss / grad_accum_steps).backward()
                    alpha_loss_list.append(raw_alpha_loss.detach().item())
                torch.distributed.all_reduce(self.raw_alpha.grad, op=torch.distributed.ReduceOp.SUM)
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_alpha, max_norm=self._grad_clip)
                self.alpha_optimizer.step()
                self.alpha_scheduler.step()

        self.engine.module.sac_update_target_network(self.sac_config.tau)
        if global_steps % self.actor_config.replay_pool_save_interval == 0:
            self.replay_pool.save(self.actor_config.replay_pool_save_dir)

        critic_positive_mask = critic_batch.batch["info.positive_sample_mask"].to(torch.bool)
        critic_valid_mask = critic_batch.batch["info.valids"].to(torch.bool)
        critic_qvalues_0 = torch.cat(critic_qvalues_0_list) if critic_qvalues_0_list else None
        positive_qvalue_mean = (
            critic_qvalues_0[(critic_positive_mask & critic_valid_mask).to(critic_qvalues_0.device)]
            .mean()
            .detach()
            .item()
            if critic_qvalues_0 is not None and (critic_positive_mask & critic_valid_mask).any()
            else 0.0
        )
        negative_qvalue_mean = (
            critic_qvalues_0[(~critic_positive_mask & critic_valid_mask).to(critic_qvalues_0.device)]
            .mean()
            .detach()
            .item()
            if critic_qvalues_0 is not None and (~critic_positive_mask & critic_valid_mask).any()
            else 0.0
        )

        metrics = {
            "data/reward_mean": valid_mean(critic_batch.batch["info.rewards"], critic_batch.batch["info.valids"])
            .detach()
            .item(),
            "data/valid_ratio": critic_batch.batch["info.valids"].float().mean().item(),
            "sac/critic_replay_sampled_ratio": critic_replay_sample_info["actual_positive_sample_ratio"],
            "sac/actor_replay_sampled_ratio": actor_replay_sample_info["actual_positive_sample_ratio"]
            if update_actor
            else 0.0,
            "sac/replay_pool_positive_size": critic_replay_sample_info["positive_size"],
            "sac/replay_pool_negative_size": critic_replay_sample_info["negative_size"],
            "sac/replay_task_count": critic_replay_sample_info["task_count"],
            "sac/alpha": self._get_alpha().detach().item(),
            "sac/actor_ema_enabled": float(self.actor_ema_enabled),
            "sac/actor_ema_decay": self.actor_ema_decay,
            "sac/replay_pool_size": len(self.replay_pool),
            "critic/loss": sum(critic_loss_list) / len(critic_loss_list) if critic_loss_list else 0.0,
            "critic/lr": self.critic_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": critic_grad_norm.detach().item(),
            "critic/qvalue0_mean": (
                valid_mean(torch.cat(critic_qvalues_0_list), critic_batch.batch["info.valids"]).detach().item()
                if critic_qvalues_0_list
                else 0.0
            ),
            "critic/qvalue1_mean": (
                valid_mean(torch.cat(critic_qvalues_1_list), critic_batch.batch["info.valids"]).detach().item()
                if critic_qvalues_1_list
                else 0.0
            ),
            "critic/positive_qvalue_mean": positive_qvalue_mean,
            "critic/negative_qvalue_mean": negative_qvalue_mean,
            "critic/diff_pos_neg_qvalue_mean": positive_qvalue_mean - negative_qvalue_mean,
        }
        if update_actor:
            metrics.update(
                {
                    "actor/loss": sum(actor_loss_list) / len(actor_loss_list),
                    "actor/lr": self.engine.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    "actor/logprob_mean": (
                        valid_mean(torch.cat(actor_logprobs_list), actor_batch.batch["info.valids"]).detach().item()
                        if actor_logprobs_list
                        else 0.0
                    ),
                    "actor/qvalue_mean": valid_mean(torch.cat(actor_qvalues_list), actor_batch.batch["info.valids"])
                    .detach()
                    .item(),
                    "sac/alpha_lr": self.alpha_optimizer.param_groups[0]["lr"]
                    if self.auto_entropy and actor_logprobs_list
                    else 0.0,
                    "sac/alpha_loss": sum(alpha_loss_list) / len(alpha_loss_list)
                    if self.auto_entropy and alpha_loss_list
                    else 0.0,
                    "sac/alpha_grad_norm": alpha_grad_norm.detach().item()
                    if self.auto_entropy and actor_logprobs_list
                    else 0.0,
                }
            )
            metrics.update({f"actor/{k}": v for k, v in actor_forward_metrics.items()})

        return metrics


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
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            actor_training_config = TrainingWorkerConfig(
                model_type="vla_model",
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )

            self.actor = VLATrainingWorker(
                config=actor_training_config,
                actor_config=actor_config,
                tokenizer=self.tokenizer,
            )
            self.actor.reset()
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

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
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
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


register_vla_models()
