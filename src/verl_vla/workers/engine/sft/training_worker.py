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

import torch
from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_device_name
from verl.workers.config import TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker

from verl_vla.workers.config import SFTActorConfig


class SFTTrainingWorker(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig, actor_config: SFTActorConfig, tokenizer=None):
        super().__init__(config=config)
        self.actor_config = actor_config
        self.tokenizer = tokenizer or self.model_config.tokenizer
        self._sft_initialized = False

    @staticmethod
    def _force_set_lr(opt: torch.optim.Optimizer, lr: float):
        for pg in opt.param_groups:
            pg["lr"] = lr

    def _ensure_sft_initialized(self):
        if self._sft_initialized:
            return
        self.engine.module.sft_init()
        self._sft_initialized = True

    def _extract_sft_obs(self, micro_batch: DataProto) -> DataProto:
        return micro_batch

    @staticmethod
    def _extract_sft_actions(micro_batch: DataProto) -> dict[str, torch.Tensor]:
        batch = micro_batch.batch
        if batch is None:
            raise ValueError("micro_batch.batch must not be None")
        if "action.full_action" in batch:
            return {"full_action": batch["action.full_action"]}
        if "action.action" in batch:
            return {"full_action": batch["action.action"]}
        if "action" in batch:
            return {"full_action": batch["action"]}
        raise KeyError("No action tensor found. Expected one of: action.full_action, action.action, action")

    @staticmethod
    def _extract_sft_valids(micro_batch: DataProto) -> torch.Tensor:
        batch = micro_batch.batch
        if batch is not None and "info.valids" in batch:
            return batch["info.valids"].float()
        return torch.ones(len(micro_batch), device=get_device_id(), dtype=torch.float32)

    def _update_sft_policy(self, data: DataProto) -> dict[str, float]:
        self._force_set_lr(self.engine.optimizer, self.actor_config.optim.lr)

        mini_batch_size = int(self.actor_config.sft_mini_batch_size)
        micro_batch_size = self.actor_config.sft_micro_batch_size_per_gpu
        if micro_batch_size is None:
            micro_batch_size = mini_batch_size
        micro_batch_size = int(micro_batch_size)

        mini_batches = data.split(mini_batch_size)
        grad_accum_steps = 0
        for mini_batch in mini_batches:
            grad_accum_steps += len(mini_batch.split(micro_batch_size))
        grad_accum_steps *= torch.distributed.get_world_size()

        self.engine.optimizer_zero_grad()

        loss_list = []
        for mini_batch in mini_batches:
            for micro_batch in mini_batch.split(micro_batch_size):
                micro_batch = micro_batch.to(get_device_id())
                obs = self._extract_sft_obs(micro_batch)
                actions = self._extract_sft_actions(micro_batch)
                valids = self._extract_sft_valids(micro_batch)
                with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                    bc_loss = self.engine.module.bc_loss(
                        obs=obs,
                        tokenizer=self.tokenizer,
                        actions=actions,
                        valids=valids,
                    )
                (bc_loss / grad_accum_steps).backward()
                loss_list.append(bc_loss.detach())

        grad_norm = self.engine.optimizer_step()
        mean_loss = torch.stack(loss_list).mean().item() if loss_list else 0.0
        return {
            "loss": mean_loss,
            "grad_norm": grad_norm if isinstance(grad_norm, float) else float(grad_norm),
            "sft/lr": self.actor_config.optim.lr,
        }

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: DataProto) -> DataProto:
        self._ensure_sft_initialized()
        with self.engine.train_mode():
            metrics = self._update_sft_policy(data)
        return DataProto(meta_info={"metrics": metrics})
