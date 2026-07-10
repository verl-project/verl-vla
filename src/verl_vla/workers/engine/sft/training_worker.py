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

import numpy as np
import torch
from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.debug import marked_timer
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
        self.actor_ema_enabled = self.actor_config.ema_decay is not None
        self.actor_ema_decay = 0.0 if self.actor_config.ema_decay is None else float(self.actor_config.ema_decay)
        self.actor_ema_shadow: dict[str, torch.Tensor] = {}
        self.actor_ema_initialized = False

    @staticmethod
    def _force_set_lr(opt: torch.optim.Optimizer, lr: float):
        for pg in opt.param_groups:
            pg["lr"] = lr

    def _ensure_sft_initialized(self):
        if self._sft_initialized:
            return
        self.engine.module.sft_init()
        self._sft_initialized = True

    def _get_named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        if hasattr(self.engine.module, "sac_get_named_actor_parameters"):
            return self.engine.module.sac_get_named_actor_parameters()
        return [(name, param) for name, param in self.engine.module.named_parameters() if param.requires_grad]

    def _init_actor_ema(self):
        if self.actor_ema_initialized:
            return
        self.actor_ema_shadow = {}
        if not self.actor_ema_enabled:
            self.actor_ema_initialized = True
            return
        for name, param in self._get_named_actor_parameters():
            self.actor_ema_shadow[name] = param.detach().clone().to(dtype=torch.float32)
        self.actor_ema_initialized = True

    @torch.no_grad()
    def _update_actor_ema(self):
        if not self.actor_ema_enabled:
            return
        one_minus_decay = 1.0 - self.actor_ema_decay
        for name, param in self._get_named_actor_parameters():
            self.actor_ema_shadow[name].mul_(self.actor_ema_decay).add_(
                param.detach().to(dtype=torch.float32), alpha=one_minus_decay
            )

    @torch.no_grad()
    def _apply_actor_ema_to_actor_module(self):
        if not self.actor_ema_enabled:
            return
        for name, param in self._get_named_actor_parameters():
            param.copy_(self.actor_ema_shadow[name].to(device=param.device, dtype=param.dtype))

    def _apply_acp_prompt_tags(self, micro_batch: DataProto) -> DataProto:
        acp_config = self.actor_config.acp
        data_keys = self.actor_config.data_keys
        if not acp_config.enable:
            return micro_batch

        indicators = micro_batch.batch[data_keys.indicator].reshape(-1)
        tasks = micro_batch.non_tensor_batch[data_keys.task]
        tagged_tasks = tasks.copy()
        keep_original = torch.rand(len(indicators), device=indicators.device) < float(acp_config.indicator_dropout_prob)
        indicators = indicators.detach().cpu().numpy()
        keep_original = keep_original.detach().cpu().numpy()
        for idx, indicator in enumerate(indicators):
            if keep_original[idx]:
                continue
            tag = acp_config.positive_tag if int(indicator) > 0 else acp_config.negative_tag
            tagged_tasks[idx] = f"{tasks[idx]}\n{tag}"
        micro_batch.non_tensor_batch[data_keys.task] = np.asarray(tagged_tasks, dtype=object)
        return micro_batch

    def _update_sft_policy(self, data: DataProto) -> dict[str, float]:
        if not self.actor_ema_initialized:
            self._init_actor_ema()

        timing_raw = {}

        with marked_timer("sft_update_policy", timing_raw):
            self._force_set_lr(self.engine.optimizer, self.actor_config.optim.lr)

            mini_batch_size = int(self.actor_config.mini_batch_size)
            micro_batch_size = self.actor_config.micro_batch_size
            if micro_batch_size is None:
                micro_batch_size = mini_batch_size
            micro_batch_size = int(micro_batch_size)

            mini_batches = data.split(mini_batch_size)
            split_micro_batches = [mini_batch.split(micro_batch_size) for mini_batch in mini_batches]
            grad_accum_steps = sum(len(micro_batches) for micro_batches in split_micro_batches)
            grad_accum_steps *= torch.distributed.get_world_size()

            loss_list = []
            sft_metric_lists: dict[str, list[torch.Tensor]] = {}
            with marked_timer("sft_forward_backward", timing_raw, color="red"):
                self.engine.optimizer_zero_grad()
                for micro_batches in split_micro_batches:
                    for micro_batch in micro_batches:
                        micro_batch = micro_batch.to(get_device_id())
                        micro_batch = self._apply_acp_prompt_tags(micro_batch)
                        batch = micro_batch.batch
                        data_keys = self.actor_config.data_keys
                        obs = micro_batch
                        valids = torch.ones(len(micro_batch), device=get_device_id(), dtype=torch.float32)
                        actions = {"action": batch[data_keys.action]}
                        action_mask = (
                            (~batch[data_keys.action_mask].bool()).float()
                            if data_keys.action_mask is not None
                            else None
                        )
                        target_values = batch[data_keys.target_value] if data_keys.target_value is not None else None
                        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                            sft_loss = self.engine.module.sft_loss(
                                obs=obs,
                                tokenizer=self.tokenizer,
                                actions=actions,
                                valids=valids,
                                action_mask=action_mask,
                                target_values=target_values,
                            )
                        (sft_loss / grad_accum_steps).backward()
                        loss_list.append(sft_loss.detach())
                        for name, value in getattr(self.engine.module, "sft_metrics", {}).items():
                            sft_metric_lists.setdefault(name, []).append(value.detach().float())

            with marked_timer("sft_optimizer_step", timing_raw):
                grad_norm = self.engine.optimizer_step()
                self._update_actor_ema()
                self._apply_actor_ema_to_actor_module()

            mean_loss = torch.stack(loss_list).mean().item() if loss_list else 0.0
            sft_metrics = {
                name: torch.stack(values).mean().item() for name, values in sft_metric_lists.items() if values
            }
            grad_norm_before_clip = grad_norm if isinstance(grad_norm, float) else float(grad_norm)
            grad_clip = float(self.engine.optimizer_config.clip_grad)
            if torch.isfinite(torch.tensor(grad_norm_before_clip)):
                grad_norm_after_clip = min(grad_norm_before_clip, grad_clip)
            else:
                grad_norm_after_clip = grad_norm_before_clip

        metrics = {
            "sft/loss": mean_loss,
            "sft/grad_norm": grad_norm_before_clip,
            "sft/grad_norm_before_clip": grad_norm_before_clip,
            "sft/grad_norm_after_clip": grad_norm_after_clip,
            "sft/lr": self.actor_config.optim.lr,
            "sft/grad_clip": grad_clip,
            "sft/num_mini_batches": len(mini_batches),
            "sft/num_micro_batches": sum(len(micro_batches) for micro_batches in split_micro_batches),
            "sft/grad_accum_steps": grad_accum_steps,
            "sft/actor_ema_enabled": float(self.actor_ema_enabled),
            "sft/actor_ema_decay": self.actor_ema_decay,
        }
        metrics.update(sft_metrics)
        metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})
        return metrics

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: DataProto) -> DataProto:
        self._ensure_sft_initialized()
        with self.engine.train_mode():
            metrics = self._update_sft_policy(data)
        return DataProto(meta_info={"metrics": metrics})
