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
from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_device_name
from verl.workers.config import TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker

from verl_vla.utils.data import get_dataproto_from_prefix, split_nested_dicts_or_tuples, valid_mean
from verl_vla.utils.replay_pool import SACReplayPool
from verl_vla.utils.scalar_schedule import ScheduledScalar
from verl_vla.workers.config import ActorConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SACTrainingWorker(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig, actor_config: ActorConfig, tokenizer=None):
        super().__init__(config=config)
        self.actor_config = actor_config
        self.sac_mini_batch_size = self.actor_config.sac_mini_batch_size // torch.distributed.get_world_size()
        self.sac_config = actor_config.sac
        self.tokenizer = tokenizer or self.model_config.tokenizer
        self._sac_initialized = False

    @staticmethod
    def _force_set_lr(opt: torch.optim.Optimizer, lr: float):
        for pg in opt.param_groups:
            pg["lr"] = lr

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

        self.actor_ema_enabled = bool(self.actor_config.actor_ema_enabled)
        self.actor_ema_decay = float(self.actor_config.actor_ema_decay)
        self.actor_ema_shadow: dict[str, torch.Tensor] = {}
        self.actor_ema_initialized = False
        self.actor_ema_scheduler = ScheduledScalar(
            base_value=self.actor_ema_decay,
            enabled=bool(self.actor_config.actor_ema_dynamic_enabled),
            initial_value=float(self.actor_config.actor_ema_strength_initial),
            final_value=float(self.actor_config.actor_ema_strength_final),
            method=self.actor_config.actor_ema_schedule_method,
        )
        critic_target_strength_default = ScheduledScalar(base_value=1.0 - float(self.sac_config.tau)).base_value
        self.critic_target_ema_scheduler = ScheduledScalar(
            base_value=critic_target_strength_default,
            enabled=bool(self.sac_config.critic_target_ema_dynamic_enabled),
            initial_value=float(self.sac_config.critic_target_ema_strength_initial),
            final_value=float(self.sac_config.critic_target_ema_strength_final),
            method=self.sac_config.critic_target_ema_schedule_method,
        )
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
        one_minus_decay = 1.0 - self.actor_ema_scheduler.current_value
        for name, param in self.engine.module.sac_get_named_actor_parameters():
            self.actor_ema_shadow[name].mul_(self.actor_ema_scheduler.current_value).add_(
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
                        task_ids=micro_batch.batch["info.task_ids"],
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
                task_ids=micro_batch.batch["info.task_ids"],
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

    def _update_sac_policy(self, data: DataProto) -> dict:
        if not self.actor_ema_initialized:
            self._init_actor_ema()

        global_steps = data.meta_info["global_steps"]
        rollout_success_rate = self.actor_ema_scheduler.control_value
        if "data/trajectory_avg_reward" in data.meta_info:
            rollout_success_rate = float(data.meta_info["data/trajectory_avg_reward"])
        self.actor_ema_scheduler.refresh(rollout_success_rate)
        self.critic_target_ema_scheduler.refresh(rollout_success_rate)

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
            self.sac_mini_batch_size,
            positive_sample_ratio=float(self.sac_config.get("critic_replay_positive_sample_ratio", 0.5)),
            return_sample_info=True,
        )
        micro_batches = critic_batch.split(self.actor_config.sac_micro_batch_size_per_gpu)
        grad_accum_steps = len(micro_batches) * torch.distributed.get_world_size()

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
                self.sac_mini_batch_size,
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

        force_tau_one_in_warmup = bool(self.sac_config.get("force_critic_tau_one_in_warmup", True))
        critic_target_tau = (
            1.0
            if force_tau_one_in_warmup and global_steps < self.actor_config.critic_warmup_steps
            else (1.0 - self.critic_target_ema_scheduler.current_value)
        )
        self.engine.module.sac_update_target_network(critic_target_tau)
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
            "sac/rollout_success_rate": self.actor_ema_scheduler.control_value,
            "sac/actor_ema_enabled": float(self.actor_ema_enabled),
            "sac/actor_ema_dynamic_enabled": float(self.actor_ema_scheduler.enabled),
            "sac/actor_ema_decay": self.actor_ema_scheduler.current_value,
            "sac/actor_ema_strength_initial": self.actor_ema_scheduler.initial_value,
            "sac/actor_ema_strength_final": self.actor_ema_scheduler.final_value,
            "sac/critic_target_ema_dynamic_enabled": float(self.critic_target_ema_scheduler.enabled),
            "sac/critic_target_tau": critic_target_tau,
            "sac/critic_target_ema_strength": 1.0 - critic_target_tau,
            "sac/critic_target_ema_strength_initial": self.critic_target_ema_scheduler.initial_value,
            "sac/critic_target_ema_strength_final": self.critic_target_ema_scheduler.final_value,
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

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: DataProto) -> DataProto:
        self._ensure_sac_initialized()
        with self.engine.train_mode():
            metrics = self._update_sac_policy(data)
        return DataProto(meta_info={"metrics": metrics})
