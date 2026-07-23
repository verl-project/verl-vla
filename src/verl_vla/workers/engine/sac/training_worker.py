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
from collections import defaultdict
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
from verl_vla.workers.config import ActorConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SACTrainingWorker(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig, actor_config: ActorConfig, tokenizer=None):
        super().__init__(config=config)
        self.actor_config = actor_config
        self.sac_mini_batch_size = self._global_to_local_batch_size(self.actor_config.mini_batch_size)
        self.online_replay_sample_batch_size = self._global_to_local_batch_size(
            self.actor_config.replay.online_sample_batch_size
            if self.actor_config.replay.online_sample_batch_size is not None
            else self.actor_config.mini_batch_size
        )
        self.offline_replay_sample_batch_size = self._global_to_local_batch_size(
            self.actor_config.replay.offline_sample_batch_size
            if self.actor_config.replay.offline_sample_batch_size is not None
            else self.actor_config.mini_batch_size
        )
        self.sac_config = actor_config.sac
        self.tokenizer = tokenizer or self.model_config.tokenizer
        self._sac_initialized = False

    @staticmethod
    def _global_to_local_batch_size(global_batch_size: int) -> int:
        return int(global_batch_size) // torch.distributed.get_world_size()

    def _ensure_sac_initialized(self):
        if self._sac_initialized:
            return

        self.engine.module.sac_init()
        self.replay_pool = SACReplayPool(
            single_pool_capacity=self.actor_config.replay.online_single_size,
            sample_device=get_device_name(),
        )
        self.replay_pool.load(self.actor_config.replay.save_dir)
        self.offline_replay_pool = SACReplayPool(
            single_pool_capacity=self.actor_config.replay.offline_single_size,
            sample_device=get_device_name(),
        )

        self.critic_optimizer = torch.optim.Adam(
            self.engine.module.sac_get_critic_parameters(),
            lr=self.actor_config.critic.lr,
            weight_decay=self.actor_config.critic.weight_decay,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ConstantLR(self.critic_optimizer, factor=1.0)

        self.backup_entropy = bool(self.sac_config.get("backup_entropy", True))
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

        self.actor_ema_enabled = self.actor_config.ema_decay is not None
        self.actor_ema_decay = 0.0 if self.actor_config.ema_decay is None else float(self.actor_config.ema_decay)
        self.actor_ema_shadow: dict[str, torch.Tensor] = {}
        self.actor_ema_initialized = False
        self.td3_enabled = bool(self.actor_config.td3.enabled)
        self.td3_bc_alpha = float(self.actor_config.td3.bc_alpha)
        self.cql_enabled = bool(self.actor_config.cql.enabled)
        self.cql_alpha = float(self.actor_config.cql.alpha)
        self.cql_temperature = float(self.actor_config.cql.temperature)
        cql_noise_scale = self.actor_config.cql.noise_scale
        self.cql_noise_scale = None if cql_noise_scale is None else float(cql_noise_scale)
        self.skip_critic_update_when_actor_update = bool(self.actor_config.critic.skip_update_when_actor_update)
        self._sac_initialized = True

    def _add_data_to_replay_pool(self, replay_pool: SACReplayPool, data: DataProto):
        replay_batch_keys = [key for key in data.batch.keys() if key.startswith(("t0.", "t1.", "info."))]
        replay_non_tensor_batch_keys = [
            key for key in data.non_tensor_batch.keys() if key.startswith(("t0.", "t1.", "info."))
        ]
        replay_pool.add_batch(
            data.select(batch_keys=replay_batch_keys, non_tensor_batch_keys=replay_non_tensor_batch_keys),
            task_ids=data.non_tensor_batch["t0.obs.task_id"],
        )

    def _task_ids_from_obs(self, data: DataProto) -> torch.Tensor:
        return torch.as_tensor(
            data.non_tensor_batch["t0.obs.task_id"],
            dtype=torch.long,
            device=get_device_id(),
        )

    def _sample_rlpd_batch(
        self,
        positive_sample_ratio: float,
    ) -> tuple[list[DataProto], dict]:
        online_batch = None
        online_sample_info = {
            "actual_positive_sample_ratio": 0.0,
            "positive_size": self.replay_pool.positive_size,
            "negative_size": self.replay_pool.negative_size,
            "task_count": len(self.replay_pool.task_pools),
        }
        if self.online_replay_sample_batch_size > 0:
            online_batch, online_sample_info = self.replay_pool.sample_batch(
                self.online_replay_sample_batch_size,
                positive_sample_ratio=positive_sample_ratio,
                return_sample_info=True,
            )

        offline_batch = None
        offline_sample_info = {
            "actual_positive_sample_ratio": 0.0,
            "positive_size": self.offline_replay_pool.positive_size,
            "negative_size": self.offline_replay_pool.negative_size,
            "task_count": len(self.offline_replay_pool.task_pools),
        }
        if self.offline_replay_sample_batch_size > 0 and len(self.offline_replay_pool) > 0:
            offline_batch, offline_sample_info = self.offline_replay_pool.sample_batch(
                self.offline_replay_sample_batch_size,
                positive_sample_ratio=positive_sample_ratio,
                return_sample_info=True,
            )

        if online_batch is None and offline_batch is None:
            raise ValueError(
                "No replay samples available. "
                f"online_sample_batch_size={self.online_replay_sample_batch_size}, "
                f"offline_sample_batch_size={self.offline_replay_sample_batch_size}, "
                f"online_pool_size={len(self.replay_pool)}, offline_pool_size={len(self.offline_replay_pool)}"
            )

        online_sample_count = len(online_batch) if online_batch is not None else 0
        offline_sample_count = len(offline_batch) if offline_batch is not None else 0

        sample_info = {
            "online": {
                "actual_positive_sample_ratio": online_sample_info["actual_positive_sample_ratio"],
                "sample_count": online_sample_count,
                "positive_size": online_sample_info["positive_size"],
                "negative_size": online_sample_info["negative_size"],
                "task_count": online_sample_info["task_count"],
                "pool_size": len(self.replay_pool),
            },
            "offline": {
                "actual_positive_sample_ratio": offline_sample_info["actual_positive_sample_ratio"],
                "sample_count": offline_sample_count,
                "positive_size": offline_sample_info["positive_size"],
                "negative_size": offline_sample_info["negative_size"],
                "task_count": offline_sample_info["task_count"],
                "pool_size": len(self.offline_replay_pool),
            },
        }
        return [batch for batch in (online_batch, offline_batch) if batch is not None], sample_info

    @property
    def _actor_grad_clip(self) -> float:
        return self.actor_config.optim.clip_grad

    @property
    def _critic_grad_clip(self) -> float:
        return self.actor_config.critic.grad_clip or self._actor_grad_clip

    def _post_clip_norm(self, pre_clip_norm: torch.Tensor | float, clip_threshold: float) -> float:
        if isinstance(pre_clip_norm, torch.Tensor):
            pre_clip_norm = pre_clip_norm.detach().item()
        if not np.isfinite(pre_clip_norm):
            return float(pre_clip_norm)
        return float(min(pre_clip_norm, clip_threshold))

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
        q_policy: Optional[torch.Tensor],
        q_target: torch.Tensor,
        rewards: torch.Tensor,
        terminateds: torch.Tensor,
        next_log_prob: Optional[torch.Tensor],
        valids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gamma = self.actor_config.critic.gamma
        alpha = self._get_alpha()
        with torch.no_grad():
            y = (
                rewards + gamma * (1.0 - terminateds) * q_target
                if next_log_prob is None
                else rewards + gamma * (1.0 - terminateds) * (q_target - alpha * next_log_prob)
            )

        y = y.unsqueeze(1).expand_as(q_predict)
        valid_mask = valids.unsqueeze(1)
        mse = F.mse_loss(q_predict, y, reduction="none")
        per_critic = (mse * valid_mask).sum(dim=0) / valid_mask.sum().clamp_min(1.0)
        td_loss = per_critic.sum()

        critic_loss_metrics = {
            "td_loss": td_loss.detach(),
        }
        cql_loss = torch.tensor(0.0, device=q_predict.device)
        if self.cql_enabled and q_policy is not None:
            q_candidates = torch.stack([q_predict, q_policy], dim=0)
            cql_per_critic = (
                torch.logsumexp(q_candidates / self.cql_temperature, dim=0) * self.cql_temperature - q_predict
            )
            cql_per_critic = (cql_per_critic * valid_mask).sum(dim=0) / valid_mask.sum().clamp_min(1.0)
            cql_loss = self.cql_alpha * cql_per_critic.sum()
            critic_loss_metrics["cql_loss"] = cql_loss.detach()

            q_gap = q_policy - q_predict
            valid_q_gap = q_gap.masked_select(valid_mask.expand_as(q_gap).bool())
            if valid_q_gap.numel() > 0:
                critic_loss_metrics["cql/q_gap_mean"] = valid_q_gap.mean().detach()
                critic_loss_metrics["cql/q_gap_max"] = valid_q_gap.max().detach()
                critic_loss_metrics["cql/q_gap_pos_ratio"] = (valid_q_gap > 0).float().mean().detach()

        total_loss = td_loss + cql_loss
        return total_loss, critic_loss_metrics

    def _forward_critic(
        self, micro_batch: DataProto, resample=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        s0 = get_dataproto_from_prefix(micro_batch, "t0.obs.")
        s1 = get_dataproto_from_prefix(micro_batch, "t1.obs.")
        a0 = get_dataproto_from_prefix(micro_batch, "t0.action.").batch
        a1 = get_dataproto_from_prefix(micro_batch, "t1.action.").batch
        task_ids = self._task_ids_from_obs(micro_batch)

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            with torch.no_grad():
                state_features = self.engine.module.sac_forward_state_features(
                    DataProto.concat([s0, s1]), self.tokenizer
                )
                s0_state_features, s1_state_features = split_nested_dicts_or_tuples(state_features, 2)
                if resample:
                    a1_actions, log_probs_1, _ = self.engine.module.sac_forward_actor(
                        s1_state_features,
                        task_ids=task_ids,
                        is_first_micro_batch=False,
                    )
                    a1 = {"action": a1_actions}
                    if not self.backup_entropy:
                        log_probs_1 = None
                else:
                    log_probs_1 = None

            q_values_0 = self.engine.module.sac_forward_critic(
                a0,
                s0_state_features,
                task_ids=task_ids,
                use_target_network=False,
                method="cat",
                requires_grad=True,
            )
            q_values_1 = self.engine.module.sac_forward_critic(
                a1,
                s1_state_features,
                task_ids=task_ids,
                use_target_network=True,
                method="min",
                requires_grad=False,
            )
            q_policy_0 = None
            if self.cql_enabled:
                with torch.no_grad():
                    policy_actions_0, _, _ = self.engine.module.sac_forward_actor(
                        s0_state_features,
                        task_ids=task_ids,
                        is_first_micro_batch=False,
                        noise_scale=self.cql_noise_scale,
                    )
                q_policy_0 = self.engine.module.sac_forward_critic(
                    {"action": policy_actions_0},
                    s0_state_features,
                    task_ids=task_ids,
                    use_target_network=False,
                    method="cat",
                    requires_grad=True,
                )

            critic_loss, critic_loss_metrics = self._calculate_critic_loss(
                q_predict=q_values_0,
                q_policy=q_policy_0,
                q_target=q_values_1,
                rewards=micro_batch.batch["info.rewards"],
                terminateds=micro_batch.batch["info.terminateds"],
                next_log_prob=log_probs_1,
                valids=micro_batch.batch["info.valids"],
            )
        return critic_loss, q_values_0, q_values_1, critic_loss_metrics

    def _forward_actor(
        self,
        micro_batch: DataProto,
        is_first_micro_batch: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, dict[str, float]]:
        s0 = get_dataproto_from_prefix(micro_batch, "t0.obs.")
        a0 = get_dataproto_from_prefix(micro_batch, "t0.action.").batch
        task_ids = self._task_ids_from_obs(micro_batch)

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            s0_state_features = self.engine.module.sac_forward_state_features(s0, self.tokenizer)
            a0_actions, log_probs_0, actor_forward_metrics = self.engine.module.sac_forward_actor(
                s0_state_features,
                task_ids=task_ids,
                is_first_micro_batch=is_first_micro_batch,
            )
            q_values_0 = self.engine.module.sac_forward_critic(
                {"action": a0_actions},
                s0_state_features,
                task_ids=task_ids,
                use_target_network=False,
                method="min",
                requires_grad=False,
            )

            sac_loss = self._calculate_actor_loss(
                log_probs=log_probs_0,
                q_values=q_values_0,
                valids=micro_batch.batch["info.valids"],
            )
            if self.td3_enabled:
                bc_loss = self.engine.module.sft_loss(
                    obs=s0,
                    tokenizer=self.tokenizer,
                    actions=a0,
                    valids=micro_batch.batch["info.valids"],
                )
                actor_loss = sac_loss + self.td3_bc_alpha * bc_loss
                actor_forward_metrics.update(
                    {
                        "td3_bc_weight": self.td3_bc_alpha,
                        "td3_bc_q_loss": sac_loss.detach().item(),
                        "td3_bc_bc_loss": bc_loss.detach().item(),
                    }
                )
            else:
                actor_loss = sac_loss
        return actor_loss, log_probs_0, q_values_0, actor_forward_metrics

    def _update_sac_policy(self, data: DataProto) -> dict:
        if not self.actor_ema_initialized:
            self._init_actor_ema()

        global_steps = data.meta_info["global_steps"]
        critic_only_update = bool(data.meta_info.get("critic_only_update", False))
        add_to_offline_replay_only = bool(data.meta_info.get("add_to_offline_replay_only", False))

        if add_to_offline_replay_only:
            self._add_data_to_replay_pool(self.offline_replay_pool, data)
            return {
                "sac/replay/offline/pool_size": len(self.offline_replay_pool),
                "sac/replay/offline/positive_size": self.offline_replay_pool.positive_size,
                "sac/replay/offline/negative_size": self.offline_replay_pool.negative_size,
                "sac/replay/offline/task_count": len(self.offline_replay_pool.task_pools),
                "sac/offline_replay_prefill_only": 1.0,
            }

        if "empty_batch" not in data.meta_info:
            self._add_data_to_replay_pool(self.replay_pool, data)

        critic_batches, critic_replay_sample_info = self._sample_rlpd_batch(
            positive_sample_ratio=float(self.actor_config.replay.critic_positive_sample_ratio)
        )
        critic_micro_batches = [
            micro_batch for batch in critic_batches for micro_batch in batch.split(self.actor_config.micro_batch_size)
        ]
        critic_grad_accum_steps = len(critic_micro_batches)

        update_actor = (
            global_steps >= self.actor_config.critic.warmup_steps
            and global_steps % self.actor_config.actor_update_interval == 0
            and not critic_only_update
        )
        skip_critic_update = self.skip_critic_update_when_actor_update and update_actor

        actor_logprobs_list, actor_qvalues_list = [], []
        critic_qvalues_0_list, critic_qvalues_1_list = [], []
        critic_valids_list, critic_positive_masks_list = [], []
        actor_loss_list, alpha_loss_list = [], []
        critic_loss_list = []
        critic_loss_metrics_agg: dict[str, list[float]] = defaultdict(list)
        actor_forward_metrics: dict[str, list[float]] = defaultdict(list)

        if skip_critic_update:
            critic_grad_norm = torch.tensor(0.0)
        else:
            self.critic_optimizer.zero_grad()
            for batch_idx, micro_batch in enumerate(critic_micro_batches):
                logger.info(f"[{batch_idx + 1}/{len(critic_micro_batches)}] critic micro batch ")
                micro_batch = micro_batch.to(get_device_id())
                raw_critic_loss, q_values_0, q_values_1, critic_loss_metrics = self._forward_critic(
                    micro_batch, resample=bool(self.actor_config.critic.resample_target_action)
                )
                (raw_critic_loss / critic_grad_accum_steps).backward()
                critic_loss_list.append(float(raw_critic_loss.detach().item()))
                for key, value in critic_loss_metrics.items():
                    critic_loss_metrics_agg[key].append(float(value.item()))
                critic_qvalues_0_list.append(q_values_0.mean(dim=-1).detach())
                critic_qvalues_1_list.append(q_values_1.detach())
                critic_valids_list.append(micro_batch.batch["info.valids"].detach())
                critic_positive_masks_list.append(micro_batch.batch["info.success_mask"].detach())
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.engine.module.sac_get_critic_parameters(), max_norm=self._critic_grad_clip
            )
            self.critic_optimizer.step()
            self.critic_scheduler.step()

        actor_replay_sample_info = {}
        actor_micro_batches = []
        actor_valids_list = []
        if update_actor:
            actor_batches, actor_replay_sample_info = self._sample_rlpd_batch(
                positive_sample_ratio=float(self.actor_config.replay.actor_positive_sample_ratio)
            )
            actor_micro_batches = [
                micro_batch
                for batch in actor_batches
                for micro_batch in batch.split(self.actor_config.micro_batch_size)
            ]
            actor_grad_accum_steps = len(actor_micro_batches)

            self.engine.optimizer_zero_grad()
            for batch_idx, micro_batch in enumerate(actor_micro_batches):
                logger.info(f"[{batch_idx + 1}/{len(actor_micro_batches)}] actor micro batch ")
                micro_batch = micro_batch.to(get_device_id())
                raw_actor_loss, log_probs, q_values, actor_forward_metrics_mb = self._forward_actor(
                    micro_batch,
                    is_first_micro_batch=(batch_idx == 0),
                )
                (raw_actor_loss / actor_grad_accum_steps).backward()
                actor_loss_list.append(raw_actor_loss.detach().item())
                if log_probs is not None:
                    actor_logprobs_list.append(log_probs.detach())
                actor_qvalues_list.append(q_values.detach())
                actor_valids_list.append(micro_batch.batch["info.valids"].detach())
                for key, value in actor_forward_metrics_mb.items():
                    actor_forward_metrics[key].append(float(value))
            actor_lr = float(self.engine.optimizer.param_groups[0]["lr"])
            actor_grad_norm = self.engine.optimizer_step()
            self.engine.lr_scheduler_step()
            self._update_actor_ema()
            self._apply_actor_ema_to_actor_module()

            if self.auto_entropy and actor_logprobs_list:
                self.alpha_optimizer.zero_grad()
                for micro_batch, log_probs in zip(actor_micro_batches, actor_logprobs_list, strict=False):
                    micro_batch = micro_batch.to(get_device_id())
                    raw_alpha_loss = self._calculate_alpha_loss(log_probs, micro_batch.batch["info.valids"])
                    (raw_alpha_loss / actor_grad_accum_steps).backward()
                    alpha_loss_list.append(raw_alpha_loss.detach().item())
                torch.distributed.all_reduce(self.raw_alpha.grad, op=torch.distributed.ReduceOp.AVG)
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_alpha, max_norm=self._actor_grad_clip)
                self.alpha_optimizer.step()
                self.alpha_scheduler.step()

        force_tau_one_in_warmup = bool(self.actor_config.critic.force_target_tau_one_in_warmup)
        critic_target_tau = (
            1.0
            if force_tau_one_in_warmup and global_steps < self.actor_config.critic.warmup_steps
            else float(self.actor_config.critic.tau)
        )
        if not skip_critic_update:
            self.engine.module.sac_update_target_network(critic_target_tau)

        if global_steps % self.actor_config.replay.save_interval == 0:
            self.replay_pool.save(self.actor_config.replay.save_dir)

        critic_rewards = torch.cat([batch.batch["info.rewards"] for batch in critic_batches])
        critic_valids = torch.cat([batch.batch["info.valids"] for batch in critic_batches])
        critic_positive_mask = (
            torch.cat(critic_positive_masks_list)
            if critic_positive_masks_list
            else torch.cat([batch.batch["info.success_mask"] for batch in critic_batches])
        ).to(torch.bool)
        critic_valid_mask = (torch.cat(critic_valids_list) if critic_valids_list else critic_valids).to(torch.bool)
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

        critic_grad_norm_pre_clip = critic_grad_norm.detach().item()
        critic_grad_norm_post_clip = self._post_clip_norm(critic_grad_norm_pre_clip, self._critic_grad_clip)

        metrics = {
            "data/reward_mean": valid_mean(critic_rewards, critic_valids).detach().item(),
            "data/valid_ratio": critic_valids.float().mean().item(),
            "sac/replay/online/sample_batch_size": self.online_replay_sample_batch_size,
            "sac/replay/online/pool_size": critic_replay_sample_info["online"]["pool_size"],
            "sac/replay/online/positive_size": critic_replay_sample_info["online"]["positive_size"],
            "sac/replay/online/negative_size": critic_replay_sample_info["online"]["negative_size"],
            "sac/replay/online/task_count": critic_replay_sample_info["online"]["task_count"],
            "sac/replay/online/critic_sample_count": critic_replay_sample_info["online"]["sample_count"],
            "sac/replay/online/critic_actual_positive_sample_ratio": critic_replay_sample_info["online"][
                "actual_positive_sample_ratio"
            ],
            "sac/replay/offline/sample_batch_size": self.offline_replay_sample_batch_size,
            "sac/replay/offline/pool_size": critic_replay_sample_info["offline"]["pool_size"],
            "sac/replay/offline/positive_size": critic_replay_sample_info["offline"]["positive_size"],
            "sac/replay/offline/negative_size": critic_replay_sample_info["offline"]["negative_size"],
            "sac/replay/offline/task_count": critic_replay_sample_info["offline"]["task_count"],
            "sac/replay/offline/critic_sample_count": critic_replay_sample_info["offline"]["sample_count"],
            "sac/replay/offline/critic_actual_positive_sample_ratio": critic_replay_sample_info["offline"][
                "actual_positive_sample_ratio"
            ],
            "sac/replay/online/actor_actual_positive_sample_ratio": actor_replay_sample_info["online"][
                "actual_positive_sample_ratio"
            ]
            if update_actor
            else 0.0,
            "sac/replay/online/actor_sample_count": actor_replay_sample_info["online"]["sample_count"]
            if update_actor
            else 0,
            "sac/replay/offline/actor_actual_positive_sample_ratio": actor_replay_sample_info["offline"][
                "actual_positive_sample_ratio"
            ]
            if update_actor
            else 0.0,
            "sac/replay/offline/actor_sample_count": actor_replay_sample_info["offline"]["sample_count"]
            if update_actor
            else 0,
            "sac/td3_enabled": float(self.td3_enabled),
            "sac/critic_only_update": float(critic_only_update),
            "sac/skip_critic_update_when_actor_update": float(self.skip_critic_update_when_actor_update),
            "sac/critic_update_skipped": float(skip_critic_update),
            **(
                {
                    "sac/cql_enabled": float(self.cql_enabled),
                    "sac/cql_alpha": self.cql_alpha,
                    "sac/cql_temperature": self.cql_temperature,
                    "sac/cql_noise_scale": -1.0 if self.cql_noise_scale is None else self.cql_noise_scale,
                }
                if self.cql_enabled
                else {}
            ),
            "sac/alpha": self._get_alpha().detach().item(),
            "sac/actor_ema_enabled": float(self.actor_ema_enabled),
            "sac/actor_ema_decay": self.actor_ema_decay,
            "sac/critic_target_tau": critic_target_tau,
            "critic/loss": sum(critic_loss_list) / len(critic_loss_list) if critic_loss_list else 0.0,
            "critic/lr": self.critic_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": critic_grad_norm_post_clip,
            "critic/grad_norm_pre_clip": critic_grad_norm_pre_clip,
            "critic/grad_clip_threshold": self._critic_grad_clip,
            "critic/qvalue0_mean": (
                valid_mean(torch.cat(critic_qvalues_0_list), torch.cat(critic_valids_list)).detach().item()
                if critic_qvalues_0_list
                else 0.0
            ),
            "critic/qvalue1_mean": (
                valid_mean(torch.cat(critic_qvalues_1_list), torch.cat(critic_valids_list)).detach().item()
                if critic_qvalues_1_list
                else 0.0
            ),
            "critic/positive_qvalue_mean": positive_qvalue_mean,
            "critic/negative_qvalue_mean": negative_qvalue_mean,
            "critic/diff_pos_neg_qvalue_mean": positive_qvalue_mean - negative_qvalue_mean,
        }
        metrics.update(
            {
                (key if key.startswith("cql/") else f"critic/{key}"): (
                    max(values) if key == "cql/q_gap_max" else sum(values) / len(values)
                )
                for key, values in critic_loss_metrics_agg.items()
                if len(values) > 0
            }
        )
        if update_actor:
            actor_grad_norm_post_clip = self._post_clip_norm(actor_grad_norm, self._actor_grad_clip)
            actor_valids = torch.cat(actor_valids_list)
            metrics.update(
                {
                    "actor/loss": sum(actor_loss_list) / len(actor_loss_list),
                    "actor/lr": actor_lr,
                    "actor/grad_norm": actor_grad_norm_post_clip,
                    "actor/grad_norm_pre_clip": actor_grad_norm,
                    "actor/grad_clip_threshold": self._actor_grad_clip,
                    "actor/logprob_mean": (
                        valid_mean(torch.cat(actor_logprobs_list), actor_valids).detach().item()
                        if actor_logprobs_list
                        else 0.0
                    ),
                    "actor/qvalue_mean": valid_mean(torch.cat(actor_qvalues_list), actor_valids).detach().item(),
                    "sac/alpha_lr": self.alpha_optimizer.param_groups[0]["lr"]
                    if self.auto_entropy and actor_logprobs_list
                    else 0.0,
                    "sac/alpha_loss": sum(alpha_loss_list) / len(alpha_loss_list)
                    if self.auto_entropy and alpha_loss_list
                    else 0.0,
                    "sac/alpha_grad_norm": self._post_clip_norm(alpha_grad_norm, self._actor_grad_clip)
                    if self.auto_entropy and actor_logprobs_list
                    else 0.0,
                    "sac/alpha_grad_norm_pre_clip": alpha_grad_norm.detach().item()
                    if self.auto_entropy and actor_logprobs_list
                    else 0.0,
                    "sac/alpha_grad_clip_threshold": self._actor_grad_clip
                    if self.auto_entropy and actor_logprobs_list
                    else 0.0,
                }
            )
            metrics.update(
                {
                    f"actor/{k}": sum(values) / len(values)
                    for k, values in actor_forward_metrics.items()
                    if len(values) > 0
                }
            )

        return metrics

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: DataProto) -> DataProto:
        self._ensure_sac_initialized()
        with self.engine.train_mode():
            metrics = self._update_sac_policy(data)
        return DataProto(meta_info={"metrics": metrics})
