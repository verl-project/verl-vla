# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .base import CriticBackend
from .critic_cross_attn import CrossAttentionCriticGroup


class MultiCrossAttentionCritic(nn.Module):
    def __init__(self, head_num: int, attn_heads: int, critic_num: int, task_to_critic: dict[int, int] | None = None):
        super().__init__()
        self.critic_num = critic_num
        self.task_to_critic = task_to_critic
        self.critics = nn.ModuleList(
            [CrossAttentionCriticGroup(head_num=head_num, attn_heads=attn_heads) for _ in range(critic_num)]
        )

    def _resolve_critic_ids(self, task_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        task_ids = task_ids.to(dtype=torch.long).reshape(-1)
        if task_ids.shape[0] != batch_size:
            raise ValueError(f"task_ids batch size {task_ids.shape[0]} does not match critic batch size {batch_size}.")

        if self.task_to_critic is None:
            return torch.remainder(task_ids, self.critic_num)

        critic_ids = torch.empty_like(task_ids)
        for idx, task_id in enumerate(task_ids.tolist()):
            if task_id not in self.task_to_critic:
                raise ValueError(f"task_id {task_id} is not configured in critic_task_to_critic.")
            critic_ids[idx] = int(self.task_to_critic[task_id])

        if torch.any(critic_ids < 0) or torch.any(critic_ids >= self.critic_num):
            raise ValueError(f"critic_task_to_critic produced invalid critic ids for critic_num={self.critic_num}.")
        return critic_ids

    @staticmethod
    def _select_state_features(state_features, mask: torch.Tensor):
        prefix_features, states = state_features
        prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_features
        return (
            (prefix_embs[mask], prefix_pad_masks[mask], prefix_att_masks[mask]),
            states[mask],
        )

    def forward(
        self,
        a: dict[str, torch.Tensor],
        state_features: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        task_ids: torch.Tensor,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        prefix_features, states = state_features
        prefix_embs, _, _ = prefix_features
        batch_size = prefix_embs.shape[0]
        critic_ids = self._resolve_critic_ids(task_ids.to(prefix_embs.device), batch_size=batch_size)

        head_num = len(self.critics[0].critic_heads)
        if method == "cat":
            q_values = torch.empty(batch_size, head_num, device=states.device, dtype=states.dtype)
        elif method == "min":
            q_values = torch.empty(batch_size, device=states.device, dtype=states.dtype)
        else:
            raise ValueError(f"Unknown method: {method}")

        dummy_zero = q_values.new_zeros(())
        for critic_id in range(self.critic_num):
            mask = critic_ids == critic_id
            has_samples = torch.any(mask)
            critic = self.critics[int(critic_id)]
            if has_samples:
                group_state_features = self._select_state_features(state_features, mask)
                group_actions = {"action": a["action"][mask]}
            else:
                group_state_features = self._select_state_features(state_features, slice(0, 1))
                group_actions = {"action": a["action"][:1]}

            group_q_values = critic(
                a=group_actions,
                state_features=group_state_features,
                use_target_network=use_target_network,
                method=method,
                requires_grad=requires_grad,
            )
            if has_samples:
                q_values[mask] = group_q_values.to(dtype=q_values.dtype)
            else:
                dummy_zero = dummy_zero + group_q_values.to(dtype=q_values.dtype).sum() * 0.0

        return q_values + dummy_zero

    def get_critic_parameters(self) -> list[torch.nn.Parameter]:
        return [param for critic in self.critics for param in critic.get_critic_parameters()]

    @torch.no_grad()
    def update_target_network(self, tau: float) -> None:
        for critic in self.critics:
            critic.update_target_network(tau)


def _normalize_task_to_critic(raw_mapping) -> dict[int, int] | None:
    if raw_mapping is None:
        return None
    mapping = {}
    for key, value in dict(raw_mapping).items():
        mapping[int(key)] = int(value)
    return mapping


class MultiCrossAttentionCriticBackend(CriticBackend):
    uses_task_ids = True

    def init(self, model) -> None:
        head_num = int(getattr(model.config, "critic_head_num", 2))
        attn_heads = int(getattr(model.config, "critic_prefix_attn_heads", 8))
        critic_num = int(getattr(model.config, "critic_num", 1))
        task_to_critic = _normalize_task_to_critic(getattr(model.config, "critic_task_to_critic", None))
        model.critic_backend = MultiCrossAttentionCritic(
            head_num=head_num,
            attn_heads=attn_heads,
            critic_num=critic_num,
            task_to_critic=task_to_critic,
        )

    def forward(
        self,
        model,
        a: dict[str, torch.Tensor],
        state_features: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        task_ids: torch.Tensor | None = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        if task_ids is None:
            raise ValueError("MultiCrossAttentionCriticBackend requires task_ids.")
        return model.critic_backend(
            a=a,
            state_features=state_features,
            task_ids=task_ids,
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    def get_critic_parameters(self, model) -> list[torch.nn.Parameter]:
        return model.critic_backend.get_critic_parameters()

    @torch.no_grad()
    def update_target_network(self, model, tau: float) -> None:
        model.critic_backend.update_target_network(tau)
