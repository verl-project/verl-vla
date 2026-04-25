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

from ....utils.models.mlp import MLP
from .base import CriticBackend


class CrossAttentionCriticGroup(nn.Module):
    def __init__(self, head_num: int, attn_heads: int):
        super().__init__()
        self.critic_state_token = nn.Parameter(torch.zeros(1, 1, 2048))
        self.target_state_token = nn.Parameter(torch.zeros(1, 1, 2048))
        nn.init.normal_(self.critic_state_token, mean=0.0, std=0.02)
        self.target_state_token.data.copy_(self.critic_state_token.data)

        self.critic_prefix_cross_attn = nn.MultiheadAttention(
            embed_dim=2048,
            num_heads=attn_heads,
            batch_first=True,
        )
        self.target_prefix_cross_attn = nn.MultiheadAttention(
            embed_dim=2048,
            num_heads=attn_heads,
            batch_first=True,
        )

        self.critic_heads = nn.ModuleList(
            [
                MLP(
                    input_dim=2150,
                    hidden_dims=[2048, 1024, 256],
                    output_dim=1,
                    activation="relu",
                    init_method="kaiming",
                )
                for _ in range(head_num)
            ]
        )
        self.target_network_heads = nn.ModuleList(
            [
                MLP(
                    input_dim=2150,
                    hidden_dims=[2048, 1024, 256],
                    output_dim=1,
                    activation="relu",
                    init_method="kaiming",
                )
                for _ in range(head_num)
            ]
        )
        self.target_network_heads.load_state_dict(self.critic_heads.state_dict())
        self.target_prefix_cross_attn.load_state_dict(self.critic_prefix_cross_attn.state_dict())

    @staticmethod
    def _multi_heads_value(
        value_heads: nn.ModuleList,
        input_tensor: torch.Tensor,
        method: Literal["cat", "min"] = "cat",
    ) -> torch.Tensor:
        q_values = [head(input_tensor) for head in value_heads]
        if method == "cat":
            return torch.cat(q_values, dim=-1)
        if method == "min":
            return torch.min(torch.cat(q_values, dim=-1), dim=-1).values
        raise ValueError(f"Unknown method: {method}")

    def _cross_attention_pool_prefix(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        use_target_network: bool,
    ) -> torch.Tensor:
        cross_attn = self.target_prefix_cross_attn if use_target_network else self.critic_prefix_cross_attn
        state_token = self.target_state_token if use_target_network else self.critic_state_token

        batch_size = prefix_embs.shape[0]
        query = state_token.expand(batch_size, -1, -1)
        key_padding_mask = ~prefix_pad_masks.to(dtype=torch.bool)

        pooled, _ = cross_attn(
            query=query,
            key=prefix_embs,
            value=prefix_embs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return pooled.squeeze(1)

    def forward(
        self,
        a: dict[str, torch.Tensor],
        state_features: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        critic_head = self.target_network_heads if use_target_network else self.critic_heads
        for p in critic_head.parameters():
            p.requires_grad_(requires_grad)
        prefix_cross_attn = self.target_prefix_cross_attn if use_target_network else self.critic_prefix_cross_attn
        for p in prefix_cross_attn.parameters():
            p.requires_grad_(requires_grad)
        (self.target_state_token if use_target_network else self.critic_state_token).requires_grad_(requires_grad)

        prefix_features, states = state_features
        prefix_embs, prefix_pad_masks, _ = prefix_features
        pooled_prefix_embs = self._cross_attention_pool_prefix(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            use_target_network=use_target_network,
        )
        actions = a["action"]
        flattened_actions = actions.reshape(actions.shape[0], -1)
        critic_input = torch.cat([pooled_prefix_embs, states, flattened_actions], dim=-1)
        return self._multi_heads_value(critic_head, critic_input, method=method)

    def get_critic_parameters(self) -> list[torch.nn.Parameter]:
        critic_head_params = [p for head in self.critic_heads for p in head.parameters()]
        critic_prefix_cross_attn_params = list(self.critic_prefix_cross_attn.parameters())
        return critic_head_params + critic_prefix_cross_attn_params + [self.critic_state_token]

    @torch.no_grad()
    def update_target_network(self, tau: float) -> None:
        for t_head, head in zip(self.target_network_heads, self.critic_heads, strict=True):
            t_sd = t_head.state_dict()
            h_sd = head.state_dict()
            for k in t_sd.keys():
                t_sd[k].mul_(1.0 - tau).add_(h_sd[k], alpha=tau)
            t_head.load_state_dict(t_sd, strict=True)

        t_cross_attn_sd = self.target_prefix_cross_attn.state_dict()
        cross_attn_sd = self.critic_prefix_cross_attn.state_dict()
        for k in t_cross_attn_sd.keys():
            t_cross_attn_sd[k].mul_(1.0 - tau).add_(cross_attn_sd[k], alpha=tau)
        self.target_prefix_cross_attn.load_state_dict(t_cross_attn_sd, strict=True)

        self.target_state_token.data.mul_(1.0 - tau).add_(self.critic_state_token.data, alpha=tau)


class CrossAttentionCriticBackend(CriticBackend):
    uses_task_ids = False

    def init(self, model) -> None:
        head_num = int(getattr(model.config, "critic_head_num", 2))
        attn_heads = int(getattr(model.config, "critic_prefix_attn_heads", 8))
        model.critic_backend = CrossAttentionCriticGroup(head_num=head_num, attn_heads=attn_heads)

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
        del task_ids
        return model.critic_backend(
            a=a,
            state_features=state_features,
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    def get_critic_parameters(self, model) -> list[torch.nn.Parameter]:
        return model.critic_backend.get_critic_parameters()

    @torch.no_grad()
    def update_target_network(self, model, tau: float) -> None:
        model.critic_backend.update_target_network(tau)
