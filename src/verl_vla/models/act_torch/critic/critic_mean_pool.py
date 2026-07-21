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


class MeanPoolCriticGroup(nn.Module):
    def __init__(self, head_num: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.critic_heads = nn.ModuleList(
            [
                MLP(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
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
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=1,
                    activation="relu",
                    init_method="kaiming",
                )
                for _ in range(head_num)
            ]
        )
        self.target_network_heads.load_state_dict(self.critic_heads.state_dict())

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

    @staticmethod
    def _mean_pool_prefix(prefix_embs: torch.Tensor, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        prefix_mask = prefix_pad_masks.to(dtype=prefix_embs.dtype).unsqueeze(-1)
        masked_prefix = prefix_embs * prefix_mask
        return masked_prefix.sum(dim=1) / prefix_mask.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        a: dict[str, torch.Tensor],
        state_features: tuple[torch.Tensor, torch.Tensor],
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        critic_head = self.target_network_heads if use_target_network else self.critic_heads
        for p in critic_head.parameters():
            p.requires_grad_(requires_grad)

        prefix_embs, states = state_features
        pooled_prefix_embs = prefix_embs.mean(dim=1)

        actions = a["action"]
        flattened_actions = actions.reshape(actions.shape[0], -1)
        critic_input = torch.cat([pooled_prefix_embs, states, flattened_actions], dim=-1)
        expected_input_dim = self.critic_heads[0].network[0].in_features
        if critic_input.shape[-1] != expected_input_dim:
            raise ValueError(
                f"ACT mean-pool critic input dim mismatch: got {critic_input.shape[-1]}, "
                f"expected {expected_input_dim}. Check critic_input_dim, n_action_steps, action_dim, and state_dim."
            )
        return self._multi_heads_value(critic_head, critic_input, method=method)

    def get_critic_parameters(self) -> list[torch.nn.Parameter]:
        return [p for head in self.critic_heads for p in head.parameters()]

    @torch.no_grad()
    def update_target_network(self, tau: float) -> None:
        for t_head, head in zip(self.target_network_heads, self.critic_heads, strict=True):
            t_sd = t_head.state_dict()
            h_sd = head.state_dict()
            for k in t_sd.keys():
                t_sd[k].mul_(1.0 - tau).add_(h_sd[k], alpha=tau)
            t_head.load_state_dict(t_sd, strict=True)


class MeanPoolCriticBackend(CriticBackend):
    uses_task_ids = False

    def init(self, model) -> None:
        head_num = int(getattr(model.config, "critic_head_num", 2))
        input_dim = int(getattr(model.config, "critic_input_dim", 0))
        if input_dim <= 0:
            input_dim = (
                int(getattr(model.config, "critic_prefix_embed_dim", 512))
                + int(getattr(model.config, "state_dim", 0))
                + int(getattr(model.config, "n_action_steps", 1)) * int(getattr(model.config, "action_dim", 0))
            )
            model.config.critic_input_dim = input_dim
        hidden_dims = [int(dim) for dim in getattr(model.config, "critic_hidden_dims", [1024, 512, 256])]
        critic_backend = MeanPoolCriticGroup(
            head_num=head_num,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        )
        object.__setattr__(model, "critic_backend", critic_backend)

    def forward(
        self,
        model,
        a: dict[str, torch.Tensor],
        state_features: tuple[torch.Tensor, torch.Tensor],
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
