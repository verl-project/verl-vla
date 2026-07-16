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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from torch import nn
from verl import DataProto


class ModelOutput(ABC):
    @abstractmethod
    def to_data_proto(self) -> DataProto:
        """Convert the model output to a DataProto format for downstream processing."""
        pass


class SupportSACTraining:
    """
    Base class for Soft Actor-Critic (SAC).

    Subclasses implement a Policy that can be plugged directly into SAC training.
    This implementation requires the actor and critic to be integrated within a
    single model instance, e.g., sharing a backbone with an additional MLP head
    that outputs critic values (Q/V) alongside the actor's action distribution.

    Note:
        This class intentionally does NOT inherit from `abc.ABC`.
        The root model may be wrapped or transformed by FSDP (Fully Sharded
        Data Parallel), which performs runtime class substitution; using
        `ABCMeta` can break FSDP's class rewriting mechanism.
    """

    def sac_init(self):
        raise NotImplementedError("Subclasses must implement sac_init method.")

    def sac_sample_actions(
        self,
        obs: DataProto,
        tokenizer: Optional[torch.nn.Module] = None,
        eval: bool = False,
    ) -> ModelOutput:
        raise NotImplementedError("Subclasses must implement sac_sample_actions method.")

    def sac_get_critic_value(
        self,
        obs: DataProto,
        actions: ModelOutput,
        tokenizer: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement sac_get_critic_value method.")

    def sac_get_critic_parameters(self) -> list[torch.nn.Parameter]:
        """Get the parameters of the critic head for optimization.

        Returns:
            A list of torch.nn.Parameter objects representing the critic head parameters.
        """

        raise NotImplementedError("Subclasses must implement sac_get_critic_parameters method.")

    def sac_get_named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        """Get named actor parameters for optimization/EMA updates.

        Returns:
            A list of (name, parameter) tuples representing actor-side trainable parameters.
        """

        raise NotImplementedError("Subclasses must implement sac_get_named_actor_parameters method.")

    def sac_forward_critic(
        self,
        a: dict[str, torch.Tensor],
        state_features: Any,
        task_ids: Optional[torch.Tensor] = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Compute Q-values for given state-action pairs.
        Args:
            a: Dictionary of tensors representing actions
            state_features: Any data structure representing the processed state features.
            task_ids: Optional tensor of shape (B,) for task-conditioned critic routing.
            use_target_network: Whether to use the target critic network heads.
            method: Method to combine multiple heads' outputs ("cat" or "min").
            requires_grad: Whether to enable gradients for the critic head parameters.

        Returns:
            q_values: torch.Tensor of shape (B, num_heads) if method is "cat",
                      or (B, 1) if method is "min", representing the computed Q-values
        """

        raise NotImplementedError("Subclasses must implement sac_forward_critic method.")

    def sac_forward_actor(
        self,
        state_features: Any,
        task_ids: Optional[torch.Tensor] = None,
        is_first_micro_batch: bool = False,
        noise_scale: Optional[float] = None,
    ) -> Any:
        """Compute actions and their log probabilities from state features.

        Args:
            state_features: Any data structure representing the processed state features.
            task_ids: Optional tensor of shape (B,) selecting task-specific actor sampling
                behavior, such as task-specific Flow-SDE noise levels.
            is_first_micro_batch: Whether the current forward corresponds to the first
                micro batch of the actor update step.
            noise_scale: Optional Flow-SDE noise scale override. When unset, subclasses use
                their default training noise scale.

        Returns:
            actions: torch.Tensor of shape (B, n_action_steps, action_dim), sampled actions.
            log_probs: Optional torch.Tensor of shape (B,), log probabilities of sampled actions.
                Can be None when SAC is configured to train without entropy/log-prob terms.
            metrics: Scalar metrics produced by actor forward, used by outer trainer for logging.
        """

        raise NotImplementedError("Subclasses must implement sac_forward_actor method.")

    def sac_forward_state_features(self, obs: DataProto, tokenizer: torch.nn.Module) -> Any:
        """Compute state features needed for SAC actor and critic.

        Args:
            obs: DataProto containing the observations
        Returns:
            state_features: Any data structure representing the processed state features.
        """

        raise NotImplementedError("Subclasses must implement sac_forward_state_features method.")

    def sac_update_target_network(self, tau: float):
        """Update the target network heads using Polyak averaging.

        Args:
            tau: The interpolation parameter for Polyak averaging.
        """

        raise NotImplementedError("Subclasses must implement sac_update_target_network method.")


class SupportSFTTraining:
    """
    Base class for models that expose one unified SFT loss interface.

    This intentionally does NOT inherit from `abc.ABC` because model classes may
    be wrapped or rewritten by FSDP at runtime.
    """

    def __init__(self, config: Any):
        self.config = config
        self.sft_metrics: dict[str, torch.Tensor] = {}

    def sft_init(self):
        self.sft_metrics = {}
        try:
            from torch.distributed.fsdp import register_fsdp_forward_method
        except ImportError:
            return
        register_fsdp_forward_method(self, "sft_loss")

    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        target_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del obs, tokenizer, actions, valids, action_mask, target_values
        raise NotImplementedError("Subclasses must implement sft_loss method.")


class TrainableVLAModelMixin:
    """Shared boundary for trainable adapters that contain one native policy."""

    policy: nn.Module

    def init_trainable_model(self, *, policy: nn.Module) -> None:
        if not isinstance(self, nn.Module):
            raise TypeError("TrainableVLAModelMixin requires an nn.Module trainable model")
        self.policy = policy

    @property
    def native_policy(self) -> nn.Module:
        """Return the upstream policy underneath an optional PEFT wrapper."""

        from peft import PeftModel

        if isinstance(self.policy, PeftModel):
            return self.policy.get_base_model()
        return self.policy

    @property
    def has_lora(self) -> bool:
        """Whether the native policy currently carries a PEFT LoRA adapter."""

        from peft import PeftModel

        return isinstance(self.policy, PeftModel)

    def apply_lora(
        self,
        *,
        rank: int,
        alpha: int,
        target_modules: str | list[str] | None,
        target_parameters: list[str] | None = None,
        exclude_modules: str | list[str] | None = None,
        adapter_path: str | None = None,
    ) -> None:
        """Attach a trainable PEFT LoRA adapter to the upstream policy only."""

        from peft import LoraConfig, PeftModel, get_peft_model

        if self.has_lora:
            raise RuntimeError("The VLA policy already has a LoRA adapter")

        if adapter_path is not None:
            self.policy = PeftModel.from_pretrained(self.policy, adapter_path, is_trainable=True)
            adapter_rank = int(self.policy.peft_config["default"].r)
            if adapter_rank != rank:
                raise ValueError(f"Configured LoRA rank {rank} does not match adapter rank {adapter_rank}")
            return

        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.policy = get_peft_model(
            self.policy,
            LoraConfig(
                task_type=None,
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                target_parameters=target_parameters,
                exclude_modules=exclude_modules,
                bias="none",
            ),
        )

    @staticmethod
    def extract_policy_state_dict(
        state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Extract native policy weights from a full adapter state dict."""

        prefix = "policy."
        extracted = {name.removeprefix(prefix): value for name, value in state_dict.items() if name.startswith(prefix)}
        return extracted or dict(state_dict)

    def export_policy(
        self,
        output_dir: str | Path,
        *,
        state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Export the upstream policy using its native ``save_pretrained`` API."""

        save_pretrained = getattr(self.native_policy, "save_pretrained", None)
        if not callable(save_pretrained):
            raise TypeError(f"{type(self.native_policy).__name__} does not implement save_pretrained()")

        policy_state_dict = None
        if state_dict is not None:
            policy_state_dict = self.extract_policy_state_dict(state_dict)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if policy_state_dict is None:
            save_pretrained(output_dir)
        else:
            save_pretrained(output_dir, state_dict=policy_state_dict)
