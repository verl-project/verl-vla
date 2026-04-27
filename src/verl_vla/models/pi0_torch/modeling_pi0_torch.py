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

import math
from typing import Literal, cast

import torch
import torch.nn.functional as F
from onnx_ir import Tensor
from torch import nn
from torch.distributed.fsdp import register_fsdp_forward_method
from transformers import PreTrainedModel
from typing_extensions import override
from verl.protocol import DataProto
from verl.utils.device import get_device_name

from verl_vla.utils.scalar_schedule import ScheduledScalar

from ..base import ModelOutput, SupportSACTraining, SupportSFTTraining
from .configuration_pi0_torch import PI0TorchConfig
from .critic import (
    CrossAttentionCriticBackend,
    MultiCrossAttentionCriticBackend,
)
from .model.modeling_pi0 import PI0Model, make_att_2d_masks
from .pi0_utils import (
    ImageTransform,
    Normalize,
    PromptTokenizerTransform,
    Unnormalize,
)
from .policy import get_pi0_policy_classes
from .policy.base import Pi0Output


def beta_schedule(step, beta0, beta_min, T):
    progress = min(step / T, 1.0)
    beta = beta_min + (beta0 - beta_min) * 0.5 * (1 + math.cos(math.pi * progress))
    return beta


CRITIC_BACKENDS = {
    "cross_attn": CrossAttentionCriticBackend(),
    "multi_cross_attn": MultiCrossAttentionCriticBackend(),
}


class PI0ForActionPrediction(PreTrainedModel, SupportSACTraining, SupportSFTTraining):
    config_class = PI0TorchConfig
    base_model_prefix = "pi0_torch"

    def __init__(self, config: PI0TorchConfig):
        super().__init__(config)
        self.model: PI0Model = None
        self.state_norm_stats = config.state_norm_stats
        self.action_norm_stats = config.action_norm_stats
        self.pi05_enabled = config.pi05_enabled
        self.policy_type = config.policy_type
        self.critic_type = getattr(config, "critic_type", "cross_attn")

        assert self.state_norm_stats, "state_norm_stats must be provided in PI0TorchConfig"
        assert self.action_norm_stats, "action_norm_stats must be provided in PI0TorchConfig"
        assert isinstance(self.pi05_enabled, bool), "pi05_enabled must be provided in PI0TorchConfig"

        # Input transforms
        self.state_normalize_transform = Normalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_normalize_transform = Normalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)
        self.image_transform = ImageTransform(resize_imgs_with_padding=(224, 224), enable_image_aug=False)
        max_length = 200 if self.pi05_enabled else 48
        self.prompt_tokenizer_transform = PromptTokenizerTransform(max_length=max_length, discrete_state_input=False)

        # Output transforms
        self.state_unnormalize_transform = Unnormalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)

        # Flow SDE parameters
        self._to(get_device_name())
        self.flow_sde_enable = bool(getattr(config, "flow_sde_enable", True))
        self.flow_sde_noise_level = float(getattr(config, "flow_sde_noise_level", 0.5))
        self.flow_sde_noise_scheduler = ScheduledScalar(
            base_value=self.flow_sde_noise_level,
            enabled=bool(getattr(config, "flow_sde_noise_schedule_enabled", False)),
            initial_value=getattr(config, "flow_sde_noise_schedule_initial", None),
            final_value=getattr(config, "flow_sde_noise_schedule_final", None),
            method=getattr(config, "flow_sde_noise_schedule_method", "cos"),
            clamp_min=0.0,
            clamp_max=None,
        )
        self.flow_sde_task_noise_level = self._parse_task_noise_levels(config.flow_sde_task_noise_level)
        self.flow_sde_rollout_noise_scale = float(getattr(config, "flow_sde_rollout_noise_scale", 1.0))
        self.flow_sde_train_noise_scale = float(getattr(config, "flow_sde_train_noise_scale", 1.0))
        self.flow_sde_initial_beta = float(getattr(config, "flow_sde_initial_beta", 1.0))
        self.flow_sde_beta_min = float(getattr(config, "flow_sde_beta_min", 0.02))
        self.flow_sde_beta_schedule_T = int(getattr(config, "flow_sde_beta_schedule_T", 2000))
        self.register_buffer("flow_sde_step", torch.zeros((), dtype=torch.long))

        ##### SAC Algorithm Support #####
        if getattr(self.config, "sac_enable", False):
            if self.critic_type not in CRITIC_BACKENDS:
                raise ValueError(f"Unsupported critic_type: {self.critic_type}")
            self.critic_api = CRITIC_BACKENDS[self.critic_type]
            self.critic_api.init(self)

    def _get_pi0_policy_classes(self):
        return get_pi0_policy_classes(self.policy_type)

    def _to(self, device: torch.device | str):
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_normalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        return self

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tensor:
        """Full forward pass for one diffusion denoising step.

        Args:
            images: List of image tensors, each shaped (B, C, H, W) after batching.
            img_masks: List of boolean masks corresponding to images, each (B,).
            lang_tokens: Language token ids (B, L).
            lang_masks: Language attention mask (B, L) with True for valid tokens.
            state: State tensor (B, state_dim) if pi05 is disabled else ignored.
            x_t: Noisy action tokens (B, n_action_steps, action_dim).
            timestep: Diffusion timestep as float tensor (B,).

        Returns:
            Predicted v_t with shape (B, n_action_steps, action_dim).
        """

        if self.model is None:
            raise RuntimeError("PI0ForActionPrediction.model is not initialized. Did from_pretrained() run?")

        return self.model(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            x_t,
            timestep,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        env_obs: DataProto,
        tokenizer,
        validate: bool = False,
    ) -> tuple[Pi0Output, dict, dict]:
        """Run one forward pass from raw inputs to final action sequence.

        Args:
            env_obs: The environment observations as DataProto.
            tokenizer: The tokenizer used for prompt tokenization.

        Returns:
            A tuple of (pi0_output, s, a):
                - pi0_output: The Pi0Output containing the predicted actions.
                - s: Dictionary of tensors representing the states, with keys
                    - "images": torch.Tensor of shape (B, n_images, C, H, W)
                    - "image_masks": torch.Tensor of shape (B, n_images)
                    - "lang_tokens": torch.Tensor of shape (B, L)
                    - "lang_masks": torch.Tensor of shape (B, L)
                    - "states": torch.Tensor of shape (B, state_dim)
                - a: Dictionary of tensors representing actions, with key:
                    - "full_action": torch.Tensor of shape (B, action_steps, action_dim)
        """

        pi0_input_cls, pi0_output_cls = self._get_pi0_policy_classes()
        pi0_input = pi0_input_cls.from_env_obs(env_obs)

        # Input transforms
        state = self.state_normalize_transform(pi0_input.state)
        images, _ = self.image_transform.call_batch(pi0_input.images)
        lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
            {"task": pi0_input.task, "observation.state": state}, tokenizer
        )

        if self.flow_sde_enable and not validate:
            prefix_features = self.model.embed_prefix(
                images=images,
                img_masks=pi0_input.img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
            )
            pred_action, rollout_log_probs = self._sample_actions_flow_sde(
                state_features=(prefix_features, state),
                noise_scale=self.flow_sde_rollout_noise_scale,
                requires_grad=False,
                return_log_prob=True,
                task_ids=torch.tensor(env_obs.non_tensor_batch["task_ids"], device=state.device, dtype=torch.long),
            )
        else:
            pred_action = self.model.sample_actions(images, pi0_input.img_masks, lang_tokens, lang_masks, state=state)
            rollout_log_probs = torch.zeros(pred_action.shape[0], device=pred_action.device, dtype=torch.float32)

        # Output transforms
        pi0_output = pi0_output_cls.from_model_output(
            {
                "full_action": self.action_unnormalize_transform(pred_action),
                "log_probs": rollout_log_probs,
            }
        )
        s = {
            "states": state,
            "images": torch.stack(images, dim=1),
            "image_masks": torch.stack(pi0_input.img_masks, dim=1),
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }
        a = {
            "full_action": pred_action,
            "log_probs": rollout_log_probs,
        }

        return pi0_output, s, a

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            config = PI0TorchConfig.from_pretrained(pretrained_model_name_or_path)

        policy = cls(config)
        policy.model = PI0Model.from_pretrained(pretrained_model_name_or_path)
        return policy

    # def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
    #     filtered_state_dict = {key: value for key, value in state_dict.items() if key.startswith("model.")}
    #     return super().load_state_dict(filtered_state_dict, strict=False, assign=assign)

    def freeze_vision_tower(self) -> None:
        """Freeze the vision tower parameters."""

        if self.model is None:
            raise RuntimeError("PI0ForActionPrediction.model is not initialized. Did from_pretrained() run?")
        vision_tower = self.model.paligemma_with_expert.vision_tower
        vision_tower.requires_grad_(False)
        vision_tower.eval()

    def sft_init(self):
        """Initialize the model for supervised fine-tuning."""
        self.freeze_vision_tower()
        register_fsdp_forward_method(self, "bc_loss")

    def bc_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the BC loss for the actor."""

        pi0_input_cls, pi0_output_cls = self._get_pi0_policy_classes()
        action_tensor = pi0_output_cls.from_model_output(
            {
                "full_action": actions["full_action"],
                "log_probs": torch.zeros(actions["full_action"].shape[0], device=actions["full_action"].device),
            }
        ).action
        action_tensor = torch.nn.functional.pad(
            action_tensor,
            (
                0,
                self.model.max_action_dim - action_tensor.shape[-1],
                0,
                self.model.n_action_steps - action_tensor.shape[1],
            ),
            value=0.0,
        )
        action_tensor = self.action_normalize_transform(action_tensor)

        with torch.no_grad():
            pi0_input = pi0_input_cls.from_env_obs(obs)
            states = self.state_normalize_transform(pi0_input.state)
            images, _ = self.image_transform.call_batch(pi0_input.images)
            img_masks = pi0_input.img_masks
            lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
                {"task": pi0_input.task, "observation.state": states}, tokenizer
            )

        noise = self.model.sample_noise(action_tensor.shape, device=action_tensor.device)
        gamma1 = torch.empty((action_tensor.shape[0],), device=action_tensor.device).uniform_(0, 1).pow(1 / 1.5)
        gamma2 = torch.empty((action_tensor.shape[0],), device=action_tensor.device).uniform_(0, 1).pow(1 / 1.0)
        time = (gamma1 / (gamma1 + gamma2)) * 0.999 + 0.001
        time = time.to(dtype=torch.float32, device=action_tensor.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * action_tensor
        u_t = noise - action_tensor

        model_pred = self.model(images, img_masks, lang_tokens, lang_masks, states, x_t, time)
        action_loss_mask = (
            (torch.arange(self.model.n_action_steps, device=model_pred.device) < 10)
            .to(model_pred.dtype)
            .unsqueeze(0)
            .expand(model_pred.shape[0], -1)
        )
        loss = F.mse_loss(u_t, model_pred, reduction="none").mean(dim=-1)
        loss = loss * action_loss_mask
        return loss.mean()

    # --- SAC Algorithm Support ---

    def _gaussian_log_prob(
        self,
        sample: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        std_safe = std.clamp_min(1e-6)
        log_prob = -0.5 * (((sample - mean) / std_safe) ** 2 + 2.0 * torch.log(std_safe) + math.log(2.0 * math.pi))
        return log_prob.mean(dim=(-1, -2))

    def flow_sde_beta(self) -> torch.Tensor:
        beta = beta_schedule(
            int(self.flow_sde_step.item()),
            beta0=self.flow_sde_initial_beta,
            beta_min=self.flow_sde_beta_min,
            T=self.flow_sde_beta_schedule_T,
        )
        return torch.tensor(beta, device=self.flow_sde_step.device, dtype=torch.float32)

    def _parse_task_noise_levels(
        self,
        task_noise_levels: str,
    ) -> dict[int, ScheduledScalar]:
        normalized: dict[int, ScheduledScalar] = {}
        if not task_noise_levels:
            return normalized
        for item in task_noise_levels.split(","):
            task_id, noise_level = item.split(":", 1)
            normalized_task_id = int(task_id)
            normalized_noise_level = float(noise_level)
            if normalized_noise_level < 0:
                raise ValueError(
                    f"flow_sde_task_noise_level[{normalized_task_id}] must be non-negative, "
                    f"got {normalized_noise_level}."
                )
            normalized[normalized_task_id] = ScheduledScalar(
                base_value=normalized_noise_level,
                method=self.flow_sde_noise_scheduler.method,
                clamp_min=0.0,
                clamp_max=None,
            )
        return normalized

    def _flow_sde_noise_control_value(self) -> float:
        return min(float(self.flow_sde_step.item()) / max(1.0, float(self.flow_sde_beta_schedule_T)), 1.0)

    def _resolve_flow_sde_noise_levels(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise_control = self._flow_sde_noise_control_value()
        noise_level = self.flow_sde_noise_scheduler.refresh(noise_control)
        noise_levels = torch.full((batch_size,), noise_level, device=device, dtype=dtype)
        task_ids = task_ids.to(device=device, dtype=torch.long).reshape(-1)
        if task_ids.shape[0] != batch_size:
            raise ValueError(f"task_ids batch size {task_ids.shape[0]} does not match batch size {batch_size}")

        for task_id, task_noise_scheduler in self.flow_sde_task_noise_level.items():
            task_mask = task_ids == task_id
            if task_mask.any():
                noise_levels = noise_levels.masked_fill(task_mask, task_noise_scheduler.refresh(noise_control))
        return noise_levels

    def _sample_actions_flow_sde(
        self,
        state_features: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        noise_scale: float,
        requires_grad: bool,
        return_log_prob: bool,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        add noise to the action sampling process using Flow-SDE method.
        see https://arxiv.org/abs/2510.25889
        """

        prefix_features, states = state_features
        prefix_embs, prefix_pad_masks, _ = prefix_features
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device
        beta = self.flow_sde_beta().to(device=device, dtype=prefix_embs.dtype)
        noise_levels = self._resolve_flow_sde_noise_levels(
            batch_size=batch_size,
            device=device,
            dtype=prefix_embs.dtype,
            task_ids=task_ids,
        )

        past_key_values = self._build_kv_cache_from_prefix(prefix_features)
        actions_shape = (batch_size, self.model.n_action_steps, self.model.max_action_dim)
        x_t = torch.randn(actions_shape, device=device, dtype=prefix_embs.dtype)

        timesteps = torch.linspace(1.0, 0.0, self.model.num_steps + 1, dtype=torch.float32, device=device)
        step_log_probs: list[torch.Tensor] = []

        for idx in range(self.model.num_steps):
            t_cur = timesteps[idx]
            t_next = timesteps[idx + 1]
            delta = (t_cur - t_next).clamp_min(1e-6)

            if requires_grad:
                v_t = self.model.denoise_step(
                    states,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    t_cur.expand(batch_size),
                )
            else:
                with torch.no_grad():
                    v_t = self.model.denoise_step(
                        states,
                        prefix_pad_masks,
                        past_key_values,
                        x_t,
                        t_cur.expand(batch_size),
                    )

            t_cur_safe = t_cur.clamp(min=1e-4, max=1.0 - 1e-4)
            t_cur_exp = t_cur_safe.view(1, 1, 1)
            t_next_exp = t_next.view(1, 1, 1)
            delta_exp = delta.view(1, 1, 1)

            x0_pred = x_t - v_t * t_cur_exp
            x1_pred = x_t + v_t * (1.0 - t_cur_exp)

            if noise_scale > 0:
                sigma_schedule = noise_levels * noise_scale * torch.sqrt(t_cur_safe / (1.0 - t_cur_safe))
                sigma = beta * sigma_schedule
                sigma_exp = sigma.view(batch_size, 1, 1)
                x0_weight = 1.0 - t_next_exp
                x1_weight = t_next_exp - sigma_exp.pow(2) * delta_exp / (2.0 * t_cur_exp)
                x_mean = x0_pred * x0_weight + x1_pred * x1_weight
                sigma_t = torch.sqrt(delta_exp) * sigma_exp
                eps = torch.randn_like(x_t)
                x_prev = x_mean + sigma_t * eps
            else:
                x0_weight = 1.0 - t_next_exp
                x1_weight = t_next_exp
                x_mean = x0_pred * x0_weight + x1_pred * x1_weight
                sigma_t = torch.zeros_like(x_mean)
                x_prev = x_mean

            if return_log_prob:
                step_log_probs.append(self._gaussian_log_prob(x_prev, x_mean, sigma_t))

            x_t = x_prev

        if return_log_prob:
            log_probs = torch.stack(step_log_probs, dim=1).sum(dim=1)
        else:
            log_probs = None

        return x_t, log_probs

    def _build_kv_cache_from_prefix(
        self,
        prefix_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """Build KV cache for prefix. No grad needed."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_features
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        with torch.no_grad():
            _, past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.model.use_cache,
                fill_kv_cache=True,
                adarms_cond=[None, None],
            )
        return past_key_values

    @override
    def sac_init(self):
        """Initialize SAC-related components."""

        self.freeze_vision_tower()
        forward_methods = [
            "bc_loss",
            "sac_sample_actions",
            "sac_forward_critic",
            "sac_forward_actor",
            "sac_forward_state_features",
        ]
        for method in forward_methods:
            register_fsdp_forward_method(self, method)

    @torch.no_grad()
    def sac_sample_actions(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module | None = None,
        validate: bool = False,
    ) -> Pi0Output:
        pi0_output, _, _ = self.sample_actions(obs, tokenizer, validate)
        return pi0_output

    @torch.no_grad()
    def sac_get_critic_value(
        self,
        obs: DataProto,
        actions: ModelOutput,
        tokenizer: nn.Module | None = None,
    ) -> torch.Tensor:
        actions = cast(Pi0Output, actions)
        state_features = self.sac_forward_state_features(obs, tokenizer)
        task_ids = None
        if self.critic_api.uses_task_ids:
            if obs.batch is not None and "task_ids" in obs.batch:
                task_ids = obs.batch["task_ids"]
            elif "task_ids" in obs.non_tensor_batch:
                task_ids = torch.tensor(obs.non_tensor_batch["task_ids"], device=actions.action.device)
            else:
                raise ValueError(f"critic_type={self.critic_type} requires task_ids in obs.")
        critic_q_values = self.sac_forward_critic(
            a={"action": actions.action},
            state_features=state_features,
            task_ids=task_ids,
            use_target_network=False,
            method="min",
            requires_grad=False,
        )
        return critic_q_values.detach().float()

    @override
    def sac_forward_actor(
        self,
        state_features: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        task_ids: torch.Tensor | None = None,
        is_first_micro_batch: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        actions, log_probs = self._sample_actions_flow_sde(
            state_features=state_features,
            noise_scale=self.flow_sde_train_noise_scale,
            requires_grad=True,
            return_log_prob=True,
            task_ids=task_ids,
        )
        if is_first_micro_batch:
            self.flow_sde_step.add_(1)
        actor_metrics: dict[str, float] = {}
        if self.flow_sde_enable:
            actor_metrics = {
                "flow_sde_beta": float(self.flow_sde_beta().item()),
                "flow_sde_step": float(self.flow_sde_step.item()),
            }
        _, pi0_output_cls = self._get_pi0_policy_classes()
        pi0_output = pi0_output_cls.from_model_output(
            {
                "full_action": self.action_unnormalize_transform(actions),
                "log_probs": log_probs,
            }
        )
        return pi0_output.action, pi0_output.log_prob, actor_metrics

    @override
    def sac_forward_critic(
        self,
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
    ):
        if self.critic_api.uses_task_ids and task_ids is None:
            raise ValueError(f"critic_type={self.critic_type} requires task_ids for critic forward.")
        return self.critic_api.forward(
            self,
            a=a,
            state_features=state_features,
            task_ids=task_ids,
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    @override
    def sac_get_critic_parameters(self) -> list[torch.nn.Parameter]:
        return self.critic_api.get_critic_parameters(self)

    @override
    def sac_get_named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        named_parameters = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        return named_parameters

    @override
    def sac_forward_state_features(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        pi0_input_cls, _ = self._get_pi0_policy_classes()
        pi0_input = pi0_input_cls.from_env_obs(obs)

        with torch.no_grad():
            state = self.state_normalize_transform(pi0_input.state)
            images, _ = self.image_transform.call_batch(pi0_input.images)
            lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
                {"task": pi0_input.task, "observation.state": state}, tokenizer
            )
            prefix_features = self.model.embed_prefix(
                images=images,
                img_masks=pi0_input.img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
            )
        return (prefix_features, state)

    @override
    @torch.no_grad()
    def sac_update_target_network(self, tau: float):
        self.critic_api.update_target_network(self, tau)
