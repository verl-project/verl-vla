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

from __future__ import annotations

import math
import os
from typing import Literal, cast

import torch
from torch import nn
from torch.distributed.fsdp import register_fsdp_forward_method
from transformers import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
from typing_extensions import override
from verl.protocol import DataProto
from verl.utils.device import get_device_name

from verl_vla.utils.scalar_schedule import ScheduledScalar

from ..base import ModelOutput, SupportSACTraining, SupportSFTTraining
from ..pi0_torch.critic import (
    CrossAttentionCriticBackend,
    MeanPoolCriticBackend,
    MultiCrossAttentionCriticBackend,
)
from .configuration_gr00t_torch import Gr00tTorchConfig
from .gr00t_utils import (
    Gr00tImageTransform,
    Gr00tVLMTransform,
    MinMaxNormalize,
    MinMaxUnnormalize,
    pad_last_dim,
)
from .model.modeling_gr00t_n1d7 import Gr00tN1d7Model
from .policy import get_gr00t_policy_classes
from .policy.base import Gr00tOutput

# The critic backends are model-agnostic: they consume
# ((prefix_embs, prefix_pad_masks, ...), states) plus a flattened action, so the
# PI0 implementations are reused as-is.
CRITIC_BACKENDS = {
    "cross_attn": CrossAttentionCriticBackend(),
    "mean_pool": MeanPoolCriticBackend(),
    "multi_cross_attn": MultiCrossAttentionCriticBackend(),
}

# GR00T state features for SAC:
#   ((vl_embeds, backbone_attention_mask, image_mask), state, embodiment_id)
# vl_embeds are the backbone features already processed by vlln + vl self-attention;
# state is the normalized *unpadded* env state. All leaves are tensors so the
# structure survives split_nested_dicts_or_tuples.
Gr00tStateFeatures = tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]


class Gr00tForActionPrediction(PreTrainedModel, SupportSACTraining, SupportSFTTraining):
    config_class = Gr00tTorchConfig
    base_model_prefix = "gr00t_torch"

    def __init__(self, config: Gr00tTorchConfig):
        super().__init__(config)
        SupportSFTTraining.__init__(self, config)
        self.model = Gr00tN1d7Model(config)

        self.state_norm_stats = config.state_norm_stats
        self.action_norm_stats = config.action_norm_stats
        self.policy_type = config.policy_type
        self.action_chunk_size = int(getattr(config, "action_chunk_size", 10))
        self.sft_action_horizon = int(getattr(config, "sft_action_horizon", 16))
        self.critic_type = getattr(config, "critic_type", "cross_attn")

        assert self.state_norm_stats, "state_norm_stats must be provided in Gr00tTorchConfig"
        assert self.action_norm_stats, "action_norm_stats must be provided in Gr00tTorchConfig"
        self.env_action_dim = len(self.action_norm_stats["q01"])

        # Input transforms
        self.state_normalize_transform = MinMaxNormalize(self.state_norm_stats, clip=True)
        self.action_normalize_transform = MinMaxNormalize(self.action_norm_stats, clip=True)
        self.image_transform = Gr00tImageTransform(
            target_size=tuple(config.image_target_size),
            crop_size=tuple(config.image_crop_size),
        )
        processor_path = config.vlm_processor_path or getattr(config, "_name_or_path", None) or config.model_name
        self.vlm_transform = Gr00tVLMTransform(processor_path, formalize=config.formalize_language)

        # Output transforms
        self.action_unnormalize_transform = MinMaxUnnormalize(self.action_norm_stats)

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
        self.flow_sde_beta_schedule_T = int(getattr(config, "flow_sde_beta_schedule_T", 2000))
        self.flow_sde_logprob_masked = bool(getattr(config, "flow_sde_logprob_masked", True))
        self.register_buffer("flow_sde_step", torch.zeros((), dtype=torch.long))

        ##### SAC Algorithm Support #####
        if getattr(self.config, "sac_enable", False):
            if self.critic_type not in CRITIC_BACKENDS:
                raise ValueError(f"Unsupported critic_type: {self.critic_type}")
            self.critic_api = CRITIC_BACKENDS[self.critic_type]
            self.critic_api.init(self)

    def _get_gr00t_policy_classes(self):
        return get_gr00t_policy_classes(self.policy_type)

    def _to(self, device: torch.device | str):
        self.state_normalize_transform.to(device)
        self.action_normalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        return self

    def forward(self, inputs: dict) -> dict:
        """Flow-matching training forward; returns the loss dict from the action head."""
        return self.model(inputs)

    # --- Preprocessing helpers ---

    def _embodiment_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size,), int(self.config.embodiment_id), dtype=torch.long, device=device)

    def _preprocess_backbone_inputs(self, gr00t_input, train: bool) -> dict[str, torch.Tensor]:
        """Images + language -> Qwen3-VL tokenizer inputs (no grad by nature)."""
        images = self.image_transform(gr00t_input.images, train=train)
        return self.vlm_transform.call_batch(images, gr00t_input.task)

    def _preprocess_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize the raw env state (kept at env dim, unpadded)."""
        return self.state_normalize_transform(state)

    def _pad_state_for_action_head(self, state: torch.Tensor) -> torch.Tensor:
        """(B, env_state_dim) -> (B, state_history_length, max_state_dim)."""
        padded = pad_last_dim(state, int(self.config.max_state_dim))
        return padded.unsqueeze(1)

    @torch.no_grad()
    def _forward_backbone_features(self, gr00t_input, train: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the VLM backbone and vlln/vl self-attention once.

        Returns (vl_embeds, backbone_attention_mask, image_mask).
        """
        vlm_inputs = self._preprocess_backbone_inputs(gr00t_input, train=train)
        backbone_inputs, _ = self.model.prepare_input(vlm_inputs)
        backbone_output = self.model.backbone(backbone_inputs)
        backbone_output = self.model.action_head.process_backbone_output(backbone_output)
        return (
            backbone_output.backbone_features,
            backbone_output.backbone_attention_mask,
            backbone_output.image_mask,
        )

    # --- Sampling ---

    @torch.no_grad()
    def sample_actions(
        self,
        env_obs: DataProto,
        tokenizer=None,
        eval: bool = False,
    ) -> tuple[Gr00tOutput, dict, dict]:
        """Run one forward pass from raw env observations to the final action chunk.

        The ``tokenizer`` argument is accepted for interface parity with PI0 but is
        unused: GR00T needs the full Qwen3-VL processor, which the wrapper owns
        (``self.vlm_transform``).
        """
        del tokenizer
        input_cls, output_cls = self._get_gr00t_policy_classes()
        gr00t_input = input_cls.from_env_obs(env_obs)

        state = self._preprocess_state(gr00t_input.state)
        vl_embeds, backbone_attention_mask, image_mask = self._forward_backbone_features(gr00t_input, train=False)
        state_features: Gr00tStateFeatures = (
            (vl_embeds, backbone_attention_mask, image_mask),
            state,
            self._embodiment_ids(state.shape[0], state.device),
        )

        if self.flow_sde_enable and not eval:
            pred_action, rollout_log_probs = self._sample_actions_flow_sde(
                state_features=state_features,
                noise_scale=self.flow_sde_rollout_noise_scale,
                requires_grad=False,
                return_log_prob=True,
                task_ids=torch.tensor(env_obs.non_tensor_batch["task_ids"], device=state.device, dtype=torch.long),
            )
        else:
            state_feat_embs = self.model.action_head.encode_state_features(
                self._pad_state_for_action_head(state),
                state_features[2],
            )
            backbone_output = BatchFeature(
                data={
                    "backbone_features": vl_embeds,
                    "backbone_attention_mask": backbone_attention_mask,
                    "image_mask": image_mask,
                }
            )
            result = self.model.action_head.get_action_with_features(
                backbone_features=vl_embeds,
                state_features=state_feat_embs,
                embodiment_id=state_features[2],
                backbone_output=backbone_output,
            )
            pred_action = result.action_pred
            rollout_log_probs = torch.zeros(pred_action.shape[0], device=pred_action.device, dtype=torch.float32)

        gr00t_output = output_cls.from_model_output(
            {
                "full_action": self.action_unnormalize_transform(pred_action[..., : self.env_action_dim]),
                "log_probs": rollout_log_probs,
                "action_chunk_size": self.action_chunk_size,
                "action_dim": self.env_action_dim,
            }
        )
        s = {
            "states": state,
            "input_ids": None,
            "vl_embeds": vl_embeds,
        }
        a = {
            "full_action": pred_action,
            "log_probs": rollout_log_probs,
        }
        return gr00t_output, s, a

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            config = Gr00tTorchConfig.from_pretrained(pretrained_model_name_or_path)
        # Make the checkpoint dir the default source for the bundled Qwen3-VL processor.
        config._name_or_path = str(pretrained_model_name_or_path)

        policy = cls(config)
        policy.model = Gr00tN1d7Model.from_pretrained(pretrained_model_name_or_path, config=config)
        return policy

    def save_pretrained(self, save_directory, *args, state_dict=None, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)
        self.model.save_pretrained(save_directory, *args, **kwargs)
        self.config.save_pretrained(save_directory)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        filtered_state_dict = {
            key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")
        }
        return self.model.load_state_dict(filtered_state_dict, strict=False, assign=assign)

    def apply_backbone_freezing(self) -> None:
        """Apply the configured freezing scheme to backbone and action head."""
        self.model.backbone.set_trainable_parameters(
            tune_llm=bool(self.config.tune_llm),
            tune_visual=bool(self.config.tune_visual),
            tune_top_llm_layers=int(self.config.tune_top_llm_layers),
        )
        self.model.action_head.set_trainable_parameters(
            tune_projector=bool(self.config.tune_projector),
            tune_diffusion_model=bool(self.config.tune_diffusion_model),
            tune_vlln=bool(self.config.tune_vlln),
        )

    @property
    def _backbone_requires_grad(self) -> bool:
        return bool(self.config.tune_llm) or bool(self.config.tune_visual) or int(self.config.tune_top_llm_layers) > 0

    @override
    def sft_init(self):
        """Override SupportSFTTraining.sft_init for GR00T SFT setup."""
        self.apply_backbone_freezing()
        register_fsdp_forward_method(self, "sft_loss")

    @override
    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        target_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Override SupportSFTTraining.sft_loss for GR00T BC training.

        RECAP return-conditioning arrives as ACP tags already appended to the task
        strings, so target_values is unused here (same as PI0).
        """
        del target_values, tokenizer
        input_cls, _ = self._get_gr00t_policy_classes()

        action_tensor = actions["action"]
        supervised_horizon = min(self.sft_action_horizon, int(self.config.action_horizon))
        action_length = min(action_tensor.shape[1], supervised_horizon)
        action_tensor = action_tensor[:, :supervised_horizon, : self.env_action_dim]
        if action_mask is not None:
            action_mask = action_mask[:, :supervised_horizon]
        if action_tensor.shape[1] < supervised_horizon:
            action_tensor = torch.nn.functional.pad(
                action_tensor, (0, 0, 0, supervised_horizon - action_tensor.shape[1]), value=0.0
            )
            if action_mask is not None:
                action_mask = torch.nn.functional.pad(
                    action_mask, (0, supervised_horizon - action_mask.shape[1]), value=0.0
                )

        # Normalize (with clipping) at env dims, then pad dims and horizon to model size.
        action_tensor = self.action_normalize_transform(action_tensor)
        action_tensor = pad_last_dim(action_tensor, int(self.config.max_action_dim))
        model_horizon = int(self.config.action_horizon)
        if action_tensor.shape[1] < model_horizon:
            action_tensor = torch.nn.functional.pad(
                action_tensor, (0, 0, 0, model_horizon - action_tensor.shape[1]), value=0.0
            )

        # Step mask (B, model_horizon)
        if action_mask is None:
            step_mask = (
                (torch.arange(model_horizon, device=action_tensor.device) < action_length)
                .to(dtype=action_tensor.dtype)
                .unsqueeze(0)
                .expand(action_tensor.shape[0], -1)
            )
        else:
            step_mask = torch.nn.functional.pad(
                action_mask.to(device=action_tensor.device, dtype=action_tensor.dtype),
                (0, model_horizon - action_mask.shape[1]),
                value=0.0,
            )
        # Dim mask: only the env action dims are supervised.
        dim_mask = (
            (torch.arange(int(self.config.max_action_dim), device=action_tensor.device) < self.env_action_dim)
            .to(dtype=action_tensor.dtype)
            .view(1, 1, -1)
        )
        action_mask_3d = step_mask.unsqueeze(-1) * dim_mask

        gr00t_input = input_cls.from_env_obs(obs)
        with torch.no_grad():
            vlm_inputs = self._preprocess_backbone_inputs(gr00t_input, train=self.training)
            state = self._pad_state_for_action_head(self._preprocess_state(gr00t_input.state))

        inputs = {
            **vlm_inputs,
            "state": state,
            "action": action_tensor,
            "embodiment_id": self._embodiment_ids(action_tensor.shape[0], action_tensor.device),
            "action_mask": action_mask_3d,
        }
        backbone_inputs, action_inputs = self.model.prepare_input(inputs)
        # The backbone forward stays inside the grad graph so tune_llm / tune_visual /
        # tune_top_llm_layers configurations receive gradients.
        if self._backbone_requires_grad:
            backbone_outputs = self.model.backbone(backbone_inputs)
        else:
            with torch.no_grad():
                backbone_outputs = self.model.backbone(backbone_inputs)
        head_output = self.model.action_head(backbone_outputs, action_inputs)

        # Per-sample masked loss, then weight by valids.
        action_loss = head_output["action_loss"]
        mask = head_output["action_mask"]
        sample_loss = action_loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
        valids = valids.to(device=sample_loss.device, dtype=sample_loss.dtype)
        return (sample_loss * valids).sum() / valids.sum().clamp_min(1.0)

    # --- SAC Algorithm Support ---

    def _gaussian_log_prob(
        self,
        sample: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        std_safe = std.clamp_min(1e-6)
        log_prob = -0.5 * (((sample - mean) / std_safe) ** 2 + 2.0 * torch.log(std_safe) + math.log(2.0 * math.pi))
        if self.flow_sde_logprob_masked:
            # GR00T pads actions to (action_horizon=40, max_action_dim=132); only the
            # executed chunk over real env dims carries signal — averaging over the
            # pure-noise padding would swamp the entropy term.
            mask = torch.zeros(1, *log_prob.shape[1:], device=log_prob.device, dtype=log_prob.dtype)
            mask[:, : self.action_chunk_size, : self.env_action_dim] = 1.0
            return (log_prob * mask).sum(dim=(-1, -2)) / mask.sum(dim=(-1, -2)).clamp_min(1.0)
        return log_prob.mean(dim=(-1, -2))

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
        normal_noise_factors = (torch.randn_like(noise_levels) / 6.0 + 0.5).clamp(0.0, 1.0)
        return noise_levels * normal_noise_factors

    def _sample_actions_flow_sde(
        self,
        state_features: Gr00tStateFeatures,
        noise_scale: float,
        requires_grad: bool,
        return_log_prob: bool,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flow-SDE stochastic action sampling (https://arxiv.org/abs/2510.25889).

        GR00T's flow-matching time runs 0 -> 1 with x_t = (1-t)*noise + t*action and
        v = action - noise (the reverse of PI0's convention), so the PI0 update rule
        is applied through the substitution tau = 1 - t:
            data estimate   x0_pred = x_t + (1 - t) * v_t
            noise estimate  x1_pred = x_t - t * v_t
            sigma           = noise_level * scale * sqrt(tau / (1 - tau))
            x_mean          = x0_pred * (1 - tau_next) + x1_pred * (tau_next - sigma^2 * dt / (2 * tau_cur))
            x_next          = x_mean + sqrt(dt) * sigma * eps
        With noise_scale = 0 this reduces exactly to the deterministic Euler update
        x_next = x_t + dt * v_t.
        """
        (vl_embeds, backbone_attention_mask, image_mask), state, embodiment_id = state_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        noise_levels = self._resolve_flow_sde_noise_levels(
            batch_size=batch_size,
            device=device,
            dtype=vl_embeds.dtype,
            task_ids=task_ids,
        )

        def _denoise(x_t: torch.Tensor, timesteps_tensor: torch.Tensor, state_feat_embs: torch.Tensor):
            return self.model.action_head.denoise_step(
                vl_embeds=vl_embeds,
                backbone_attention_mask=backbone_attention_mask,
                image_mask=image_mask,
                state_features=state_feat_embs,
                x_t=x_t,
                timesteps_tensor=timesteps_tensor,
                embodiment_id=embodiment_id,
            )

        padded_state = self._pad_state_for_action_head(state)
        if requires_grad:
            state_feat_embs = self.model.action_head.encode_state_features(padded_state, embodiment_id)
        else:
            with torch.no_grad():
                state_feat_embs = self.model.action_head.encode_state_features(padded_state, embodiment_id)

        num_steps = int(self.config.flow_sde_num_inference_timesteps or self.config.num_inference_timesteps)
        num_timestep_buckets = int(self.config.num_timestep_buckets)
        actions_shape = (batch_size, int(self.config.action_horizon), int(self.config.max_action_dim))
        x_t = torch.randn(actions_shape, device=device, dtype=vl_embeds.dtype)

        # GR00T time: 0 (pure noise) -> 1 (data).
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32, device=device)
        step_log_probs: list[torch.Tensor] = []

        for idx in range(num_steps):
            t_cur = timesteps[idx]
            t_next = timesteps[idx + 1]
            delta = (t_next - t_cur).clamp_min(1e-6)

            t_discretized = (t_cur * num_timestep_buckets).long()
            timesteps_tensor = t_discretized.expand(batch_size)
            if requires_grad:
                v_t = _denoise(x_t, timesteps_tensor, state_feat_embs)
            else:
                with torch.no_grad():
                    v_t = _denoise(x_t, timesteps_tensor, state_feat_embs)

            # Noise-time tau = 1 - t (this is PI0's time variable).
            tau_cur = (1.0 - t_cur).clamp(min=1e-4, max=1.0 - 1e-4)
            tau_cur_exp = tau_cur.view(1, 1, 1)
            tau_next_exp = (1.0 - t_next).view(1, 1, 1)
            delta_exp = delta.view(1, 1, 1)

            t_cur_exp = 1.0 - tau_cur_exp
            x0_pred = x_t + v_t * (1.0 - t_cur_exp)
            x1_pred = x_t - v_t * t_cur_exp

            if noise_scale > 0:
                sigma = noise_levels * noise_scale * torch.sqrt(tau_cur / (1.0 - tau_cur))
                sigma_exp = sigma.view(batch_size, 1, 1)
                x0_weight = 1.0 - tau_next_exp
                x1_weight = tau_next_exp - sigma_exp.pow(2) * delta_exp / (2.0 * tau_cur_exp)
                x_mean = x0_pred * x0_weight + x1_pred * x1_weight
                sigma_t = torch.sqrt(delta_exp) * sigma_exp
                eps = torch.randn_like(x_t)
                x_prev = x_mean + sigma_t * eps
            else:
                x0_weight = 1.0 - tau_next_exp
                x1_weight = tau_next_exp
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

    @override
    def sac_init(self):
        """Initialize SAC-related components."""

        self.apply_backbone_freezing()
        forward_methods = [
            "sft_loss",
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
        eval: bool = False,
    ) -> Gr00tOutput:
        gr00t_output, _, _ = self.sample_actions(obs, tokenizer, eval)
        return gr00t_output

    @torch.no_grad()
    def sac_get_critic_value(
        self,
        obs: DataProto,
        actions: ModelOutput,
        tokenizer: nn.Module | None = None,
    ) -> torch.Tensor:
        actions = cast(Gr00tOutput, actions)
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
        state_features: Gr00tStateFeatures,
        task_ids: torch.Tensor | None = None,
        is_first_micro_batch: bool = False,
        noise_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        actor_metrics: dict[str, float] = {}
        if self.flow_sde_enable:
            resolved_noise_scale = self.flow_sde_train_noise_scale if noise_scale is None else noise_scale
            actions, log_probs = self._sample_actions_flow_sde(
                state_features=state_features,
                noise_scale=resolved_noise_scale,
                requires_grad=True,
                return_log_prob=True,
                task_ids=task_ids,
            )
            if is_first_micro_batch:
                self.flow_sde_step.add_(1)
            actor_metrics = {
                "flow_sde_step": float(self.flow_sde_step.item()),
                "flow_sde_noise_level": float(self.flow_sde_noise_scheduler.current_value),
                "flow_sde_noise_control": float(self.flow_sde_noise_scheduler.control_value),
                "flow_sde_noise_scale": float(resolved_noise_scale),
            }
        else:
            actions, log_probs = self._sample_actions_flow_sde(
                state_features=state_features,
                noise_scale=0.0,
                requires_grad=True,
                return_log_prob=False,
                task_ids=task_ids,
            )
        _, gr00t_output_cls = self._get_gr00t_policy_classes()
        gr00t_output = gr00t_output_cls.from_model_output(
            {
                "full_action": self.action_unnormalize_transform(actions[..., : self.env_action_dim]),
                "log_probs": log_probs,
                "action_chunk_size": self.action_chunk_size,
                "action_dim": self.env_action_dim,
            }
        )
        return gr00t_output.action, gr00t_output.log_prob, actor_metrics

    @override
    def sac_forward_critic(
        self,
        a: dict[str, torch.Tensor],
        state_features: Gr00tStateFeatures,
        task_ids: torch.Tensor | None = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ):
        if self.critic_api.uses_task_ids and task_ids is None:
            raise ValueError(f"critic_type={self.critic_type} requires task_ids for critic forward.")
        prefix_features, state, _embodiment_id = state_features
        return self.critic_api.forward(
            self,
            a=a,
            state_features=(prefix_features, state),
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
        tokenizer: torch.nn.Module | None = None,
    ) -> Gr00tStateFeatures:
        del tokenizer
        input_cls, _ = self._get_gr00t_policy_classes()
        gr00t_input = input_cls.from_env_obs(obs)

        with torch.no_grad():
            state = self._preprocess_state(gr00t_input.state)
            vl_embeds, backbone_attention_mask, image_mask = self._forward_backbone_features(gr00t_input, train=False)
        return (
            (vl_embeds, backbone_attention_mask, image_mask),
            state,
            self._embodiment_ids(state.shape[0], state.device),
        )

    @override
    @torch.no_grad()
    def sac_update_target_network(self, tau: float):
        self.critic_api.update_target_network(self, tau)


class Gr00tForConditionalGeneration(Gr00tForActionPrediction):
    pass
