# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR00T N1.7 core model, vendored from Isaac-GR00T (gr00t/model/gr00t_n1d7/gr00t_n1d7.py).

Differences from upstream:
- Config comes from ``verl_vla.models.gr00t_torch.configuration_gr00t_torch``.
- The data collator / ``vlm_content`` path is removed; preprocessing lives in the
  ``Gr00tForActionPrediction`` wrapper.
- The Euler denoising loop is factored into ``denoise_step`` so the SAC flow-SDE
  sampler can run it with gradients enabled.
- The RTC (receding-horizon inpainting) branch is dropped.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from ..configuration_gr00t_torch import Gr00tTorchConfig
from .dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from .embodiment_conditioned_mlp import CategorySpecificMLP, MultiEmbodimentActionEncoder
from .qwen3_backbone import Qwen3Backbone

logger = logging.getLogger(__name__)


def _map_to_device(batch: BatchFeature, device: torch.device, dtype: torch.dtype) -> BatchFeature:
    """Move all tensors in a BatchFeature to device; floating tensors also get dtype."""

    def _convert(x):
        if not torch.is_tensor(x):
            return x
        if torch.is_floating_point(x):
            return x.to(device, dtype=dtype)
        return x.to(device)

    return BatchFeature(data={k: _convert(v) for k, v in batch.items()})


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly so the sampler depends
        # only on the config, not on the construction-time device/dtype context
        # (the head can be instantiated under a meta / no_init_weights context).
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model, config.tune_vlln)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> dict:
        """
        Forward pass through the action head (flow matching training loss).

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - action: [B, action_horizon, action_dim]
                - embodiment_id: [B]
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            dict containing the masked flow-matching loss and per-element action_loss.
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def encode_state_features(self, state: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        """Embed the (normalized, padded) state.

        Args:
            state: [B, state_history_length, max_state_dim]
            embodiment_id: [B]

        Returns:
            state_features: [B, 1, input_embedding_dim]
        """
        assert state.shape[1] == self.config.state_history_length, "state history length mismatch"
        state = state.view(state.shape[0], 1, -1)
        return self.state_encoder(state, embodiment_id)

    def _encode_features(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Encode backbone features (vlln + vl self-attention) and state features."""
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        state_features = self.encode_state_features(action_input.state, action_input.embodiment_id)
        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    def denoise_step(
        self,
        vl_embeds: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        image_mask: torch.Tensor | None,
        state_features: torch.Tensor,
        x_t: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the flow-matching velocity for one denoising step.

        Gradient-capable: callers decide whether to wrap in ``torch.no_grad()``.

        Args:
            vl_embeds: [B, seq_len, backbone_embedding_dim] processed backbone features
            backbone_attention_mask: [B, seq_len] bool
            image_mask: [B, seq_len] bool (required for AlternateVLDiT)
            state_features: [B, 1, input_embedding_dim]
            x_t: [B, action_horizon, action_dim] current noisy actions
            timesteps_tensor: [B] discretized timestep in [0, num_timestep_buckets)
            embodiment_id: [B]

        Returns:
            pred_velocity: [B, action_horizon, action_dim]
        """
        device = vl_embeds.device
        action_features = self.action_encoder(x_t, timesteps_tensor, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )
        pred = self.action_decoder(model_output, embodiment_id)
        return pred[:, -self.action_horizon :]

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """
        Generate actions using the deterministic flow matching Euler integration.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, 1, input_embedding_dim]
            embodiment_id: [B]
            backbone_output: Output from the backbone model (for attention masks)
        """
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device

        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            pred_velocity = self.denoise_step(
                vl_embeds=vl_embeds,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
                image_mask=backbone_output.get("image_mask"),
                state_features=state_features,
                x_t=actions,
                timesteps_tensor=timesteps_tensor,
                embodiment_id=embodiment_id,
            )
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Generate actions from raw backbone output (deterministic eval path)."""
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


class Gr00tN1d7Model(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tTorchConfig
    supports_gradient_checkpointing = True

    def __init__(self, config, transformers_loading_kwargs: dict | None = None):
        super().__init__(config)
        self.config = config
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}

        self.backbone = Qwen3Backbone(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        self.action_head = Gr00tN1d7ActionHead(config)

    def prepare_input(self, inputs: dict) -> tuple[BatchFeature, BatchFeature]:
        """Split an input dict into backbone and action-head inputs on the right device."""
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)
        backbone_inputs = _map_to_device(backbone_inputs, self.device, self.dtype)
        action_inputs = _map_to_device(action_inputs, self.device, self.dtype)
        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> dict:
        """Training forward: flow-matching loss."""
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        return self.action_head(backbone_outputs, action_inputs)

    def get_action(self, inputs: dict) -> BatchFeature:
        """Deterministic action generation (eval path)."""
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        return self.action_head.get_action(backbone_outputs, action_inputs)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
