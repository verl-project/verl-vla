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

import os
from itertools import chain
from pathlib import Path

import einops
import torch
import torchvision
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from ..configuration_act_torch import ACTTorchConfig
from .act_transformer import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    create_sinusoidal_pos_embedding,
)


def _is_accelerate_meta_init_enabled() -> bool:
    return getattr(nn.Module.register_parameter, "__name__", "") == "register_empty_parameter"


class ACT(nn.Module):
    def __init__(self, config: ACTTorchConfig):
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.state_dim > 0:
                self.vae_encoder_robot_state_input_proj = nn.Linear(self.config.state_dim, config.dim_model)
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_dim,
                config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.state_dim > 0:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        if self.config.num_cameras > 0:
            backbone_weights = None if _is_accelerate_meta_init_enabled() else config.pretrained_backbone_weights
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        if self.config.state_dim > 0:
            self.encoder_robot_state_input_proj = nn.Linear(self.config.state_dim, config.dim_model)
        if self.config.env_state_dim > 0:
            self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_dim, config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.num_cameras > 0:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)

        n_1d_tokens = 1
        if self.config.state_dim > 0:
            n_1d_tokens += 1
        if self.config.env_state_dim > 0:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.num_cameras > 0:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        self.action_head = nn.Linear(config.dim_model, self.config.action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        images: dict[str, Tensor] | list[Tensor],
        state: Tensor,
        actions: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        env_state: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        if isinstance(images, dict):
            image_list = list(images.values())
        else:
            image_list = images

        batch_size = image_list[0].shape[0] if image_list else state.shape[0]

        if self.config.use_vae and actions is not None and self.training:
            cls_embed = self.vae_encoder_cls_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            vae_encoder_input = [cls_embed]

            if self.config.state_dim > 0:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(state).unsqueeze(1)
                vae_encoder_input.append(robot_state_embed)

            action_embed = self.vae_encoder_action_input_proj(actions)
            vae_encoder_input.append(action_embed)
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            if self.config.state_dim > 0:
                cls_joint_is_pad = torch.full(
                    (batch_size, 2),
                    False,
                    device=state.device,
                )
            else:
                cls_joint_is_pad = torch.full(
                    (batch_size, 1),
                    False,
                    device=state.device,
                )

            if action_is_pad is None:
                if actions.dim() == 3:
                    action_is_pad = torch.zeros(batch_size, actions.shape[1], dtype=torch.bool, device=state.device)
                else:
                    action_is_pad = torch.zeros(
                        batch_size, self.config.chunk_size, dtype=torch.bool, device=state.device
                    )

            key_padding_mask = torch.cat([cls_joint_is_pad, action_is_pad], axis=1)

            vae_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )
            cls_token_out = vae_out[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=state.device)

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.state_dim > 0:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(state))
        if self.config.env_state_dim > 0:
            if env_state is None:
                env_state = torch.zeros(batch_size, self.config.env_state_dim, device=state.device)
            encoder_in_tokens.append(self.encoder_env_state_input_proj(env_state))

        if self.config.num_cameras > 0 and image_list:
            for img in image_list:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions_pred = self.action_head(decoder_out)

        return actions_pred, (mu, log_sigma_x2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if config is None:
            config = ACTTorchConfig.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=local_files_only,
            )

        model = cls(config)

        model_path = os.fspath(pretrained_model_name_or_path)
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        else:
            weights_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(weights_path):
                state_dict = safetensors_load_file(weights_path, device="cpu")
            elif os.path.isdir(model_path):
                raise FileNotFoundError(
                    f"No ACT weights found in {model_path!r}; expected pytorch_model.bin or model.safetensors."
                )
            else:
                try:
                    weights_path = hf_hub_download(
                        repo_id=model_path,
                        filename="model.safetensors",
                        local_files_only=local_files_only,
                    )
                except Exception as exc:
                    raise FileNotFoundError(
                        f"Could not download model.safetensors for ACT checkpoint {model_path!r}."
                    ) from exc
                state_dict = safetensors_load_file(weights_path, device="cpu")

        model.load_state_dict(state_dict)

        return model

    def save_pretrained(self, save_directory, *, state_dict=None, **kwargs):
        del kwargs
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict() if state_dict is None else state_dict, save_directory / "pytorch_model.bin")
        self.config.save_pretrained(save_directory)
