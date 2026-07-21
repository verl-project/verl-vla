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

import logging
import os
from collections import deque
from typing import Literal, Optional, cast

import einops
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor
from torch.distributed.fsdp import register_fsdp_forward_method
from transformers import PreTrainedModel
from typing_extensions import override
from verl.protocol import DataProto

from ..base import ModelOutput, SupportSACTraining, SupportSFTTraining
from .act_utils import IdentityTransform, ImageTransform, Normalize, Unnormalize
from .configuration_act_torch import ACTTorchConfig
from .critic import CRITIC_BACKENDS
from .model.modeling_act import ACT
from .policy import get_act_policy_classes
from .policy.base import ActOutput

logger = logging.getLogger(__name__)


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACTForActionPrediction(PreTrainedModel, SupportSACTraining, SupportSFTTraining):
    config_class = ACTTorchConfig
    base_model_prefix = "act"

    def __init__(self, config: ACTTorchConfig):
        super().__init__(config)
        SupportSFTTraining.__init__(self, config)
        self.config = config

        self.model: ACT = ACT(config)

        if self.config.state_norm_stats:
            self.state_normalize_transform = Normalize(self.config.state_norm_stats)
            self.state_unnormalize_transform = Unnormalize(self.config.state_norm_stats)
        else:
            self.state_normalize_transform = IdentityTransform()
            self.state_unnormalize_transform = IdentityTransform()

        if self.config.action_norm_stats:
            self.action_normalize_transform = Normalize(self.config.action_norm_stats)
            self.action_unnormalize_transform = Unnormalize(self.config.action_norm_stats)
        else:
            self.action_normalize_transform = IdentityTransform()
            self.action_unnormalize_transform = IdentityTransform()

        self.image_transform = ImageTransform(
            resize_size=self.config.image_resolution,
            norm_stats=getattr(self.config, "image_norm_stats", None),
        )

        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(self.config.temporal_ensemble_coeff, self.config.chunk_size)
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

        if getattr(self.config, "sac_enable", False):
            self._ensure_sac_components()

    def _ensure_sac_components(self):
        if hasattr(self, "critic_api") and hasattr(self, "critic_backend"):
            return
        if self.config.critic_type not in CRITIC_BACKENDS:
            raise ValueError(f"Unsupported critic_type: {self.config.critic_type}")
        self.config.sac_enable = True
        self.critic_api = CRITIC_BACKENDS[self.config.critic_type]
        self.critic_api.init(self)
        self.critic_backend.to(next(self.model.parameters()).device)

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _get_act_policy_classes(self):
        return get_act_policy_classes(self.config.policy_type)

    def _to(self, device: torch.device | str):
        if hasattr(self.state_normalize_transform, "to"):
            self.state_normalize_transform.to(device)
            self.state_unnormalize_transform.to(device)
            self.action_normalize_transform.to(device)
            self.action_unnormalize_transform.to(device)
        return self

    def forward(
        self, images: list[Tensor], state: Tensor, actions: Tensor | None = None, env_state: Tensor | None = None
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        return self.model(images, state, actions, env_state=env_state)

    def embed_prefix(
        self, images: dict[str, Tensor], state: Tensor, env_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if isinstance(images, dict):
            image_list = list(images.values())
        else:
            image_list = images

        batch_size = image_list[0].shape[0] if image_list else state.shape[0]

        latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=state.device)

        encoder_in_tokens = [self.model.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.state_dim > 0:
            encoder_in_tokens.append(self.model.encoder_robot_state_input_proj(state))
        if self.config.env_state_dim > 0:
            if env_state is None:
                env_state = torch.zeros(batch_size, self.config.env_state_dim, device=state.device)
            encoder_in_tokens.append(self.model.encoder_env_state_input_proj(env_state))

        if self.config.num_cameras > 0 and image_list:
            for img in image_list:
                cam_features = self.model.backbone(img)["feature_map"]
                cam_pos_embed = self.model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.model.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        if encoder_in_pos_embed.shape[1] == 1 and encoder_in_tokens.shape[1] != 1:
            encoder_in_pos_embed = encoder_in_pos_embed.expand(-1, encoder_in_tokens.shape[1], -1)

        encoder_out = self.model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        encoder_out = encoder_out.transpose(0, 1)
        encoder_in_pos_embed = encoder_in_pos_embed.transpose(0, 1)

        return encoder_out, encoder_in_pos_embed

    @torch.no_grad()
    def sample_actions(
        self, env_obs: DataProto, tokenizer=None, validate: bool = False
    ) -> tuple[ActOutput, dict, dict]:
        act_input_cls, act_output_cls = self._get_act_policy_classes()
        act_input = act_input_cls.from_env_obs(env_obs)

        state = self.state_normalize_transform(act_input.state)
        env_state = act_input.env_state
        images = act_input.images
        if images and self.image_transform:
            images = {k: self.image_transform.call_batch([v], key=k)[0] for k, v in images.items()}

        self.model.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions_pred, _ = self.model(images, state, env_state=env_state)
            actions_pred = self.action_unnormalize_transform(actions_pred)
            action = self.temporal_ensembler.update(actions_pred)
            action = action.unsqueeze(1)
        else:
            actions_pred, _ = self.model(images, state, env_state=env_state)
            action = self.action_unnormalize_transform(actions_pred)[:, : self.config.n_action_steps]

        action = action.float()

        act_output = act_output_cls.from_model_output(
            {
                "full_action": action,
                "log_probs": torch.zeros(action.shape[0], device=action.device, dtype=torch.float32),
                "action_chunk_size": 1
                if self.config.temporal_ensemble_coeff is not None
                else self.config.n_action_steps,
            }
        )

        s = {
            "states": state,
            "images": torch.stack(list(images.values()), dim=1) if images else torch.tensor([]),
        }
        a = {
            "full_action": action,
        }

        return act_output, s, a

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if config is None:
            config = ACTTorchConfig.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=local_files_only,
            )

        policy = cls(config)

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
                    from huggingface_hub import hf_hub_download

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

        incompatible = policy.load_state_dict(state_dict, strict=False)
        if incompatible is not None:
            logger.info("Loaded ACT weights from %s", weights_path)
            logger.info("ACT missing keys: %s", list(incompatible.missing_keys))
            logger.info("ACT unexpected keys: %s", list(incompatible.unexpected_keys))

        return policy

    def save_pretrained(self, save_directory, *args, state_dict=None, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        if state_dict is not None:
            own_state_keys = set(self.state_dict().keys())
            filtered_state_dict = {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in state_dict.items()
                if key in own_state_keys
            }
            ignored_keys = [key for key in state_dict if key not in own_state_keys]
            if ignored_keys:
                logger.info("Ignoring non-ACT keys while saving HF checkpoint: %s", ignored_keys)
            torch.save(filtered_state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        else:
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        model_state_dict = {
            key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")
        }
        if model_state_dict:
            model_incompatible = self.model.load_state_dict(model_state_dict, strict=False, assign=assign)
            logger.info("ACT core missing keys: %s", list(model_incompatible.missing_keys))
            logger.info("ACT core unexpected keys: %s", list(model_incompatible.unexpected_keys))
            non_model_keys = {key: value for key, value in state_dict.items() if not key.startswith("model.")}
            if non_model_keys:
                own_state = self.state_dict()
                wrapper_state = {key: value for key, value in non_model_keys.items() if key in own_state}
                ignored_keys = [key for key in non_model_keys if key not in own_state]
                if ignored_keys:
                    logger.info("Ignoring non-ACT checkpoint keys: %s", ignored_keys)
                if wrapper_state:
                    wrapper_incompatible = super().load_state_dict(wrapper_state, strict=False, assign=assign)
                    logger.info("ACT wrapper missing keys: %s", list(wrapper_incompatible.missing_keys))
                    logger.info("ACT wrapper unexpected keys: %s", list(wrapper_incompatible.unexpected_keys))
                    return wrapper_incompatible
            return model_incompatible

        own_state = self.state_dict()
        if any(key in own_state for key in state_dict):
            return super().load_state_dict(state_dict, strict=False, assign=assign)

        return self.model.load_state_dict(state_dict, strict=False, assign=assign)

    def freeze_vision_tower(self) -> None:
        if hasattr(self.model, "backbone") and self.model.backbone is not None:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            self.model.backbone.eval()

    def get_optim_params(self) -> list[dict]:
        backbone_params = [p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad]
        other_params = [p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]
        param_groups = [{"params": other_params}]
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": self.config.optimizer_lr_backbone,
                }
            )
        return param_groups

    @override
    def sft_init(self):
        super().sft_init()
        if getattr(self.config, "freeze_vision_tower", True):
            self.freeze_vision_tower()
        register_fsdp_forward_method(self, "sft_loss")

    @override
    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        actions: dict[str, Tensor],
        valids: Tensor,
        action_mask: Tensor | None = None,
        target_values: Tensor | None = None,
    ) -> Tensor:
        del target_values

        act_input_cls, _ = self._get_act_policy_classes()
        action_tensor = actions["action"]

        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(1)

        action_horizon = self.config.chunk_size
        action_length = min(action_tensor.shape[1], action_horizon)
        action_tensor = action_tensor[:, :action_horizon, : self.config.action_dim]

        if action_mask is not None:
            action_mask = action_mask[:, :action_horizon]

        if action_tensor.shape[1] < action_horizon:
            action_tensor = torch.nn.functional.pad(
                action_tensor,
                (0, 0, 0, action_horizon - action_tensor.shape[1]),
                value=0.0,
            )
            if action_mask is not None:
                action_mask = torch.nn.functional.pad(
                    action_mask,
                    (0, action_horizon - action_mask.shape[1]),
                    value=0.0,
                )

        action_tensor = self.action_normalize_transform(action_tensor)

        with torch.no_grad():
            act_input = act_input_cls.from_env_obs(obs)
            state = self.state_normalize_transform(act_input.state)
            env_state = act_input.env_state
            images = act_input.images
            if images and self.image_transform:
                images = {k: self.image_transform.call_batch([v], key=k)[0] for k, v in images.items()}

        action_is_pad = None
        if action_mask is not None:
            action_is_pad = ~action_mask.to(torch.bool)

        actions_pred, (mu, log_sigma_x2) = self.model(images, state, action_tensor, action_is_pad, env_state=env_state)

        abs_err = F.l1_loss(action_tensor, actions_pred, reduction="none")

        if action_is_pad is None:
            action_is_pad = (torch.arange(action_horizon, device=abs_err.device) >= action_length).unsqueeze(0)
        valid_mask = ~action_is_pad.unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss = l1_loss

        if self.config.use_vae and mu is not None and log_sigma_x2 is not None:
            mean_kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
            loss = l1_loss + mean_kld * self.config.kl_weight
            self.sft_metrics["kl_div"] = mean_kld.detach()

        self.sft_metrics["l1_loss"] = l1_loss.detach()

        valids = valids.to(device=loss.device, dtype=loss.dtype)
        return (loss * valids).sum() / valids.sum().clamp_min(1.0)

    @override
    def sac_init(self):
        if not hasattr(self, "critic_backend"):
            raise RuntimeError(
                "ACT SAC components must be created before distributed wrapping. "
                "Set model.override_config.sac_enable=true when building the model."
            )
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
    @override
    def sac_sample_actions(
        self,
        obs: DataProto,
        tokenizer: Optional[torch.nn.Module] = None,
        eval: bool = False,
    ) -> ModelOutput:
        act_output, _, _ = self.sample_actions(obs, tokenizer, validate=eval)
        return act_output

    @torch.no_grad()
    @override
    def sac_get_critic_value(
        self,
        obs: DataProto,
        actions: ModelOutput,
        tokenizer: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        actions = cast(ActOutput, actions)
        state_features = self.sac_forward_state_features(obs, tokenizer)
        critic_q_values = self.sac_forward_critic(
            a={"action": actions.action},
            state_features=state_features,
            use_target_network=False,
            method="min",
            requires_grad=False,
        )
        return critic_q_values.detach().float()

    @override
    def sac_get_critic_parameters(self) -> list[torch.nn.Parameter]:
        return self.critic_api.get_critic_parameters(self)

    @override
    def sac_get_named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        named_parameters = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        return named_parameters

    @override
    def sac_forward_critic(
        self,
        a: dict[str, Tensor],
        state_features: tuple[Tensor, Tensor, Tensor],
        task_ids: Optional[Tensor] = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> Tensor:
        prefix_embs, states, _ = state_features
        return self.critic_api.forward(
            self,
            a=a,
            state_features=(prefix_embs, states),
            task_ids=task_ids,
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    @override
    def sac_forward_actor(
        self,
        state_features: tuple[Tensor, Tensor, Tensor],
        task_ids: Optional[Tensor] = None,
        is_first_micro_batch: bool = False,
        noise_scale: Optional[float] = None,
    ) -> tuple[Tensor, Tensor | None, dict[str, float]]:
        del task_ids, is_first_micro_batch

        prefix_embs, states, encoder_in_pos_embed = state_features
        batch_size = prefix_embs.shape[0]

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=prefix_embs.dtype,
            device=prefix_embs.device,
        )

        encoder_out = prefix_embs.transpose(0, 1)
        encoder_in_pos_embed = encoder_in_pos_embed.transpose(0, 1)

        decoder_out = self.model.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.model.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.model.action_head(decoder_out)

        actions = actions[:, : self.config.n_action_steps, :]

        resolved_noise_scale = self.config.sac_action_noise_scale if noise_scale is None else noise_scale
        if resolved_noise_scale > 0:
            noise = torch.randn_like(actions) * resolved_noise_scale
            actions = actions + noise

        _, act_output_cls = self._get_act_policy_classes()
        act_output = act_output_cls.from_model_output(
            {
                "full_action": self.action_unnormalize_transform(actions),
                "log_probs": None,
                "action_chunk_size": self.config.n_action_steps,
            }
        )

        return act_output.action, act_output.log_prob, {}

    @override
    def sac_forward_state_features(self, obs: DataProto, tokenizer: torch.nn.Module) -> tuple[Tensor, Tensor, Tensor]:
        act_input_cls, _ = self._get_act_policy_classes()
        act_input = act_input_cls.from_env_obs(obs)

        with torch.no_grad():
            state = self.state_normalize_transform(act_input.state)
            env_state = act_input.env_state
            images = act_input.images
            if images and self.image_transform:
                images = {k: self.image_transform.call_batch([v], key=k)[0] for k, v in images.items()}

        prefix_embs, encoder_in_pos_embed = self.embed_prefix(images, state, env_state)

        return (prefix_embs, state, encoder_in_pos_embed)

    @override
    @torch.no_grad()
    def sac_update_target_network(self, tau: float):
        self.critic_api.update_target_network(self, tau)


class ACTForConditionalGeneration(ACTForActionPrediction):
    pass
