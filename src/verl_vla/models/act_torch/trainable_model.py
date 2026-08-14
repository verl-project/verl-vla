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
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, cast

import einops
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor
from torch.distributed.fsdp import register_fsdp_forward_method
from typing_extensions import override
from verl.protocol import DataProto

from ..base import ModelOutput, SupportSACTraining, SupportSFTTraining, TrainableVLAModelBase
from .adapter_config import ACTAdapterConfig
from .critic import CRITIC_BACKENDS
from .policy import get_act_policy_classes
from .policy.base import ActOutput

logger = logging.getLogger(__name__)


class ACTTrainableModel(TrainableVLAModelBase, SupportSACTraining, SupportSFTTraining):
    def __init__(
        self,
        policy: ACTPolicy,
        *,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        adapter_config: Mapping | None = None,
        model_path: str | Path | None = None,
    ):
        super().__init__(policy=policy)
        config = ACTAdapterConfig(model_path=model_path, **dict(adapter_config or {}))
        SupportSFTTraining.__init__(self, config)
        self.config = config
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        if self.config.sac_enable:
            self._ensure_sac_components()

    def _ensure_sac_components(self):
        if hasattr(self, "critic_api") and hasattr(self, "critic_backend"):
            return
        if self.config.critic_type not in CRITIC_BACKENDS:
            raise ValueError(f"Unsupported critic_type: {self.config.critic_type}")
        self.critic_api = CRITIC_BACKENDS[self.config.critic_type]
        self.critic_api.init(self)
        self.critic_backend.to(next(self.policy.parameters()).device)

    def reset(self):
        self.policy.reset()

    def _rollout_action_chunk_size(self) -> int:
        native_chunk_size = (
            1 if self.policy.config.temporal_ensemble_coeff is not None else int(self.policy.config.n_action_steps)
        )
        configured_chunk_size = self.config.action_chunk_size
        if configured_chunk_size is None:
            return native_chunk_size
        chunk_size = int(configured_chunk_size)
        if not 1 <= chunk_size <= native_chunk_size:
            raise ValueError(
                f"ACT rollout action_chunk_size must be between 1 and {native_chunk_size}, got {chunk_size}"
            )
        return chunk_size

    def _get_act_policy_classes(self):
        return get_act_policy_classes(self.config.policy_type)

    def _to(self, device: torch.device | str):
        self.to(device)
        return self

    def _build_policy_batch(
        self,
        act_input,
        *,
        actions: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch: dict[str, Tensor] = {}
        policy_config = self.policy.config
        if policy_config.robot_state_feature is not None:
            state_dim = int(policy_config.robot_state_feature.shape[0])
            if act_input.state.shape[-1] != state_dim:
                raise ValueError(
                    f"ACT state dim mismatch: source has {act_input.state.shape[-1]}, checkpoint expects {state_dim}"
                )
            batch[OBS_STATE] = act_input.state

        if policy_config.env_state_feature is not None:
            if act_input.env_state is None:
                raise ValueError(f"ACT checkpoint requires {OBS_ENV_STATE}")
            env_state_dim = int(policy_config.env_state_feature.shape[0])
            if act_input.env_state.shape[-1] != env_state_dim:
                raise ValueError(
                    f"ACT environment state dim mismatch: source has {act_input.env_state.shape[-1]}, "
                    f"checkpoint expects {env_state_dim}"
                )
            batch[OBS_ENV_STATE] = act_input.env_state

        expected_images = tuple(policy_config.image_features)
        source_images = tuple(act_input.images)
        if set(source_images) != set(expected_images):
            raise ValueError(
                f"ACT image feature mismatch: source has {source_images}, checkpoint expects {expected_images}"
            )
        for key in expected_images:
            image = act_input.images[key]
            expected_shape = tuple(policy_config.image_features[key].shape)
            if tuple(image.shape[1:]) != expected_shape:
                raise ValueError(
                    f"ACT image shape mismatch for {key}: source has {tuple(image.shape[1:])}, "
                    f"checkpoint expects {expected_shape}"
                )
            batch[key] = image

        if actions is not None:
            batch[ACTION] = actions
            if action_is_pad is None:
                raise ValueError("ACT training requires an action padding mask")
            batch["action_is_pad"] = action_is_pad
        return self.preprocessor(batch)

    def embed_prefix(
        self, images: dict[str, Tensor], state: Tensor, env_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if isinstance(images, dict):
            image_list = list(images.values())
        else:
            image_list = images

        batch_size = image_list[0].shape[0] if image_list else state.shape[0]

        model = self.policy.model
        config = self.policy.config
        latent_sample = torch.zeros([batch_size, config.latent_dim], dtype=torch.float32, device=state.device)

        encoder_in_tokens = [model.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if config.robot_state_feature is not None:
            encoder_in_tokens.append(model.encoder_robot_state_input_proj(state))
        if config.env_state_feature is not None:
            if env_state is None:
                raise ValueError(f"ACT checkpoint requires {OBS_ENV_STATE}")
            encoder_in_tokens.append(model.encoder_env_state_input_proj(env_state))

        if config.image_features:
            for img in image_list:
                cam_features = model.backbone(img)["feature_map"]
                cam_pos_embed = model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = model.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        encoder_in_pos_embed = encoder_in_pos_embed.expand(-1, batch_size, -1)

        encoder_out = model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        encoder_out = encoder_out.transpose(0, 1)
        encoder_in_pos_embed = encoder_in_pos_embed.transpose(0, 1)

        return encoder_out, encoder_in_pos_embed

    @torch.no_grad()
    def sample_actions(
        self, env_obs: DataProto, tokenizer=None, validate: bool = False
    ) -> tuple[ActOutput, dict, dict]:
        act_input_cls, act_output_cls = self._get_act_policy_classes()
        act_input = act_input_cls.from_env_obs(env_obs)
        policy_batch = self._build_policy_batch(act_input)
        action_chunk_size = self._rollout_action_chunk_size()

        if self.policy.config.temporal_ensemble_coeff is not None:
            action = self.policy.select_action(policy_batch).unsqueeze(1)
        else:
            action = self.policy.predict_action_chunk(policy_batch)[:, :action_chunk_size]
        action = self.postprocessor(action).float()

        act_output = act_output_cls.from_model_output(
            {
                "full_action": action,
                "log_probs": torch.zeros(action.shape[0], device=action.device, dtype=torch.float32),
                "action_chunk_size": action_chunk_size,
            }
        )

        s = {
            "states": policy_batch.get(OBS_STATE, torch.tensor([], device=action.device)),
            "images": torch.stack([policy_batch[key] for key in self.policy.config.image_features], dim=1)
            if self.policy.config.image_features
            else torch.tensor([], device=action.device),
        }
        a = {
            "full_action": action,
        }

        return act_output, s, a

    def save_pretrained(self, save_directory, *args, state_dict=None, **kwargs):
        del args
        self.export_policy(save_directory, state_dict=state_dict, **kwargs)

    def can_generate(self) -> bool:
        return False

    def export_policy(self, output_dir, *, state_dict=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported ACT export options: {sorted(kwargs)}")
        policy_state = self.policy.state_dict() if state_dict is None else self.extract_policy_state_dict(state_dict)
        native_config = deepcopy(self.policy.config)
        construction_config = deepcopy(native_config)
        construction_config.pretrained_backbone_weights = None
        export_policy = ACTPolicy(construction_config)
        export_policy.config = native_config
        export_policy.load_state_dict(policy_state, strict=True)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_policy.save_pretrained(output_dir)
        self.preprocessor.save_pretrained(output_dir)
        self.postprocessor.save_pretrained(output_dir)

    def freeze_vision_tower(self) -> None:
        if hasattr(self.policy.model, "backbone") and self.policy.model.backbone is not None:
            for param in self.policy.model.backbone.parameters():
                param.requires_grad = False
            self.policy.model.backbone.eval()

    def get_optim_params(self) -> list[dict]:
        backbone_prefix = "policy.model.backbone"
        backbone_params = [p for n, p in self.named_parameters() if n.startswith(backbone_prefix) and p.requires_grad]
        other_params = [
            p
            for n, p in self.named_parameters()
            if not n.startswith(backbone_prefix) and not n.startswith("critic_backend") and p.requires_grad
        ]
        param_groups = [{"params": other_params}]
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": self.policy.config.optimizer_lr_backbone,
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
        policy_config = self.policy.config
        expected_shape = (policy_config.chunk_size, policy_config.action_feature.shape[0])
        if action_tensor.ndim != 3 or tuple(action_tensor.shape[1:]) != expected_shape:
            raise ValueError(
                f"ACT action batch must have shape (batch, {expected_shape[0]}, {expected_shape[1]}), "
                f"got {tuple(action_tensor.shape)}"
            )
        if action_mask is None:
            # Environment rollouts always contain a complete ACT action chunk.
            # Only dataset batches carry padding metadata.
            action_mask = torch.ones(action_tensor.shape[:2], device=action_tensor.device, dtype=torch.bool)
        elif tuple(action_mask.shape) != tuple(action_tensor.shape[:2]):
            raise ValueError(
                f"ACT action mask must have shape {tuple(action_tensor.shape[:2])}, got {tuple(action_mask.shape)}"
            )

        with torch.no_grad():
            act_input = act_input_cls.from_env_obs(obs)
            action_is_pad = ~action_mask.to(torch.bool)
            policy_batch = self._build_policy_batch(
                act_input,
                actions=action_tensor,
                action_is_pad=action_is_pad,
            )
            if self.policy.config.image_features:
                policy_batch[OBS_IMAGES] = [policy_batch[key] for key in self.policy.config.image_features]

        actions_pred, (mu, log_sigma_x2) = self.policy.model(policy_batch)
        normalized_actions = policy_batch[ACTION]
        abs_err = torch.abs(normalized_actions - actions_pred)
        valid_mask = ~action_is_pad.unsqueeze(-1)
        l1_loss = (abs_err * valid_mask).mean(dim=(1, 2))

        loss = l1_loss

        if self.policy.config.use_vae and mu is not None and log_sigma_x2 is not None:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1)
            loss = l1_loss + kld * self.policy.config.kl_weight
            self.sft_metrics["kl_div"] = kld.mean().detach()

        self.sft_metrics["l1_loss"] = l1_loss.mean().detach()

        valids = valids.to(device=loss.device, dtype=loss.dtype)
        return (loss * valids).sum() / valids.sum().clamp_min(1.0)

    @override
    def sac_init(self):
        if getattr(self.config, "freeze_vision_tower", True):
            self.freeze_vision_tower()

        forward_methods = ["sac_sample_actions"]
        if not self.config.sac_enable:
            for method in forward_methods:
                register_fsdp_forward_method(self, method)
            return

        if not hasattr(self, "critic_backend"):
            raise RuntimeError(
                "ACT SAC components must be created before distributed wrapping. "
                "Set model.adapter.critic.enabled=true when building the model."
            )
        forward_methods.extend(
            [
                "sft_loss",
                "sac_forward_critic",
                "sac_forward_actor",
                "sac_forward_state_features",
            ]
        )
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
        state_features = self.sac_forward_state_features(obs, tokenizer)
        actions, _, _ = self.sac_forward_actor(
            state_features,
            noise_scale=0.0 if eval else self.config.sac_rollout_noise_scale,
        )
        _, act_output_cls = self._get_act_policy_classes()
        return act_output_cls.from_model_output(
            {
                "full_action": actions,
                "log_probs": None,
                "action_chunk_size": self._rollout_action_chunk_size(),
            }
        )

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
        named_parameters = [(name, param) for name, param in self.policy.named_parameters() if param.requires_grad]
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
            state_features=(prefix_embs.detach(), states.detach()),
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
            (self.policy.config.chunk_size, batch_size, self.policy.config.dim_model),
            dtype=prefix_embs.dtype,
            device=prefix_embs.device,
        )

        encoder_out = prefix_embs.transpose(0, 1)
        encoder_in_pos_embed = encoder_in_pos_embed.transpose(0, 1)

        decoder_out = self.policy.model.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.policy.model.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.policy.model.action_head(decoder_out)

        actions = actions[:, : self.policy.config.n_action_steps, :]

        resolved_noise_scale = self.config.sac_train_noise_scale if noise_scale is None else noise_scale
        if resolved_noise_scale > 0:
            noise = torch.randn_like(actions) * resolved_noise_scale
            actions = actions + noise

        _, act_output_cls = self._get_act_policy_classes()
        act_output = act_output_cls.from_model_output(
            {
                "full_action": self.postprocessor(actions).to(device=actions.device),
                "log_probs": None,
                "action_chunk_size": self.policy.config.n_action_steps,
            }
        )

        return act_output.action, act_output.log_prob, {}

    @override
    def sac_forward_state_features(self, obs: DataProto, tokenizer: torch.nn.Module) -> tuple[Tensor, Tensor, Tensor]:
        act_input_cls, _ = self._get_act_policy_classes()
        act_input = act_input_cls.from_env_obs(obs)

        with torch.no_grad():
            policy_batch = self._build_policy_batch(act_input)
            state = policy_batch.get(OBS_STATE)
            if state is None:
                state = torch.empty((len(obs), 0), device=next(self.policy.parameters()).device)
            env_state = policy_batch.get(OBS_ENV_STATE)
            images = {key: policy_batch[key] for key in self.policy.config.image_features}

        prefix_embs, encoder_in_pos_embed = self.embed_prefix(images, state, env_state)

        return (prefix_embs, state, encoder_in_pos_embed)

    @override
    @torch.no_grad()
    def sac_update_target_network(self, tau: float):
        self.critic_api.update_target_network(self, tau)
