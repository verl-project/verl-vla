# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Trainable verl-vla composition around the official GR00T N1.6 policy.

Aligns with ``pi0_torch.PI0TrainableModel``: a thin wrapper around the native
policy plus optional SAC critic / Flow-SDE actor sampling. SAC no longer uses a
``Gr00tN1d6`` subclass + separate loader.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from torch import nn
from torch.distributed.fsdp import register_fsdp_forward_method
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_utils import no_init_weights
from verl import DataProto

from verl_vla.models.base import ModelOutput, SupportSACTraining, SupportSFTTraining, TrainableVLAModelMixin

from .adapter_config import Gr00tAdapterConfig
from .compat import apply_gr00t_compat_patches
from .critic import CrossAttentionCriticBackend, MeanPoolCriticBackend
from .gr00t_adapter import GR00TN16Adapter
from .policy import get_gr00t_policy_classes
from .utils import GR1, GR00TDim, extract_critic_state_dict, normalize_adapter_state_dict

logger = logging.getLogger(__name__)

CRITIC_BACKENDS = {
    "cross_attn": CrossAttentionCriticBackend(),
    "mean_pool": MeanPoolCriticBackend(),
}


class _CpuBeta(torch.distributions.Beta):
    def __init__(self, concentration1: float, concentration0: float):
        super().__init__(
            torch.tensor(float(concentration1), dtype=torch.float32, device="cpu"),
            torch.tensor(float(concentration0), dtype=torch.float32, device="cpu"),
        )


_BETA_PATCH_LOCK = Lock()


def beta_schedule(step: int, beta0: float, beta_min: float, T: int) -> float:
    """Cosine anneal of the Flow-SDE exploration scale (GR00T-specific; pi0 uses ScheduledScalar)."""
    progress = min(step / T, 1.0)
    return beta_min + (beta0 - beta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


@contextlib.contextmanager
def _hide_non_peft_adapter_config(path):
    """Temporarily hide verl-vla ``adapter_config.json`` from Transformers PEFT.

    HuggingFace treats any ``adapter_config.json`` as a PEFT adapter.  Our file
    is framework config, not PEFT, so rename it for the duration of
    ``from_pretrained``.
    """
    adapter_path = Path(path) / "adapter_config.json"
    hidden_path = Path(path) / ".verl_adapter_config.json.hide"
    moved = False
    if adapter_path.is_file():
        try:
            payload = json.loads(adapter_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if not isinstance(payload, dict) or "base_model_name_or_path" not in payload:
            adapter_path.rename(hidden_path)
            moved = True
    try:
        yield
    finally:
        if moved and hidden_path.is_file() and not adapter_path.exists():
            hidden_path.rename(adapter_path)


def load_gr00t_n1d6_policy(path, *, config, torch_dtype):
    """Load the official policy while handling its meta-init Beta distribution."""
    import gr00t.model.gr00t_n1d6.gr00t_n1d6 as upstream_model

    apply_gr00t_compat_patches()

    with _BETA_PATCH_LOCK, no_init_weights(), _hide_non_peft_adapter_config(path):
        original_beta = upstream_model.Beta
        upstream_model.Beta = _CpuBeta
        try:
            return Gr00tN1d6.from_pretrained(path, config=config, torch_dtype=torch_dtype)
        finally:
            upstream_model.Beta = original_beta


def _cfg_get(obj: Any, name: str, default=None):
    """Read attribute treating explicit ``None`` as unset (gr00t override contract)."""
    val = getattr(obj, name, None)
    return default if val is None else val


class Gr00tN1d6TrainableModel(nn.Module, TrainableVLAModelMixin, SupportSACTraining, SupportSFTTraining):
    def __init__(self, policy: Gr00tN1d6, adapter_config: Gr00tAdapterConfig | None = None):
        super().__init__()
        if adapter_config is None:
            adapter_config = Gr00tAdapterConfig(model_path=getattr(policy.config, "_name_or_path", None))
        self.config = adapter_config
        SupportSFTTraining.__init__(self, self.config)
        self.init_trainable_model(policy=policy)
        self._adapter: Optional[GR00TN16Adapter] = None
        self._adapter_model_path = (
            adapter_config.adapter_model_path
            or adapter_config.model_path
            or getattr(policy.config, "_name_or_path", None)
        )

        policy_cfg = policy.config
        self.policy_type = str(getattr(adapter_config, "policy_type", "libero"))
        self.embodiment_tag = str(getattr(adapter_config, "embodiment_tag", GR1.name))

        self.action_horizon = int(_cfg_get(policy_cfg, "action_horizon", GR00TDim.ACTION_HORIZON))
        self.max_action_dim = int(_cfg_get(policy_cfg, "max_action_dim", GR00TDim.MAX_ACTION_DIM))
        self.max_state_dim = int(_cfg_get(policy_cfg, "max_state_dim", GR00TDim.MAX_STATE_DIM))
        self.backbone_feature_dim = int(_cfg_get(policy_cfg, "backbone_embedding_dim", 2048))
        self.num_inference_timesteps = int(_cfg_get(policy_cfg, "num_inference_timesteps", 4))
        self.num_timestep_buckets = int(_cfg_get(policy_cfg, "num_timestep_buckets", 1000))
        self.add_pos_embed = bool(_cfg_get(policy_cfg, "add_pos_embed", False))
        self.use_alternate_vl_dit = bool(_cfg_get(policy_cfg, "use_alternate_vl_dit", False))
        self.state_horizon = int(
            _cfg_get(
                adapter_config, "state_horizon", _cfg_get(adapter_config, "sac_state_horizon", GR00TDim.STATE_HORIZON)
            )
        )

        self.num_action_chunks = min(
            int(getattr(adapter_config, "num_action_chunks", GR00TDim.ACTION_HORIZON)),
            self.action_horizon,
        )
        self.action_dim = int(getattr(adapter_config, "action_dim", GR1.action_dim))
        self.embodiment_id = int(getattr(adapter_config, "embodiment_id", GR1.embodiment_id))

        critic_cfg = adapter_config.critic
        self.critic_type = str(critic_cfg.type).lower()
        self.critic_action_dim = int(critic_cfg.action_dim if critic_cfg.action_dim is not None else self.action_dim)
        self.critic_action_horizon = int(
            critic_cfg.action_horizon if critic_cfg.action_horizon is not None else self.num_action_chunks
        )
        self._state_feature_dim = int(getattr(policy.action_head, "input_embedding_dim", self.max_state_dim))

        base_state_width = self._state_feature_dim if critic_cfg.use_encoded_state else self.max_state_dim
        if (not critic_cfg.use_encoded_state) and critic_cfg.state_real_dim is not None:
            base_state_width = int(critic_cfg.state_real_dim)
        self._critic_state_width = base_state_width
        pooled_dim = int(critic_cfg.pool_proj_dim) if int(critic_cfg.pool_proj_dim) > 0 else self.backbone_feature_dim
        privileged_dim = int(critic_cfg.privileged_obs_dim or 0) if critic_cfg.privileged_obs else 0
        resolved_input_dim = (
            pooled_dim
            + self.state_horizon * self._critic_state_width
            + self.critic_action_horizon * self.critic_action_dim
            + privileged_dim
        )
        if critic_cfg.input_dim is None:
            critic_cfg.input_dim = resolved_input_dim
        self.critic_input_dim = int(critic_cfg.input_dim)

        # Flow-SDE
        self.flow_sde_enable = bool(getattr(adapter_config, "flow_sde_enable", False))
        self.flow_sde_noise_level = float(getattr(adapter_config, "flow_sde_noise_level", 0.065))
        self.flow_sde_rollout_noise_scale = float(getattr(adapter_config, "flow_sde_rollout_noise_scale", 1.0))
        self.flow_sde_train_noise_scale = float(getattr(adapter_config, "flow_sde_train_noise_scale", 1.0))
        per_dim = getattr(adapter_config, "flow_sde_noise_level_per_dim", None)
        if per_dim is not None:
            vec = torch.full((self.max_action_dim,), float(self.flow_sde_noise_level), dtype=torch.float32)
            for i, v in enumerate(list(per_dim)[: self.max_action_dim]):
                vec[i] = float(v)
            self.register_buffer("flow_sde_noise_level_vec", vec.view(1, 1, -1), persistent=False)
        else:
            self.flow_sde_noise_level_vec = None
        self.flow_sde_initial_beta = float(getattr(adapter_config, "flow_sde_initial_beta", 1.0))
        self.flow_sde_beta_min = float(getattr(adapter_config, "flow_sde_beta_min", 0.02))
        self.flow_sde_beta_schedule_T = int(getattr(adapter_config, "flow_sde_beta_schedule_T", 4000))
        self.register_buffer("flow_sde_step", torch.zeros((), dtype=torch.long))

        self.flow_sde_std_head = bool(getattr(adapter_config, "flow_sde_std_head", False))
        if self.flow_sde_std_head:
            std_in_dim = int(getattr(policy.action_head, "hidden_size", self.backbone_feature_dim))
            std_hidden = int(getattr(adapter_config, "flow_sde_std_head_hidden", 256))
            self.flow_sde_std_net = nn.Sequential(
                nn.Linear(std_in_dim, std_hidden),
                nn.ReLU(),
                nn.Linear(std_hidden, self.max_action_dim),
            )
            nn.init.zeros_(self.flow_sde_std_net[-1].weight)
            nn.init.constant_(self.flow_sde_std_net[-1].bias, math.log(max(self.flow_sde_noise_level, 1e-6)))
            self._flow_sde_log_std_min = math.log(1e-3)
            self._flow_sde_log_std_max = math.log(0.5)
            self._flow_sde_std_in_dim = std_in_dim

        train_dims = getattr(adapter_config, "sac_action_train_dims", None)
        mask = torch.zeros(self.max_action_dim, dtype=torch.bool)
        if train_dims is None:
            mask[:] = True
        else:
            for rng in train_dims:
                s, e = int(rng[0]), int(rng[1])
                mask[s:e] = True
        self.register_buffer("sac_action_train_mask", mask, persistent=False)
        self.sac_action_train_all = bool(mask.all().item())

        self.critic = None
        self.critic_api = None
        if self.config.critic.enabled:
            if self.critic_type not in CRITIC_BACKENDS:
                raise ValueError(f"Unsupported critic_type: {self.critic_type}")
            self.critic_api = CRITIC_BACKENDS[self.critic_type]
            self.critic_api.init(self)
            logger.info(
                "Gr00tN1d6TrainableModel: critic=%s heads=%d input_dim=%d pooling=%s",
                self.critic_type,
                int(critic_cfg.head_num),
                self.critic_input_dim,
                critic_cfg.pooling,
            )

        self.freeze_vision_tower_enabled = bool(getattr(adapter_config, "freeze_vision_tower", True))
        if self.freeze_vision_tower_enabled and self.config.critic.enabled:
            self.freeze_vision_tower()
        self.freeze_action_io_enabled = bool(getattr(adapter_config, "freeze_action_io", False))
        if self.freeze_action_io_enabled:
            self.freeze_action_io()

    @property
    def device(self):
        return next(self.policy.parameters()).device

    @property
    def action_head(self):
        return self.policy.action_head

    @property
    def backbone(self):
        return self.policy.backbone

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    def can_generate(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Policy adapters
    # ------------------------------------------------------------------

    def _get_policy_classes(self):
        return get_gr00t_policy_classes(self.policy_type)

    def _resolve_norm_stats_path(self) -> str | None:
        path = getattr(self.config, "norm_stats_path", None) or getattr(self.policy.config, "norm_stats_path", None)
        if path in (None, "", "null", "None"):
            return None
        return str(path)

    def _get_adapter(self, *, training: bool | None = None) -> GR00TN16Adapter:
        if self._adapter is None:
            if not self._adapter_model_path:
                raise RuntimeError(
                    "Gr00tN1d6TrainableModel: cannot build GR00TN16Adapter without a checkpoint path; "
                    "set adapter.model_path / adapter_model_path."
                )
            processor_path = getattr(self.policy.config, "verl_processor_path", None) or self._adapter_model_path
            norm_stats_path = self._resolve_norm_stats_path()
            self._adapter = GR00TN16Adapter(
                str(processor_path),
                embodiment_tag=self.embodiment_tag,
                norm_stats_path=norm_stats_path,
                override_modality_configs=getattr(self.config, "override_modality_configs", None),
                use_relative_action=getattr(self.config, "use_relative_action", None),
                training=bool(training) if training is not None else False,
            )
            # Projector index + real action width come from the processor/checkpoint.
            self.embodiment_id = int(self._adapter.embodiment_id)
            self.action_dim = int(self._adapter.action_dim)
        elif training is not None:
            self._adapter.set_training(training)
        return self._adapter

    def _prepare_inputs(
        self,
        obs: DataProto,
        tokenizer=None,
        *,
        actions: torch.Tensor | None = None,
        action_valid_mask: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> tuple[dict[str, torch.Tensor], Any]:
        del tokenizer
        adapter = self._get_adapter(training=training)
        input_cls, _ = self._get_policy_classes()
        if actions is not None and hasattr(input_cls, "from_data_proto"):
            model_input = input_cls.from_data_proto(obs, actions=actions)
            demo_actions = model_input.actions
        else:
            model_input = input_cls.from_env_obs(obs)
            demo_actions = actions if actions is not None else getattr(model_input, "actions", None)

        inputs, raw_state_groups = adapter.build_inputs(
            model_input.images,
            model_input.state,
            model_input.task,
            actions=demo_actions,
            action_valid_mask=action_valid_mask,
        )

        pixel_values = inputs["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values, dim=0)
        n_views = len(adapter.video_keys)
        batch_size = next(iter(model_input.images.values())).shape[0]
        pixel_values = pixel_values.reshape(batch_size, n_views, *pixel_values.shape[1:])
        s = {
            "images": pixel_values,
            "lang_tokens": inputs["input_ids"],
            "lang_masks": inputs["attention_mask"],
            "states": inputs["state"],
        }
        # Stash full processor inputs for official policy.forward / get_action.
        s["_processor_inputs"] = inputs
        return s, raw_state_groups

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def freeze_vision_tower(self) -> None:
        eagle = getattr(self.backbone, "eagle_model", None)
        vision_model = getattr(eagle, "vision_model", None) if eagle is not None else None
        if vision_model is None:
            logger.warning("[gr00t] backbone.eagle_model.vision_model not found; skipping freeze_vision_tower")
            return
        vision_model.requires_grad_(False)
        vision_model.eval()
        mlp1 = getattr(eagle, "mlp1", None)
        if mlp1 is not None:
            mlp1.requires_grad_(False)
            mlp1.eval()
        logger.info(
            "[gr00t] vision tower frozen (eagle_model.vision_model%s)",
            " + mlp1 connector" if mlp1 is not None else "",
        )

    def freeze_action_io(self) -> None:
        ah = self.action_head
        if ah is None:
            logger.warning("[gr00t] action_head not found; skipping freeze_action_io")
            return
        frozen = []
        for name in ("state_encoder", "action_encoder", "action_decoder"):
            mod = getattr(ah, name, None)
            if mod is None:
                logger.warning("[gr00t] action_head.%s not found; skipping", name)
                continue
            mod.requires_grad_(False)
            mod.eval()
            frozen.append(name)
        logger.info("[gr00t] action I/O frozen (action_head.%s)", ", action_head.".join(frozen))

    # ------------------------------------------------------------------
    # SFT / SAC init
    # ------------------------------------------------------------------

    def sft_init(self):
        self.sft_metrics = {}
        if self.freeze_vision_tower_enabled:
            self.freeze_vision_tower()
        register_fsdp_forward_method(self, "sft_loss")

    def sac_init(self):
        if self.config.critic.enabled and self.critic is None and self.critic_api is not None:
            self.critic_api.init(self)
        if self.freeze_vision_tower_enabled:
            self.freeze_vision_tower()
        for method in (
            "sac_sample_actions",
            "sac_forward_state_features",
            "sac_forward_actor",
            "sac_forward_critic",
            "sac_update_target_network",
            "sft_loss",
        ):
            register_fsdp_forward_method(self, method)

    # ------------------------------------------------------------------
    # SFT / BC loss
    # ------------------------------------------------------------------

    def sft_loss(self, obs, tokenizer, actions, valids, action_mask=None, target_values=None):
        del target_values
        if actions is None or valids is None:
            raise ValueError("Gr00tN1d6TrainableModel.sft_loss requires both `actions` and `valids`.")
        action_tensor = actions.get("action") if isinstance(actions, dict) else None
        s, raw_state_groups = self._prepare_inputs(
            obs,
            tokenizer,
            actions=action_tensor,
            action_valid_mask=action_mask,
            training=True,
        )
        state_features = self._state_features_impl(s)
        # Prefer Adapter-normalised demos when embodiment IO + processor already ran.
        proc_inputs = s.get("_processor_inputs") or {}
        if action_tensor is not None and "action" in proc_inputs:
            bc_actions: dict[str, torch.Tensor] = {"full_action": proc_inputs["action"]}
            bc_mask = proc_inputs.get("action_mask", action_mask)
        else:
            bc_actions = actions
            bc_mask = action_mask
        return self._bc_mse(
            state_features,
            bc_actions,
            valids,
            action_mask=bc_mask,
            raw_state_groups=raw_state_groups,
        )

    # ------------------------------------------------------------------
    # Rollout / critic value
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sac_sample_actions(self, obs: DataProto, tokenizer=None, eval: bool = False):
        _, output_cls = self._get_policy_classes()
        s, raw_state_groups = self._prepare_inputs(obs, tokenizer, training=False)
        state_features = self._state_features_impl(s)

        override = obs.meta_info.get("rollout_noise_scale", None) if obs.meta_info else None
        if eval:
            noise_scale = 0.0
        elif override is not None:
            noise_scale = float(override)
        else:
            noise_scale = self.flow_sde_rollout_noise_scale if self.flow_sde_enable else 0.0
        return_log_prob = self.flow_sde_enable and noise_scale > 0.0

        full_action, log_probs = self._denoise(
            state_features, noise_scale=noise_scale, requires_grad=False, return_log_prob=return_log_prob
        )
        full_action_norm = full_action.detach().float()
        decoded_flat = self._get_adapter().decode_actions_flat(full_action_norm.cpu().numpy(), raw_state_groups)
        decoded = torch.as_tensor(decoded_flat, dtype=torch.float32, device=full_action_norm.device)
        return output_cls.from_model_output(
            {
                "full_action": full_action_norm,
                "decoded_action": decoded,
                "log_probs": log_probs,
                "num_action_chunks": self.num_action_chunks,
            }
        )

    @torch.no_grad()
    def sac_get_critic_value(
        self,
        obs: DataProto,
        actions: ModelOutput,
        tokenizer: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        state_features = self.sac_forward_state_features(obs, tokenizer)
        q_values = self.sac_forward_critic(
            {"full_action": actions.full_action},
            state_features,
            use_target_network=False,
            method="min",
            requires_grad=False,
        )
        return q_values.detach().float().reshape(-1)

    # ------------------------------------------------------------------
    # State features
    # ------------------------------------------------------------------

    def sac_forward_state_features(
        self, obs: DataProto, tokenizer: Optional[torch.nn.Module] = None
    ) -> dict[str, torch.Tensor]:
        s, _ = self._prepare_inputs(obs, tokenizer)
        return self._state_features_impl(s)

    def _state_features_impl(self, s: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ah = self.action_head
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype

        imgs = s["images"]
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.flatten(0, 1)
            pixel_values = [imgs[i].to(device, dtype=dtype) for i in range(imgs.shape[0])]
        else:
            pixel_values = [t.to(device, dtype=dtype) for t in imgs]

        backbone_inputs = BatchFeature(
            data={
                "input_ids": s["lang_tokens"].to(device).long(),
                "attention_mask": s["lang_masks"].to(device),
                "pixel_values": pixel_values,
            }
        )
        backbone_outputs = self.backbone(backbone_inputs)

        raw_features = backbone_outputs["backbone_features"]
        attn_mask = backbone_outputs["backbone_attention_mask"]
        image_mask = backbone_outputs.get("image_mask", None)

        vl_embeds = ah.vlln(raw_features)
        mask_b = attn_mask.unsqueeze(-1).bool()
        vl_safe = torch.where(mask_b, vl_embeds, torch.zeros_like(vl_embeds))
        denom = mask_b.sum(dim=1).clamp(min=1).to(vl_embeds.dtype)
        pooled = vl_safe.sum(dim=1) / denom

        state = s["states"].to(device, dtype=dtype)
        batch_size = state.shape[0]
        embodiment_id = torch.full((batch_size,), self.embodiment_id, dtype=torch.long, device=device)
        state_features = ah.state_encoder(state, embodiment_id)

        out = {
            "pooled": pooled,
            "backbone_features": vl_embeds,
            "backbone_attention_mask": attn_mask,
            "state_features": state_features,
            "state": state,
            "embodiment_id": embodiment_id,
        }
        if image_mask is not None:
            out["image_mask"] = image_mask
        if self.config.critic.privileged_obs and "priv_obs" in s:
            out["priv_obs"] = s["priv_obs"].to(device, dtype=dtype)
        return out

    # ------------------------------------------------------------------
    # Flow-SDE actor
    # ------------------------------------------------------------------

    def _gaussian_log_prob(self, sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        std_safe = std.clamp_min(1e-6)
        log_prob = -0.5 * (((sample - mean) / std_safe) ** 2 + 2.0 * torch.log(std_safe) + math.log(2.0 * math.pi))
        if self.sac_action_train_all:
            return log_prob.mean(dim=(-1, -2))
        m = self.sac_action_train_mask.view(1, 1, -1)
        n_train = int(self.sac_action_train_mask.sum().item())
        log_prob = log_prob.masked_fill(~m, 0.0).sum(dim=(-1, -2))
        return log_prob / max(self.action_horizon * n_train, 1)

    def flow_sde_beta(self) -> torch.Tensor:
        beta = beta_schedule(
            int(self.flow_sde_step.item()),
            beta0=self.flow_sde_initial_beta,
            beta_min=self.flow_sde_beta_min,
            T=self.flow_sde_beta_schedule_T,
        )
        return torch.tensor(beta, device=self.flow_sde_step.device, dtype=torch.float32)

    def _flow_sde_noise_level(self, dtype=None, device=None, features=None):
        if getattr(self, "flow_sde_std_head", False) and features is not None:
            feat = features[:, -self.action_horizon :]
            assert feat.shape[-1] == self._flow_sde_std_in_dim, (
                f"flow_sde_std_head in_dim {self._flow_sde_std_in_dim} != DiT feature dim "
                f"{feat.shape[-1]}; std_in_dim should be action_head.hidden_size"
            )
            sigma0 = self.flow_sde_std_net(feat).clamp(self._flow_sde_log_std_min, self._flow_sde_log_std_max).exp()
            if dtype is not None or device is not None:
                sigma0 = sigma0.to(dtype=dtype, device=device)
            return sigma0
        vec = getattr(self, "flow_sde_noise_level_vec", None)
        if vec is not None:
            if dtype is not None or device is not None:
                return vec.to(dtype=dtype, device=device)
            return vec
        return self.flow_sde_noise_level

    def _denoise(
        self,
        sf: dict[str, torch.Tensor],
        *,
        noise_scale: float,
        requires_grad: bool,
        return_log_prob: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        vl_embeds = sf["backbone_features"]
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype
        x0 = torch.randn((batch_size, self.action_horizon, self.max_action_dim), dtype=dtype, device=device)

        if self.sac_action_train_all:
            return self._run_flow(
                sf, x0, noise_scale=noise_scale, requires_grad=requires_grad, return_log_prob=return_log_prob
            )

        x_train, log_probs = self._run_flow(
            sf, x0, noise_scale=noise_scale, requires_grad=requires_grad, return_log_prob=return_log_prob
        )
        with torch.no_grad():
            x_base, _ = self._run_flow(sf, x0, noise_scale=0.0, requires_grad=False, return_log_prob=False)
        mask = self.sac_action_train_mask.view(1, 1, -1)
        x = torch.where(mask, x_train, x_base)
        return x, log_probs

    def _run_flow(
        self,
        sf: dict[str, torch.Tensor],
        x: torch.Tensor,
        *,
        noise_scale: float,
        requires_grad: bool,
        return_log_prob: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Flow-matching trajectory. GR00T: noise (t=0) -> action (t=1)."""
        ah = self.action_head
        vl_embeds = sf["backbone_features"]
        attn_mask = sf["backbone_attention_mask"]
        image_mask = sf.get("image_mask")
        state_features = sf["state_features"]
        emb_id = sf["embodiment_id"]

        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype

        num_steps = self.num_inference_timesteps
        use_sde = self.flow_sde_enable and noise_scale > 0.0
        beta = self.flow_sde_beta().to(device=device, dtype=dtype) if use_sde else None
        step_log_probs: list[torch.Tensor] = []
        ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()

        with ctx:
            for i in range(num_steps):
                t_cur = i / float(num_steps)
                t_next = (i + 1) / float(num_steps)
                delta = t_next - t_cur

                t_disc = int(t_cur * self.num_timestep_buckets)
                timesteps = torch.full((batch_size,), t_disc, device=device)

                action_features = ah.action_encoder(x, timesteps, emb_id)
                if self.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    action_features = action_features + ah.position_embedding(pos_ids).unsqueeze(0)

                sa_embs = torch.cat((state_features, action_features), dim=1)
                if self.use_alternate_vl_dit:
                    model_output = ah.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps,
                        image_mask=image_mask,
                        backbone_attention_mask=attn_mask,
                    )
                else:
                    model_output = ah.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps,
                    )
                pred = ah.action_decoder(model_output, emb_id)
                v = pred[:, -self.action_horizon :]

                if use_sde:
                    sigma0 = self._flow_sde_noise_level(dtype=dtype, device=device, features=model_output)
                    if self.flow_sde_std_head:
                        x_mean = x + delta * v
                        sigma_t = sigma0 * noise_scale
                    else:
                        data_pred = x + v * (1.0 - t_cur)
                        noise_pred = x - v * t_cur
                        s_cur = min(max(1.0 - t_cur, 1e-4), 1.0 - 1e-4)
                        s_next = 1.0 - t_next
                        sigma = beta * (sigma0 * noise_scale * math.sqrt(s_cur / (1.0 - s_cur)))
                        data_w = 1.0 - s_next
                        noise_w = s_next - (sigma**2 * delta) / (2.0 * s_cur)
                        x_mean = data_pred * data_w + noise_pred * noise_w
                        sigma_t = math.sqrt(delta) * sigma
                        sigma_t = torch.as_tensor(sigma_t, device=device, dtype=dtype).reshape(1, 1, -1)
                    eps = torch.randn_like(x)
                    x = x_mean + sigma_t * eps
                    if return_log_prob:
                        step_log_probs.append(self._gaussian_log_prob(x, x_mean, sigma_t))
                else:
                    x = x + delta * v

        log_probs = torch.stack(step_log_probs, dim=1).sum(dim=1) if step_log_probs else None
        return x, log_probs

    def sac_forward_actor(
        self,
        state_features: dict[str, torch.Tensor],
        task_ids: Optional[torch.Tensor] = None,
        is_first_micro_batch: bool = False,
        noise_scale: Optional[float] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, float]]:
        del task_ids
        resolved_noise_scale = (
            (self.flow_sde_train_noise_scale if self.flow_sde_enable else 0.0)
            if noise_scale is None
            else float(noise_scale)
        )
        actions, log_probs = self._denoise(
            state_features,
            noise_scale=resolved_noise_scale,
            requires_grad=True,
            return_log_prob=self.flow_sde_enable and resolved_noise_scale > 0.0,
        )
        if is_first_micro_batch and self.flow_sde_enable:
            self.flow_sde_step.add_(1)
        metrics: dict[str, float] = {}
        if self.flow_sde_enable:
            metrics = {
                "flow_sde_beta": float(self.flow_sde_beta().item()),
                "flow_sde_step": float(self.flow_sde_step.item()),
                "flow_sde_noise_scale": float(resolved_noise_scale),
            }
        return actions, log_probs, metrics

    # ------------------------------------------------------------------
    # Critic API
    # ------------------------------------------------------------------

    def sac_forward_critic(
        self,
        a: dict[str, torch.Tensor],
        state_features: dict[str, torch.Tensor],
        task_ids: Optional[torch.Tensor] = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        if self.critic_api is None:
            raise RuntimeError("Critic is not enabled; set adapter.critic.enabled=True")
        return self.critic_api.forward(
            self,
            a=a,
            state_features=state_features,
            task_ids=task_ids,
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    def sac_get_critic_parameters(self) -> list[torch.nn.Parameter]:
        if self.critic_api is None:
            raise RuntimeError("Critic is not enabled; set adapter.critic.enabled=True")
        return self.critic_api.get_critic_parameters(self)

    def sac_get_named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        actor_params = []
        for name, p in self.action_head.named_parameters():
            if p.requires_grad:
                actor_params.append((f"policy.action_head.{name}", p))
        for name, p in self.backbone.named_parameters():
            if p.requires_grad:
                actor_params.append((f"policy.backbone.{name}", p))
        return actor_params

    @torch.no_grad()
    def sac_update_target_network(self, tau: float):
        if self.critic_api is None:
            raise RuntimeError("Critic is not enabled; set adapter.critic.enabled=True")
        self.critic_api.update_target_network(self, tau)

    # ------------------------------------------------------------------
    # BC helpers (flow-matching, noise→action)
    # ------------------------------------------------------------------

    def _demo_action_normalized(
        self,
        actions: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        raw_state_groups: Optional[Any] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "full_action" in actions:
            demo = actions["full_action"].to(device=device, dtype=dtype)
            B, H, D = demo.shape
            if H > self.action_horizon:
                demo = demo[:, : self.action_horizon]
                H = self.action_horizon
            elif H < self.action_horizon:
                demo = F.pad(demo, (0, 0, 0, self.action_horizon - H), value=0.0)
            if D < self.max_action_dim:
                demo = F.pad(demo, (0, self.max_action_dim - D), value=0.0)
            elif D > self.max_action_dim:
                demo = demo[..., : self.max_action_dim]
            mask = torch.zeros((B, self.action_horizon, self.max_action_dim), device=device, dtype=dtype)
            real_h = min(H, self.action_horizon)
            mask[:, :real_h, : self.action_dim] = 1.0
            return demo, mask

        if "action" not in actions:
            raise KeyError("Gr00t BC requires `full_action` (normalised) or env-space `action` in the actions dict.")
        if raw_state_groups is None:
            raise ValueError("Env-space `action` demos require raw_state_groups from `_prepare_inputs`.")
        input_cls, _ = self._get_policy_classes()
        env_act = input_cls.actions_to_processor_space(actions["action"])
        if env_act.ndim != 3:
            raise ValueError(f"actions['action'] must be (B, horizon, dim), got {tuple(env_act.shape)}")
        H = env_act.shape[1]
        if H > self.action_horizon:
            env_act = env_act[:, : self.action_horizon]
        elif H < self.action_horizon:
            env_act = F.pad(env_act, (0, 0, 0, self.action_horizon - H), value=0.0)
        norm_np, mask_np = self._get_adapter().encode_actions_flat(
            env_act.detach().float().cpu().numpy(),
            raw_state_groups,
            max_action_dim=self.max_action_dim,
            max_action_horizon=self.action_horizon,
        )
        demo = torch.as_tensor(norm_np, device=device, dtype=dtype)
        mask = torch.as_tensor(mask_np, device=device, dtype=dtype)
        return demo, mask

    def _predict_flow_velocity(
        self,
        sf: dict[str, torch.Tensor],
        x_t: torch.Tensor,
        t_discretized: torch.Tensor,
    ) -> torch.Tensor:
        ah = self.action_head
        vl_embeds = sf["backbone_features"]
        attn_mask = sf["backbone_attention_mask"]
        image_mask = sf.get("image_mask")
        state_features = sf["state_features"]
        emb_id = sf["embodiment_id"]
        device = vl_embeds.device

        action_features = ah.action_encoder(x_t, t_discretized, emb_id)
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + ah.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)
        if self.use_alternate_vl_dit:
            model_output = ah.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_discretized,
                image_mask=image_mask,
                backbone_attention_mask=attn_mask,
            )
        else:
            model_output = ah.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_discretized,
            )
        pred = ah.action_decoder(model_output, emb_id)
        return pred[:, -self.action_horizon :]

    def _bc_mse(
        self,
        state_features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        raw_state_groups: Optional[Any] = None,
    ) -> torch.Tensor:
        vl = state_features["backbone_features"]
        device, dtype = vl.device, vl.dtype
        demo, dense_mask = self._demo_action_normalized(
            actions, device=device, dtype=dtype, raw_state_groups=raw_state_groups
        )
        batch_size = demo.shape[0]

        noise = torch.randn(demo.shape, device=device, dtype=dtype)
        ah = self.action_head
        if hasattr(ah, "sample_time"):
            t = ah.sample_time(batch_size, device=device, dtype=dtype)
        else:
            t = torch.rand(batch_size, device=device, dtype=dtype)
        t_exp = t[:, None, None]
        x_t = (1.0 - t_exp) * noise + t_exp * demo
        u_t = demo - noise
        t_disc = (t * self.num_timestep_buckets).long()

        pred_v = self._predict_flow_velocity(state_features, x_t, t_disc)
        per_elem = F.mse_loss(pred_v, u_t, reduction="none")

        mask = dense_mask
        if action_mask is not None:
            am = action_mask.to(device=device, dtype=dtype)
            if am.ndim == 2:
                if am.shape[1] != self.action_horizon:
                    if am.shape[1] < self.action_horizon:
                        am = F.pad(am, (0, self.action_horizon - am.shape[1]), value=0.0)
                    else:
                        am = am[:, : self.action_horizon]
                am = am.unsqueeze(-1).expand_as(mask)
            elif am.ndim == 3:
                if am.shape[1] != self.action_horizon or am.shape[2] != self.max_action_dim:
                    h = min(am.shape[1], self.action_horizon)
                    d = min(am.shape[2], self.max_action_dim)
                    aligned = torch.zeros_like(mask)
                    aligned[:, :h, :d] = am[:, :h, :d]
                    am = aligned
            else:
                raise ValueError(f"action_mask must be (B,H) or (B,H,D), got {tuple(am.shape)}")
            mask = mask * am

        masked = per_elem * mask
        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
        sample_loss = masked.sum(dim=(1, 2)) / denom
        valid_f = valids.to(device=device, dtype=sample_loss.dtype)
        return (sample_loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    @staticmethod
    def extract_critic_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Extract critic (+ target) weights from a full adapter state dict."""
        return extract_critic_state_dict(state_dict)

    @staticmethod
    def normalize_adapter_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap legacy critic key prefixes (aligned with pi05 ``load_state_dict``)."""
        return normalize_adapter_state_dict(state_dict)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Preserve complete adapter checkpoints, including critic/target state.
        normalized_state_dict = normalize_adapter_state_dict(state_dict)
        return nn.Module.load_state_dict(self, normalized_state_dict, strict=strict, assign=assign)

    def export_policy(self, output_dir, *, state_dict=None):
        """Export native GR00T weights plus optional SAC critic sidecar.

        Policy weights go through ``policy.save_pretrained`` (HF layout). When a
        critic is present, its state dict is written to ``critic.pt`` so SAC
        resume / distillation can reload Q-heads without the full FSDP shard.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processor = None
        try:
            processor = self._get_adapter(training=False).processor
        except Exception:
            processor = None
        policy_state = self.extract_policy_state_dict(state_dict) if state_dict is not None else None
        original_processor_path = getattr(self.policy.config, "verl_processor_path", None)
        original_norm_stats_path = getattr(self.policy.config, "norm_stats_path", None)
        original_architectures = getattr(self.policy.config, "architectures", None)
        self.policy.config.verl_processor_path = None
        self.policy.config.norm_stats_path = None
        self.policy.config.architectures = ["Gr00tN1d6"]
        try:
            self.native_policy.save_pretrained(output_dir, state_dict=policy_state, safe_serialization=True)
            if processor is not None:
                processor.save_pretrained(output_dir)
            self.config.save_pretrained(output_dir)
        finally:
            self.policy.config.verl_processor_path = original_processor_path
            self.policy.config.norm_stats_path = original_norm_stats_path
            self.policy.config.architectures = original_architectures

        critic_state = None
        if state_dict is not None:
            critic_state = extract_critic_state_dict(normalize_adapter_state_dict(state_dict))
        elif self.critic is not None:
            critic_state = self.critic.state_dict()
        if critic_state:
            torch.save(critic_state, output_dir / "critic.pt")


__all__ = ["Gr00tN1d6TrainableModel", "load_gr00t_n1d6_policy", "beta_schedule"]
