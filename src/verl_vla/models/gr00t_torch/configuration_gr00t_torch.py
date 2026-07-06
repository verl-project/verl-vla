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

from transformers import PretrainedConfig

DEFAULT_DIFFUSION_MODEL_CFG = {
    "positional_embeddings": None,
    "num_layers": 16,
    "num_attention_heads": 32,
    "attention_head_dim": 48,
    "norm_type": "ada_norm",
    "dropout": 0.2,
    "final_dropout": True,
    "output_dim": 1024,
    "interleave_self_attention": True,
}


class Gr00tTorchConfig(PretrainedConfig):
    model_type = "gr00t_torch"

    def __init__(self, **kwargs):
        kwargs["architectures"] = ["Gr00tForConditionalGeneration"]
        super().__init__(**kwargs)

        # ---- GR00T N1.7 backbone (Cosmos-Reason2-2B, Qwen3-VL architecture) ----
        self.model_name = kwargs.get("model_name", "nvidia/Cosmos-Reason2-2B")
        self.select_layer = kwargs.get("select_layer", 12)
        self.backbone_embedding_dim = kwargs.get("backbone_embedding_dim", 2048)
        self.tune_llm = kwargs.get("tune_llm", False)
        self.tune_visual = kwargs.get("tune_visual", False)
        self.tune_top_llm_layers = kwargs.get("tune_top_llm_layers", 0)
        self.reproject_vision = kwargs.get("reproject_vision", False)
        self.use_flash_attention = kwargs.get("use_flash_attention", False)
        self.load_bf16 = kwargs.get("load_bf16", False)
        self.backbone_trainable_params_fp32 = kwargs.get("backbone_trainable_params_fp32", True)

        # ---- GR00T N1.7 action head (flow-matching DiT) ----
        self.max_state_dim = kwargs.get("max_state_dim", 132)
        self.max_action_dim = kwargs.get("max_action_dim", 132)
        self.action_horizon = kwargs.get("action_horizon", 40)
        self.hidden_size = kwargs.get("hidden_size", 1024)
        self.input_embedding_dim = kwargs.get("input_embedding_dim", 1536)
        self.state_history_length = kwargs.get("state_history_length", 1)
        self.add_pos_embed = kwargs.get("add_pos_embed", True)
        self.use_vlln = kwargs.get("use_vlln", True)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.use_alternate_vl_dit = kwargs.get("use_alternate_vl_dit", True)
        self.attend_text_every_n_blocks = kwargs.get("attend_text_every_n_blocks", 2)
        self.diffusion_model_cfg = kwargs.get("diffusion_model_cfg", dict(DEFAULT_DIFFUSION_MODEL_CFG))
        self.vl_self_attention_cfg = kwargs.get("vl_self_attention_cfg", None)
        self.tune_projector = kwargs.get("tune_projector", True)
        self.tune_diffusion_model = kwargs.get("tune_diffusion_model", True)
        self.tune_vlln = kwargs.get("tune_vlln", True)
        self.max_num_embodiments = kwargs.get("max_num_embodiments", 32)

        # ---- Flow matching ----
        self.num_inference_timesteps = kwargs.get("num_inference_timesteps", 4)
        self.noise_beta_alpha = kwargs.get("noise_beta_alpha", 1.5)
        self.noise_beta_beta = kwargs.get("noise_beta_beta", 1.0)
        self.noise_s = kwargs.get("noise_s", 0.999)
        self.num_timestep_buckets = kwargs.get("num_timestep_buckets", 1000)

        # RL keeps the state as a real input: no dropout by default (upstream pretrain uses 0.8).
        self.state_dropout_prob = kwargs.get("state_dropout_prob", 0.0)

        # ---- verl-vla integration ----
        self.policy_type = kwargs.get("policy_type", "libero")
        self.embodiment_id = kwargs.get("embodiment_id", 2)  # libero_sim projector index
        self.action_chunk_size = kwargs.get("action_chunk_size", 10)
        # Supervised horizon for SFT; upstream libero_sim uses delta_indices=range(16).
        self.sft_action_horizon = kwargs.get("sft_action_horizon", 16)
        self.state_norm_stats = kwargs.get("state_norm_stats", {})
        self.action_norm_stats = kwargs.get("action_norm_stats", {})
        # Qwen3-VL processor path; None falls back to the checkpoint directory itself.
        self.vlm_processor_path = kwargs.get("vlm_processor_path", None)
        self.formalize_language = kwargs.get("formalize_language", True)
        self.image_target_size = kwargs.get("image_target_size", (256, 256))
        self.image_crop_size = kwargs.get("image_crop_size", (230, 230))

        # ---- SAC critic (names shared with PI0 so pi0_torch critic backends work) ----
        self.sac_enable = kwargs.get("sac_enable", False)
        self.critic_type = kwargs.get("critic_type", "cross_attn")
        self.critic_num = kwargs.get("critic_num", 1)
        self.critic_task_to_critic = kwargs.get("critic_task_to_critic", None)
        # 2048 (pooled VL embedding) + 8 (libero state) + 10 * 7 (env action chunk)
        self.critic_input_dim = kwargs.get("critic_input_dim", 2126)
        self.critic_hidden_dims = kwargs.get("critic_hidden_dims", [1024, 512, 256])
        self.critic_prefix_embed_dim = kwargs.get("critic_prefix_embed_dim", 2048)

        # ---- Flow-SDE stochastic sampling (same knob set as PI0) ----
        self.flow_sde_enable = kwargs.get("flow_sde_enable", True)
        self.flow_sde_noise_level = kwargs.get("flow_sde_noise_level", 0.5)
        self.flow_sde_noise_schedule_enabled = kwargs.get("flow_sde_noise_schedule_enabled", False)
        self.flow_sde_noise_schedule_initial = kwargs.get("flow_sde_noise_schedule_initial", None)
        self.flow_sde_noise_schedule_final = kwargs.get("flow_sde_noise_schedule_final", None)
        self.flow_sde_noise_schedule_method = kwargs.get("flow_sde_noise_schedule_method", "cos")
        self.flow_sde_task_noise_level = kwargs.get("flow_sde_task_noise_level", {})
        self.flow_sde_rollout_noise_scale = kwargs.get("flow_sde_rollout_noise_scale", 1.0)
        self.flow_sde_train_noise_scale = kwargs.get("flow_sde_train_noise_scale", 1.0)
        self.flow_sde_beta_schedule_T = kwargs.get("flow_sde_beta_schedule_T", 2000)
        # Optional override of denoising steps for SDE sampling (e.g. 8); None keeps
        # num_inference_timesteps.
        self.flow_sde_num_inference_timesteps = kwargs.get("flow_sde_num_inference_timesteps", None)
        # Restrict the per-step Gaussian log-prob to valid action dims/steps. GR00T pads
        # actions to (40, 132); unmasked log-probs would be dominated by pure-noise padding.
        self.flow_sde_logprob_masked = kwargs.get("flow_sde_logprob_masked", True)
