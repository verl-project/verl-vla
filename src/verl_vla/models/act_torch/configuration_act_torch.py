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

from transformers import PretrainedConfig


class ACTTorchConfig(PretrainedConfig):
    model_type = "act"

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __init__(self, **kwargs):
        kwargs["architectures"] = ["ACTForConditionalGeneration"]
        super().__init__(**kwargs)

        self.vision_backbone = kwargs.get("vision_backbone", "resnet18")
        self.pretrained_backbone_weights = kwargs.get("pretrained_backbone_weights", "ResNet18_Weights.IMAGENET1K_V1")
        self.replace_final_stride_with_dilation = kwargs.get("replace_final_stride_with_dilation", False)
        self.pre_norm = kwargs.get("pre_norm", False)

        self.dim_model = kwargs.get("dim_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.dim_feedforward = kwargs.get("dim_feedforward", 3200)
        self.feedforward_activation = kwargs.get("feedforward_activation", "relu")
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 4)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 1)

        self.use_vae = kwargs.get("use_vae", True)
        self.latent_dim = kwargs.get("latent_dim", 32)
        self.n_vae_encoder_layers = kwargs.get("n_vae_encoder_layers", 4)

        self.temporal_ensemble_coeff = kwargs.get("temporal_ensemble_coeff", None)
        self.dropout = kwargs.get("dropout", 0.1)
        self.kl_weight = kwargs.get("kl_weight", 10.0)

        self.chunk_size = kwargs.get("chunk_size", 100)
        self.n_action_steps = kwargs.get("n_action_steps", 100)
        self.action_dim = kwargs.get("action_dim", 14)
        self.state_dim = kwargs.get("state_dim", 14)
        self.env_state_dim = kwargs.get("env_state_dim", 0)

        self.num_cameras = kwargs.get("num_cameras", 1)
        self.image_resolution = kwargs.get("image_resolution", (224, 224))

        self.state_norm_stats = kwargs.get("state_norm_stats", None)
        self.action_norm_stats = kwargs.get("action_norm_stats", None)
        self.image_norm_stats = kwargs.get("image_norm_stats", None)

        self.policy_type = kwargs.get("policy_type", "libero")

        self.sac_enable = kwargs.get("sac_enable", False)
        self.critic_type = kwargs.get("critic_type", "mean_pool")
        self.critic_head_num = kwargs.get("critic_head_num", 2)
        self.critic_prefix_embed_dim = kwargs.get("critic_prefix_embed_dim", 512)
        self.critic_input_dim = kwargs.get("critic_input_dim", 0)
        if self.critic_input_dim <= 0:
            self.critic_input_dim = (
                self.critic_prefix_embed_dim + self.state_dim + (self.n_action_steps * self.action_dim)
            )
        self.critic_hidden_dims = kwargs.get("critic_hidden_dims", [1024, 512, 256])

        self.sac_action_noise_scale = kwargs.get("sac_action_noise_scale", 0.1)
        self.sac_action_noise_schedule_enabled = kwargs.get("sac_action_noise_schedule_enabled", False)
        self.sac_action_noise_schedule_initial = kwargs.get("sac_action_noise_schedule_initial", None)
        self.sac_action_noise_schedule_final = kwargs.get("sac_action_noise_schedule_final", None)
        self.sac_action_noise_schedule_method = kwargs.get("sac_action_noise_schedule_method", "cos")

        self.attn_implementation = kwargs.get("attn_implementation", "eager")

        self.architectures = kwargs.get("architectures", None)

        self.freeze_vision_tower = kwargs.get("freeze_vision_tower", True)
        self.optimizer_lr_backbone = kwargs.get("optimizer_lr_backbone", 1e-5)
