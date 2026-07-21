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
        framework_fields = {
            "policy_type",
            "n_action_steps",
            "temporal_ensemble_coeff",
            "state_norm_stats",
            "action_norm_stats",
            "image_norm_stats",
            "sac_enable",
            "critic_type",
            "critic_head_num",
            "critic_prefix_embed_dim",
            "critic_input_dim",
            "critic_hidden_dims",
            "sac_action_noise_scale",
            "sac_action_noise_schedule_enabled",
            "sac_action_noise_schedule_initial",
            "sac_action_noise_schedule_final",
            "sac_action_noise_schedule_method",
            "freeze_vision_tower",
            "optimizer_lr_backbone",
        }
        for field in framework_fields:
            kwargs.pop(field, None)
        kwargs["architectures"] = ["ACT"]
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

        self.dropout = kwargs.get("dropout", 0.1)
        self.kl_weight = kwargs.get("kl_weight", 10.0)

        self.chunk_size = kwargs.get("chunk_size", 100)
        self.action_dim = kwargs.get("action_dim", 14)
        self.state_dim = kwargs.get("state_dim", 14)
        self.env_state_dim = kwargs.get("env_state_dim", 0)

        self.num_cameras = kwargs.get("num_cameras", 1)
        self.image_resolution = kwargs.get("image_resolution", (224, 224))

        self.attn_implementation = kwargs.get("attn_implementation", "eager")

        self.architectures = kwargs.get("architectures", None)
