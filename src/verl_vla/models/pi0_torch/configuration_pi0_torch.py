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


class PI0TorchConfig(PretrainedConfig):
    model_type = "pi0_torch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.state_norm_stats = kwargs.get("state_norm_stats", {})
        self.action_norm_stats = kwargs.get("action_norm_stats", {})
        self.pi05_enabled = kwargs.get("pi05_enabled", False)
        self.policy_type = kwargs.get("policy_type", "libero")
        self.critic_type = kwargs.get("critic_type", "cross_attn")
        self.critic_num = kwargs.get("critic_num", 1)
        self.critic_task_to_critic = kwargs.get("critic_task_to_critic", None)
        self.flow_sde_noise_schedule_enabled = kwargs.get("flow_sde_noise_schedule_enabled", False)
        self.flow_sde_noise_schedule_initial = kwargs.get("flow_sde_noise_schedule_initial", None)
        self.flow_sde_noise_schedule_final = kwargs.get("flow_sde_noise_schedule_final", None)
        self.flow_sde_noise_schedule_method = kwargs.get("flow_sde_noise_schedule_method", "cos")
        self.flow_sde_task_noise_level = kwargs.get("flow_sde_task_noise_level", {})
