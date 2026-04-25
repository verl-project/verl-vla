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
import logging
import os
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from verl import DataProto
from verl.utils.device import get_device_name
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    """HF rollout that reuses the actor-side FSDP engine/module."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        engine=None,
        module=None,
        tokenizer: Any = None,
        **kwargs,
    ):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        self.engine = engine
        self.module = module if module is not None else (engine.module if engine is not None else None)
        self.tokenizer = tokenizer
        rollout_custom_config = config.custom or {}
        self.output_critic_value = bool(
            config.get("output_critic_value", rollout_custom_config.get("output_critic_value", True))
        )

        if self.module is None:
            raise ValueError("HFRollout requires a shared actor engine or module.")

        from torch.distributed.fsdp import register_fsdp_forward_method

        register_fsdp_forward_method(self.module, "sac_sample_actions")
        if self.output_critic_value:
            register_fsdp_forward_method(self.module, "sac_get_critic_value")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            validate = bool(prompts.meta_info.get("validate", False))
            output = self.module.sac_sample_actions(
                prompts,
                tokenizer=self.tokenizer,
                validate=validate,
            )

        ret = output.to_data_proto()
        if self.output_critic_value:
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                critic_value = self.module.sac_get_critic_value(prompts, output, self.tokenizer)
            ret.batch["critic_value"] = critic_value
        return ret

    async def update_weights(self, weights, **kwargs):
        return None

    async def release(self):
        return None

    async def resume(self, tags=None):
        return None
