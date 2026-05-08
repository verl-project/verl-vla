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
from verl.workers.config import HFModelConfig
from verl.workers.rollout.base import BaseRollout

from verl_vla.workers.config import RolloutConfig

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
        self.tokenizer = tokenizer if tokenizer is not None else model_config.tokenizer
        self.output_critic_value = bool(config.output_critic_value)

        if self.module is None:
            logger.info("No shared actor engine provided, loading model from path...")
            
            # 像 trainer 那样正确加载模型
            from verl.utils.transformers_compat import get_auto_model_for_vision2seq
            from verl.utils.model import get_hf_auto_model_class, update_model_config
            from transformers import AutoConfig
            
            # 1. 获取正确的 AutoClass
            AutoModelForVision2Seq = get_auto_model_for_vision2seq()
            
            # 2. 加载并应用 override_config 到 hf_config
            hf_config = AutoConfig.from_pretrained(
                model_config.path,
                trust_remote_code=True,
            )
            
            # 应用 override_config，确保 sac_enable=True
            if hasattr(model_config, "override_config") and model_config.override_config:
                override_config = (
                    model_config.override_config["model_config"] 
                    if "model_config" in model_config.override_config 
                    else model_config.override_config
                )
                update_model_config(hf_config, override_config)
            
            # 确保 sac_enable 为 True（如果需要 critic）
            if self.output_critic_value:
                if not getattr(hf_config, "sac_enable", False):
                    logger.info("Setting sac_enable=True for rollout to use critic...")
                    setattr(hf_config, "sac_enable", True)
            
            # 3. 用正确的 config 加载模型
            self.module = AutoModelForVision2Seq.from_pretrained(
                model_config.path,
                config=hf_config,
                trust_remote_code=True,
            )
            self.module = self.module.to(get_device_name())
            self.module.eval()

            # 4. 初始化 SAC 组件
            if hasattr(self.module, "sac_init"):
                logger.info("Initializing SAC components for rollout model...")
                self.module.sac_init()
            
            # 5. 确保 critic_api 被初始化（如果还没有）
            if self.output_critic_value and not hasattr(self.module, "critic_api"):
                logger.info("Initializing critic_api for rollout...")
                from verl_vla.models.pi0_torch.modeling_pi0_torch import CRITIC_BACKENDS
                critic_type = getattr(self.module.config, "critic_type", "cross_attn")
                if critic_type in CRITIC_BACKENDS:
                    self.module.critic_api = CRITIC_BACKENDS[critic_type]
                    self.module.critic_api.init(self.module)
                else:
                    raise ValueError(f"Unsupported critic_type: {critic_type}")

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
        """Update rollout model weights from checkpoint engine.
        
        Args:
            weights: Async generator yielding (name, param) tuples
            **kwargs: Additional arguments (e.g., global_steps)
        """
        if self.module is None or self.engine is not None:
            logger.info("Skipping weight update: using shared actor module")
            return None
        
        try:
            prefix = "_fsdp_wrapped_module."
            target_state_dict = self.module.state_dict()
            loaded_tensors_count = 0
            skipped_count = 0
            
            async for name, param in weights:
                cleaned_name = name.replace(prefix, "")
                
                if cleaned_name in target_state_dict:
                    target_tensor = target_state_dict[cleaned_name]
                    try:
                        target_tensor.copy_(param, non_blocking=True)
                        loaded_tensors_count += 1
                    except Exception as e:
                        logger.warning(f"Warning: Failed to copy tensor '{cleaned_name}'. Error: {e}")
                else:
                    skipped_count += 1
                    logger.debug(f"Skipping tensor '{cleaned_name}': not found in rollout model")
            
            logger.info(f"Rollout model weights updated. Loaded {loaded_tensors_count} tensors including actor and critic. "
                       f"Skipped {skipped_count} unmatched tensors.")
        except Exception as e:
            logger.error(f"Error during weight update: {e}")
            raise
        
        return None

    async def release(self):
        return None

    async def resume(self, tags=None):
        return None
