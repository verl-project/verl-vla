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
            from verl.utils.transformers_compat import get_auto_model_for_vision2seq

            AutoModelForVision2Seq = get_auto_model_for_vision2seq()
            self.module = AutoModelForVision2Seq.from_pretrained(
                model_config.path,
                trust_remote_code=True,
            )
            self.module = self.module.to(get_device_name())
            self.module.eval()

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
        # Only update weights if we have an independent module (not shared from actor)
        if self.module is None or self.engine is not None:
            logger.info("Skipping weight update: using shared actor module")
            return None
        
        try:
            prefix = "_fsdp_wrapped_module."
            target_state_dict = self.module.state_dict()
            loaded_tensors_count = 0
            skipped_critic_count = 0
            
            # Process weights from async generator
            async for name, param in weights:
                cleaned_name = name.replace(prefix, "")
                
                # Skip critic-related weights (they are not part of the rollout model
                if 'critic' in cleaned_name.lower() or 'target_network' in cleaned_name.lower():
                    skipped_critic_count += 1
                    continue
                
                if cleaned_name in target_state_dict:
                    target_tensor = target_state_dict[cleaned_name]
                    try:
                        target_tensor.copy_(param, non_blocking=True)
                        loaded_tensors_count += 1
                    except Exception as e:
                        logger.warning(f"Warning: Failed to copy tensor '{cleaned_name}'. Error: {e}")
                else:
                    logger.warning(f"Warning: Failed to copy tensor '{cleaned_name}'. Model has no such key.")
            
            logger.info(f"Rollout model weights updated. Loaded {loaded_tensors_count} tensors one by one. "
                       f"Skipped {skipped_critic_count} critic-related tensors.")
        except Exception as e:
            logger.error(f"Error during weight update: {e}")
            raise
        
        return None

    async def release(self):
        return None

    async def resume(self, tags=None):
        return None
