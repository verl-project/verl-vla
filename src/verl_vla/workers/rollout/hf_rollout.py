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

import numpy as np
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
        actor_config=None,
        data_keys=None,
        tokenizer: Any = None,
        **kwargs,
    ):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        self.engine = engine
        self.module = module if module is not None else (engine.module if engine is not None else None)
        self.actor_config = actor_config
        self.data_keys = data_keys
        self.tokenizer = tokenizer if tokenizer is not None else getattr(model_config, "tokenizer", None)
        self.output_critic_value = bool(config.output_critic_value)

        if self.module is None:
            logger.info("No shared actor engine provided, loading model from path...")

            from transformers import AutoConfig
            from verl.utils.model import update_model_config
            from verl.utils.transformers_compat import get_auto_model_for_vision2seq

            # 1. Get the correct AutoClass
            AutoModelForVision2Seq = get_auto_model_for_vision2seq()

            # 2. Load and apply override_config to hf_config
            hf_config = AutoConfig.from_pretrained(
                model_config.path,
                trust_remote_code=True,
            )

            # Apply override_config, ensure sac_enable=True
            if hasattr(model_config, "override_config") and model_config.override_config:
                override_config = (
                    model_config.override_config["model_config"]
                    if "model_config" in model_config.override_config
                    else model_config.override_config
                )
                update_model_config(hf_config, override_config)

            # Ensure sac_enable is True (if critic is needed)
            if self.output_critic_value:
                if not getattr(hf_config, "sac_enable", False):
                    logger.info("Setting sac_enable=True for rollout to use critic...")
                    hf_config.sac_enable = True

            # 3. Load model with correct config
            self.module = AutoModelForVision2Seq.from_pretrained(
                model_config.path,
                config=hf_config,
                trust_remote_code=True,
            )
            self.module = self.module.to(get_device_name())
            self.module.eval()

            # 4. Initialize SAC components
            if hasattr(self.module, "sac_init"):
                logger.info("Initializing SAC components for rollout model...")
                self.module.sac_init()

        from torch.distributed.fsdp import register_fsdp_forward_method

        register_fsdp_forward_method(self.module, "sac_sample_actions")
        if self.output_critic_value:
            register_fsdp_forward_method(self.module, "sac_get_critic_value")

    def _apply_acp_prompt_tag(self, prompts: DataProto) -> DataProto:
        acp_config = self.config.acp
        if not acp_config.enable:
            return prompts

        data_keys = self.data_keys or getattr(self.actor_config, "data_keys", None)
        if data_keys is None:
            raise ValueError("ACP rollout requires actor_rollout_ref.data_keys for data key lookup.")
        if data_keys.task not in prompts.non_tensor_batch:
            return prompts

        values = prompts.non_tensor_batch[data_keys.task]
        tagged_values = values.copy()
        indicator_key = data_keys.indicator
        indicators = (
            prompts.batch.get(indicator_key) if indicator_key is not None and prompts.batch is not None else None
        )
        indicators = indicators.reshape(-1).detach().cpu().numpy() if indicators is not None else None
        for idx, value in enumerate(values):
            if indicators is not None and int(indicators[idx]) <= 0:
                continue
            tagged_values[idx] = f"{value}\n{acp_config.positive_tag}"
        prompts.non_tensor_batch[data_keys.task] = np.asarray(tagged_values, dtype=object)
        return prompts

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        prompts = self._apply_acp_prompt_tag(prompts)
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            eval = bool(prompts.meta_info.get("eval", False))
            output = self.module.sac_sample_actions(
                prompts,
                tokenizer=self.tokenizer,
                eval=eval,
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

            logger.info(
                f"Rollout model weights updated. Loaded {loaded_tensors_count} tensors including actor and critic. "
                f"Skipped {skipped_count} unmatched tensors."
            )
        except Exception as e:
            logger.error(f"Error during weight update: {e}")
            raise

        return None

    async def release(self):
        return None

    async def resume(self, tags=None):
        return None
