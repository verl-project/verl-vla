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
import contextlib
import logging
import os

import torch
from packaging import version
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._unshard_param_utils import _get_module_fsdp_state, _unshard_params_for_summon
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    merged_lora_context,
    normalize_peft_param_name,
    offload_fsdp_model_to_cpu,
    set_reshard_after_forward,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="vla_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class VLAFSDPEngine(FSDPEngine):
    """VLA-specific FSDP engine skeleton."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollout_eval_ctx = None
        self._rollout_rng_state = None
        self._train_rng_state = None
        self._fsdp_unshard_exit_stack = None

    def _build_module(self):
        if getattr(self.model_config, "native_architecture", None) is None:
            raise ValueError(
                "VLA checkpoint architecture was not recognized. Add an explicit builder in "
                "verl_vla.models.builder instead of registering the model with a Transformers AutoClass."
            )

        from verl.utils.torch_dtypes import PrecisionType

        from verl_vla.models import build_vla_model

        torch_dtype = self.engine_config.model_dtype
        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        module = build_vla_model(self.model_config, torch_dtype=torch_dtype)
        module.to(torch_dtype)
        return module

    def _build_lora_module(self, module):
        """Apply PEFT to the native policy without freezing VLA auxiliary modules."""

        from verl_vla.models.base import TrainableVLAModelMixin

        if not isinstance(module, TrainableVLAModelMixin):
            raise TypeError(f"LoRA requires a TrainableVLAModelMixin policy wrapper, got {type(module).__name__}")
        if not self.model_config.lora.get("merge", False):
            raise ValueError("VLA LoRA requires model.lora.merge=True for native rollout weight synchronization")

        lora_config = self.model_config.lora
        adapter_path = lora_config["adapter_path"]
        if adapter_path is not None:
            adapter_path = copy_to_local(adapter_path, use_shm=self.model_config.use_shm)

        parameter_dtype = next(module.parameters()).dtype
        module.apply_lora(
            rank=int(lora_config["rank"]),
            alpha=int(lora_config["alpha"]),
            target_modules=convert_to_regular_types(lora_config["target_modules"]),
            target_parameters=convert_to_regular_types(lora_config["target_parameters"]),
            exclude_modules=convert_to_regular_types(lora_config["exclude_modules"]),
            adapter_path=adapter_path,
        )
        # PEFT initializes or loads adapter parameters in float32 by default.
        # FSDP2 requires the original parameters in each group to share one dtype.
        module.to(parameter_dtype)
        return module

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        """Expose merged native VLA weights to colocated or disaggregated rollout workers."""

        if not self._is_lora:
            return super().get_per_tensor_param(
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
                **kwargs,
            )

        load_fsdp_model_to_gpu(self.module)
        with merged_lora_context(self.module, backup_adapters=True):
            params = normalize_peft_param_name(self.module.state_dict())
            params = {name: param.clone() for name, param in params.items()}

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        device = get_device_id()
        per_tensor_param = (
            (
                name,
                param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                if isinstance(param, DTensor)
                else param,
            )
            for name, param in params.items()
        )
        return per_tensor_param, None

    def disable_adapter(self):
        """Temporarily disable the policy adapter while preserving the VLA wrapper."""

        model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if not model.has_lora:
            return contextlib.nullcontext()
        return model.policy.disable_adapter()

    def initialize(self):
        super().initialize()
        from .native_policy_checkpoint_manager import NativePolicyFSDPCheckpointManager

        self.checkpoint_manager = NativePolicyFSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    def switch_to_rollout(self):
        if self._rollout_eval_ctx is not None:
            return

        self._rollout_eval_ctx = self.eval_mode()
        self._rollout_eval_ctx.__enter__()
        aggressive_empty_cache(force_sync=True)

        self._rollout_rng_state = get_torch_device().get_rng_state()
        if self._train_rng_state is None:
            self._train_rng_state = self._rollout_rng_state
        get_torch_device().set_rng_state(self._train_rng_state)

        fsdp_ver = fsdp_version(self.module)
        if fsdp_ver == 1:
            exit_stack = contextlib.ExitStack()
            optional_state = _get_module_fsdp_state(self.module)
            states_and_modules = ([optional_state], [self.module])

            for state, fsdp_module in zip(*states_and_modules, strict=False):
                exit_stack.enter_context(
                    _unshard_params_for_summon(
                        module=fsdp_module,
                        state=state,
                        writeback=False,
                        rank0_only=False,
                        offload_to_cpu=False,
                        with_grads=False,
                    )
                )

            self._fsdp_unshard_exit_stack = exit_stack
        elif fsdp_ver == 2:
            self.module.unshard()
            for module in self.module.modules():
                if isinstance(module, FSDPModule) or hasattr(module, "unshard"):
                    module.unshard()
            if version.parse(torch.__version__) < version.parse("2.8"):
                set_reshard_after_forward(self.module, False)
            else:
                self.module.set_reshard_after_forward(False)
        else:
            raise NotImplementedError(f"Unsupported fsdp version {fsdp_ver}")

    def switch_to_train(self):
        self._train_rng_state = get_torch_device().get_rng_state()
        if self._rollout_rng_state is not None:
            get_torch_device().set_rng_state(self._rollout_rng_state)

        fsdp_ver = fsdp_version(self.module)
        if fsdp_ver == 1:
            if self._fsdp_unshard_exit_stack is not None:
                self._fsdp_unshard_exit_stack.close()
                self._fsdp_unshard_exit_stack = None
        elif fsdp_ver == 2:
            self.module.reshard()
            for module in self.module.modules():
                if isinstance(module, FSDPModule) or hasattr(module, "reshard"):
                    module.reshard()
            if version.parse(torch.__version__) < version.parse("2.8"):
                set_reshard_after_forward(self.module, True)
            else:
                self.module.set_reshard_after_forward(True)
        else:
            raise NotImplementedError(f"Unsupported fsdp version {fsdp_ver}")

        if self._rollout_eval_ctx is not None:
            self._rollout_eval_ctx.__exit__(None, None, None)
            self._rollout_eval_ctx = None
