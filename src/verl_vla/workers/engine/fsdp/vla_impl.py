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
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import fsdp_version, set_reshard_after_forward
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

from verl_vla.models import register_vla_models

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="vla_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithActionHEAD(FSDPEngine):
    """VLA-specific FSDP engine skeleton."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollout_eval_ctx = None
        self._rollout_rng_state = None
        self._train_rng_state = None
        self._fsdp_unshard_exit_stack = None

    def _build_module(self):
        register_vla_models()
        logger.info("Registered VLA models before building VLA FSDP engine")
        return super()._build_module()

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
