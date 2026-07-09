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


"""Utility helpers to register custom VLA models with Hugging Face Auto classes."""

import importlib.util
import logging

from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor
from verl.utils.transformers_compat import get_auto_model_for_vision2seq

from .gr00t_n1d6 import GR00T_N1D6_COMMIT
from .openvla_oft.configuration_prismatic import OpenVLAConfig
from .openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
from .openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from .pi0_torch import PI0ForConditionalGeneration, PI0TorchConfig
from .recap_value_critic import (
    ReCapValueCriticConfig,
    ReCapValueCriticForConditionalGeneration,
)

_REGISTERED_MODELS = {
    "openvla_oft": False,
    "pi0_torch": False,
    "gr00t_n1d6": False,
    "recap_value_critic": False,
}
AutoModelForVision2Seq = get_auto_model_for_vision2seq()


def register_openvla_oft() -> None:
    """Register the OpenVLA OFT model and processors."""
    if _REGISTERED_MODELS["openvla_oft"]:
        return

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    _REGISTERED_MODELS["openvla_oft"] = True


def register_pi0_torch_model() -> None:
    """Register the PI0 wrapper with the HF auto classes."""
    if _REGISTERED_MODELS["pi0_torch"]:
        return

    AutoConfig.register("pi0_torch", PI0TorchConfig)
    AutoModelForVision2Seq.register(PI0TorchConfig, PI0ForConditionalGeneration)

    _REGISTERED_MODELS["pi0_torch"] = True


def register_gr00t_n1d6_model(*, required: bool = False) -> bool:
    """Register the optional external GR00T N1.6 package with ``AutoModel``.

    GR00T is intentionally not a verl-vla dependency.  Importing verl-vla must
    therefore continue to work when the user has not opted into GR00T.
    """
    if _REGISTERED_MODELS["gr00t_n1d6"]:
        return True
    if importlib.util.find_spec("gr00t") is None:
        if required:
            raise ModuleNotFoundError(
                "GR00T N1.6 is not installed. Install the pinned source package with "
                "`python -m pip install --no-deps \"gr00t @ "
                f"git+https://github.com/NVIDIA/Isaac-GR00T.git@{GR00T_N1D6_COMMIT}\"`."
            )
        return False

    try:
        from .gr00t_n1d6.modeling_gr00t_n1d6 import register_with_transformers

        register_with_transformers()
    except Exception:
        if required:
            raise
        logging.getLogger(__name__).warning(
            "The optional GR00T package is present but its N1.6 integration could not be loaded. "
            "Other verl-vla models remain available.",
            exc_info=True,
        )
        return False

    _REGISTERED_MODELS["gr00t_n1d6"] = True
    return True


def register_recap_value_critic_model() -> None:
    """Register the RECAP value critic with the HF auto classes."""
    if _REGISTERED_MODELS["recap_value_critic"]:
        return

    AutoConfig.register("recap_value_critic", ReCapValueCriticConfig)
    AutoModel.register(ReCapValueCriticConfig, ReCapValueCriticForConditionalGeneration)
    AutoModelForVision2Seq.register(ReCapValueCriticConfig, ReCapValueCriticForConditionalGeneration)

    _REGISTERED_MODELS["recap_value_critic"] = True


def register_vla_models() -> None:
    """Register all custom VLA models with Hugging Face."""
    register_openvla_oft()
    register_pi0_torch_model()
    register_gr00t_n1d6_model()
    register_recap_value_critic_model()
