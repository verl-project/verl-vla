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

from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor
from verl.utils.transformers_compat import get_auto_model_for_vision2seq

from .gr00t_torch import Gr00tForConditionalGeneration, Gr00tTorchConfig
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
    "gr00t_torch": False,
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


def register_gr00t_torch_model() -> None:
    """Register the GR00T wrapper with the HF auto classes."""
    if _REGISTERED_MODELS["gr00t_torch"]:
        return

    AutoConfig.register("gr00t_torch", Gr00tTorchConfig)
    AutoModelForVision2Seq.register(Gr00tTorchConfig, Gr00tForConditionalGeneration)

    _REGISTERED_MODELS["gr00t_torch"] = True


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
    register_gr00t_torch_model()
    register_recap_value_critic_model()
