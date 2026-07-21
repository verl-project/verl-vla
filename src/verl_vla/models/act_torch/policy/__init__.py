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

from .base import ActInput, ActOutput
from .libero_policy import LiberoActInput, LiberoActOutput

__all__ = [
    "ActInput",
    "ActOutput",
    "LiberoActInput",
    "LiberoActOutput",
    "get_act_policy_classes",
]


_ACT_POLICY_REGISTRY = {
    "libero": (LiberoActInput, LiberoActOutput),
}


def get_act_policy_classes(policy_type: str):
    try:
        return _ACT_POLICY_REGISTRY[policy_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_ACT_POLICY_REGISTRY))
        raise ValueError(f"Unknown act policy_type: {policy_type}. Supported values: {supported}") from exc
