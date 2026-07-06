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

from .base import Gr00tInput, Gr00tOutput
from .libero_policy import LiberoGr00tInput, LiberoGr00tOutput

_GR00T_POLICY_REGISTRY = {
    "libero": (LiberoGr00tInput, LiberoGr00tOutput),
}


def get_gr00t_policy_classes(policy_type: str) -> tuple[type[Gr00tInput], type[Gr00tOutput]]:
    try:
        return _GR00T_POLICY_REGISTRY[policy_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_GR00T_POLICY_REGISTRY))
        raise ValueError(f"Unknown gr00t policy_type: {policy_type}. Supported values: {supported}") from exc
