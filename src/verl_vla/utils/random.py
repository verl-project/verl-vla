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

"""Randomness utilities."""

from __future__ import annotations


def compose_seed(*fields: int, modulo: int = 2**31 - 1) -> int:
    """Compose a deterministic seed from an ordered list of integer fields."""
    mixed_seed = 0
    for field in fields:
        mixed_seed = mixed_seed * 1000003 + int(field)
    return int(mixed_seed % modulo)
