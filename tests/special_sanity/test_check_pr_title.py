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

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "check_pr_title.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_pr_title", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


validate_pr_title = _load_checker().validate_pr_title


@pytest.mark.parametrize(
    "title",
    [
        "[env] fix: preserve terminal signals",
        "[env, recorder] feat: add episode recording",
        "[BREAKING][cfg] refactor: rename rollout fields",
        "[1/N][trainer] feat: add staged training support",
        "[2/3][worker] refactor: migrate cluster orchestration",
    ],
)
def test_valid_pr_titles(title: str) -> None:
    assert validate_pr_title(title) == []


@pytest.mark.parametrize(
    ("title", "expected_error"),
    [
        ("", "must not be empty"),
        ("[0/N][trainer] feat: add support", "invalid PR series prefix"),
        ("[3/2][trainer] feat: add support", "must not exceed"),
        ("[1/N] [trainer] feat: add support", "expected '[module] type: description'"),
        ("fixup! [env] fix: preserve signals", "not valid in PR titles"),
        ("Merge branch 'main'", "not valid in PR titles"),
        ('Revert "[env] fix: preserve signals"', "not valid in PR titles"),
        ("[libero] feat: add support", "unknown module"),
        (f"[1/N][env] fix: {'a' * 85}", "at most 100 characters"),
    ],
)
def test_invalid_pr_titles(title: str, expected_error: str) -> None:
    assert any(expected_error in error for error in validate_pr_title(title))
