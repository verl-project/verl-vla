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
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "check_commit_message.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_commit_message", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()
validate_commit_title = checker.validate_commit_title

EXPECTED_MODULES = frozenset(
    {
        "cfg",
        "ci",
        "data",
        "doc",
        "docker",
        "entrypoints",
        "env",
        "misc",
        "model",
        "perf",
        "recorder",
        "rollout",
        "teleop",
        "trainer",
        "worker",
    }
)


def test_allowed_module_set() -> None:
    assert checker.ALLOWED_MODULES == EXPECTED_MODULES


@pytest.mark.parametrize("module", sorted(EXPECTED_MODULES))
def test_each_allowed_module_is_accepted(module: str) -> None:
    assert validate_commit_title(f"[{module}] chore: validate commit title") == []


@pytest.mark.parametrize(
    "title",
    [
        "[teleop] feat: add gamepad calibration",
        "[rollout] chore: clarify rollout progress counters",
        "[env, recorder] fix: preserve terminal signals across action chunks",
        "[BREAKING][cfg] refactor: rename rollout configuration fields",
        "fixup! [env] fix: preserve terminal signals",
        "squash! [trainer, cfg] refactor: unify rollout configuration",
        "amend! [doc] chore: update contribution instructions",
        'Revert "[env] fix: preserve terminal signals"',
        "Merge branch 'main' into feature/teleop",
    ],
)
def test_valid_commit_titles(title: str) -> None:
    assert validate_commit_title(title) == []


@pytest.mark.parametrize(
    ("title", "expected_error"),
    [
        ("", "must not be empty"),
        ("fix(env): preserve terminal signals", "expected '[module] type: description'"),
        ("[unknown] feat: add support", "unknown module"),
        ("[sac] feat: add replay sampling", "unknown module"),
        ("[libero] fix: shard reset states", "unknown module"),
        ("[ray] fix: clean up placement groups", "unknown module"),
        ("[env,recorder] feat: add recording", "separated by ', '"),
        ("[Env] feat: add support", "modules must be lowercase"),
        ("[env] Fix: preserve terminal signals", "unknown type 'Fix'"),
        ("[env] docs: update documentation", "unknown type 'docs'"),
        ("[env] fix: Preserve terminal signals", "start with a lowercase letter"),
        ("[env] fix: preserve terminal signals.", "must not end with a period"),
        ("[env, env] fix: preserve terminal signals", "must not be repeated"),
        (f"[env] fix: {'a' * 90}", "must be at most 100 characters"),
    ],
)
def test_invalid_commit_titles(title: str, expected_error: str) -> None:
    assert any(expected_error in error for error in validate_commit_title(title))


def test_cli_accepts_direct_title() -> None:
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--title", "[misc] chore: validate commit titles"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0


def test_cli_rejects_invalid_direct_title() -> None:
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--title", "chore: validate commit titles"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Invalid commit title" in result.stdout
