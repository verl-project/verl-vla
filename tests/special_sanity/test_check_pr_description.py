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
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("check_pr_description.py")


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_pr_description", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_valid_pr_description() -> None:
    body = """### What does this PR do?

Adds repository checks for pull request titles and descriptions.

### Test

`pytest -q tests/special_sanity`
"""

    assert checker.validate_pr_description(body) == []


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [
        ("", "missing required section"),
        ("### What does this PR do?\n\n### Test", "must not be empty"),
        (
            "### What does this PR do?\n\n> Add a concise overview of what this PR aims to achieve.",
            "replace the PR overview placeholder",
        ),
        ("### What does this PR do?\n\n<!--\nexplanation\n-->", "must contain visible text"),
    ],
)
def test_invalid_pr_description(body: str, expected_error: str) -> None:
    assert any(expected_error in error for error in checker.validate_pr_description(body))


def test_load_pr_body(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"pull_request": {"body": "PR body"}}), encoding="utf-8")

    assert checker.load_pr_body(event_path) == "PR body"
