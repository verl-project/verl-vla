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

from __future__ import annotations

import argparse
import re
from pathlib import Path

ALLOWED_TYPES = frozenset({"feat", "fix", "refactor", "chore", "test"})
ALLOWED_MODULES = frozenset(
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
MAX_TITLE_LENGTH = 100

_TITLE_PATTERN = re.compile(
    r"^(?P<breaking>\[BREAKING\])?\[(?P<modules>[^\]]+)\] "
    r"(?P<type>[^:\s]+): (?P<description>.+)$"
)
_MODULE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_AUTOSQUASH_PREFIXES = ("fixup! ", "squash! ", "amend! ")


def _strip_autosquash_prefix(title: str) -> str:
    for prefix in _AUTOSQUASH_PREFIXES:
        if title.startswith(prefix):
            return title.removeprefix(prefix)
    return title


def validate_commit_title(title: str) -> list[str]:
    """Return validation errors for a commit title."""
    if not title:
        return ["commit title must not be empty"]

    if title.startswith("Merge ") or title.startswith('Revert "'):
        return []

    core_title = _strip_autosquash_prefix(title)
    errors: list[str] = []

    if len(core_title) > MAX_TITLE_LENGTH:
        errors.append(f"title must be at most {MAX_TITLE_LENGTH} characters (got {len(core_title)})")

    match = _TITLE_PATTERN.fullmatch(core_title)
    if match is None:
        errors.append("expected '[module] type: description' or '[BREAKING][module] type: description'")
        return errors

    raw_modules = match.group("modules")
    modules = raw_modules.split(", ")
    if ", ".join(modules) != raw_modules or any(not _MODULE_PATTERN.fullmatch(module) for module in modules):
        errors.append("modules must be lowercase and separated by ', '")
    else:
        unknown_modules = sorted(set(modules) - ALLOWED_MODULES)
        if unknown_modules:
            errors.append(f"unknown module(s): {', '.join(unknown_modules)}")
        if len(modules) != len(set(modules)):
            errors.append("modules must not be repeated")

    commit_type = match.group("type")
    if commit_type not in ALLOWED_TYPES:
        errors.append(f"unknown type '{commit_type}'; allowed types: {', '.join(sorted(ALLOWED_TYPES))}")

    description = match.group("description")
    if description != description.strip():
        errors.append("description must not have leading or trailing whitespace")
    if not re.match(r"[a-z]", description):
        errors.append("description must start with a lowercase letter")
    if description.endswith("."):
        errors.append("description must not end with a period")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a verl-vla commit message.")
    title_source = parser.add_mutually_exclusive_group(required=True)
    title_source.add_argument("commit_message_file", nargs="?", type=Path)
    title_source.add_argument("--title", help="Validate a title directly instead of reading a commit message file.")
    args = parser.parse_args()

    if args.title is not None:
        title = args.title
    else:
        lines = args.commit_message_file.read_text(encoding="utf-8").splitlines()
        title = lines[0] if lines else ""
    errors = validate_commit_title(title)
    if not errors:
        return 0

    print(f"Invalid commit title: {title!r}")
    for error in errors:
        print(f"  - {error}")
    print("See CONTRIBUTING.md for the commit message convention.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
