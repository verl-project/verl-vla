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

from check_commit_message import MAX_TITLE_LENGTH, validate_commit_title

_SERIES_PATTERN = re.compile(r"^\[(?P<position>[1-9]\d*)/(?P<total>[1-9]\d*|[Nn])\](?P<title>.+)$")
_SERIES_LIKE_PATTERN = re.compile(r"^\[[^\]]*/[^\]]*\]")
_COMMIT_ONLY_PREFIXES = ("fixup! ", "squash! ", "amend! ", "Merge ", 'Revert "')


def validate_pr_title(title: str) -> list[str]:
    """Return validation errors for a pull request title."""
    if not title:
        return ["PR title must not be empty"]

    core_title = title
    series_match = _SERIES_PATTERN.fullmatch(title)
    if series_match is not None:
        core_title = series_match.group("title")
        total = series_match.group("total")
        if total.isdigit() and int(series_match.group("position")) > int(total):
            return ["PR series position must not exceed its total"]
    elif _SERIES_LIKE_PATTERN.match(title):
        return ["invalid PR series prefix; expected '[1/N]' or '[1/3]' followed immediately by the title"]

    if core_title.startswith(_COMMIT_ONLY_PREFIXES):
        return ["autosquash, merge, and revert prefixes are not valid in PR titles"]

    errors = validate_commit_title(core_title)
    if len(title) > MAX_TITLE_LENGTH and len(core_title) <= MAX_TITLE_LENGTH:
        errors.insert(0, f"PR title must be at most {MAX_TITLE_LENGTH} characters (got {len(title)})")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a verl-vla pull request title.")
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    errors = validate_pr_title(args.title)
    if not errors:
        return 0

    print(f"Invalid PR title: {args.title!r}")
    for error in errors:
        print(f"  - {error}")
    print("See CONTRIBUTING.md for the pull request title convention.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
