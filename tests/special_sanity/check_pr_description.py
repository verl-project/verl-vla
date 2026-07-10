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

import json
import os
import re
from pathlib import Path

OVERVIEW_HEADING = "### What does this PR do?"
OVERVIEW_PLACEHOLDER = "Add a concise overview of what this PR aims to achieve."


def extract_section(body: str, heading: str) -> str | None:
    """Return the Markdown content below a third-level heading."""
    lines = body.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == heading) + 1
    except StopIteration:
        return None

    section: list[str] = []
    for line in lines[start:]:
        if line.startswith("### "):
            break
        section.append(line)
    return "\n".join(section).strip()


def validate_pr_description(body: str) -> list[str]:
    """Return validation errors for the required PR overview section."""
    overview = extract_section(body, OVERVIEW_HEADING)
    if overview is None:
        return [f"missing required section: {OVERVIEW_HEADING}"]
    if not overview:
        return ["the PR overview must not be empty"]
    if OVERVIEW_PLACEHOLDER in overview:
        return ["replace the PR overview placeholder with a concise description"]

    visible_overview = re.sub(r"<!--.*?-->", "", overview, flags=re.DOTALL).strip()
    if not visible_overview:
        return ["the PR overview must contain visible text"]
    return []


def load_pr_body(event_path: Path) -> str:
    """Load a pull request body from a GitHub event payload."""
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    return payload.get("pull_request", {}).get("body", "") or ""


def main() -> int:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("GITHUB_EVENT_PATH is not set")
        return 1

    errors = validate_pr_description(load_pr_body(Path(event_path)))
    if not errors:
        return 0

    print("Invalid PR description:")
    for error in errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
