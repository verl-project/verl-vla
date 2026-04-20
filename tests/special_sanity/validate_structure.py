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
import sys
from pathlib import Path


def discover_allowed_modules(impl_root: Path, extra: list[str]) -> set[str]:
    """Return the set of first-level directories that tests may live under."""
    allowed = {path.name for path in impl_root.iterdir() if path.is_dir()}
    allowed.update(extra)
    return allowed


def find_violations(tests_root: Path, allowed: set[str], allowed_files: list[str]) -> list[str]:
    """Return a list of test files that do not mirror the implementation structure."""
    errors: list[str] = []
    for test_file in tests_root.rglob("test*.py"):
        if str(test_file) in allowed_files:
            continue
        rel_parts = test_file.relative_to(tests_root).parts
        if len(rel_parts) < 2:
            errors.append(f"{test_file}: must be inside one of {sorted(allowed)} (not at tests root)")
            continue

        first_folder = rel_parts[0]
        if first_folder not in allowed:
            errors.append(
                f"{test_file}: subfolder '{first_folder}' under tests/ is not an allowed module. "
                f"The valid ones are: {sorted(allowed)}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that test files follow tests/<module>/... layout.")
    parser.add_argument(
        "--impl-root",
        type=Path,
        default="src/verl_vla",
        help="Implementation root",
    )
    parser.add_argument(
        "--tests-root",
        type=Path,
        default="tests",
        help="Root of test tree",
    )
    parser.add_argument(
        "--allow-dirs",
        nargs="*",
        default=["special_sanity"],
        help="Extra top-level test folders that are exempt from the rule",
    )
    parser.add_argument(
        "--allow-files",
        nargs="*",
        default=[],
        help="Extra top-level test files that are exempt from the rule",
    )
    args = parser.parse_args()

    if not args.impl_root.is_dir():
        raise Exception(f"Implementation root '{args.impl_root}' does not exist.")
    if not args.tests_root.is_dir():
        raise Exception(f"Tests root '{args.tests_root}' does not exist.")

    allowed = discover_allowed_modules(args.impl_root, args.allow_dirs)
    violations = find_violations(args.tests_root, allowed, args.allow_files)

    if violations:
        print("Test layout violations found:\n", file=sys.stderr)
        for err in violations:
            print(f"  - {err}", file=sys.stderr)

        print(
            "\nGuideline:\n"
            "  Place each test file under tests/<module_name>/...\n"
            "  where <module_name> is one of the top-level packages inside the implementation root,\n"
            "  or is explicitly listed via --allow-dirs.\n",
            file=sys.stderr,
        )
        raise Exception("Test layout violations found.")

    print("Tests folder structure looks good.")


if __name__ == "__main__":
    main()
