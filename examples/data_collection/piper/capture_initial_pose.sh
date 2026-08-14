#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "verl-vla-piper" ]]; then
  echo "Activate the Piper environment first: conda activate verl-vla-piper" >&2
  exit 1
fi

exec python "$SCRIPT_DIR/capture_initial_pose.py" "$@"
