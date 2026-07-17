#!/usr/bin/env bash
# Download and validate the pinned community GR00T N1.6 LIBERO checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT=${DATA_ROOT:-/data}
MODEL_REPO=${MODEL_REPO:-0xAnkitSingh/GR00T-N1.6-LIBERO}
MODEL_REVISION=${MODEL_REVISION:-d690a226ad06e81736786f56cf879d2ed1dd3f0f}
MODEL_PATH=${MODEL_PATH:-${DATA_ROOT}/models/GR00T-N1.6-LIBERO}

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "The 'hf' CLI is required to download ${MODEL_REPO}." >&2
    echo "Install it with: python -m pip install 'huggingface-hub[cli]'" >&2
    exit 2
  fi
  mkdir -p "$MODEL_PATH"
  hf download "$MODEL_REPO" \
    --repo-type model \
    --revision "$MODEL_REVISION" \
    --local-dir "$MODEL_PATH"
fi

export MODEL_PATH
export NORM_STATS_PATH=${NORM_STATS_PATH:-null}
export MODEL_GPUS=${MODEL_GPUS:-1}
export ENV_WORKERS=${ENV_WORKERS:-1}
export NUM_ENVS=${NUM_ENVS:-3}
export TASK_SUITE=${TASK_SUITE:-libero_spatial}
export TASK_IDS=${TASK_IDS:-'[0]'}
export TRIALS_PER_TASK=${TRIALS_PER_TASK:-3}
export MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-720}
export ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-8}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-gr00t_n1d6_community_libero_spatial}

cd "$REPO_ROOT"
exec bash examples/gr00t_sft/run_gr00t_libero_eval.sh
