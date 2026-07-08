#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

PYTHON=${PYTHON:-python}

overrides=(
  "num_episodes=10"

  "cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial"
  "cluster.env.env_worker.simulator.libero.task_ids=null"
  "cluster.env.env_worker.simulator.libero.num_trials_per_task=null"
  "cluster.env.env_worker.simulator.libero.specific_reset_id=null"
  "cluster.env.env_worker.simulator.libero.max_episode_steps=512"

  "cluster.env.env_worker.recorder.lerobot.root=$PROJECT_ROOT/outputs/record/lerobot"
  "cluster.env.env_worker.recorder.lerobot.repo_id=local/verl_vla_libero_record"
  "cluster.env.env_worker.recorder.video.root=$PROJECT_ROOT/outputs/record/videos"

  "ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa"
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO"
)

"$PYTHON" -m verl_vla.entrypoints.main_record "${overrides[@]}" "$@"
