#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

PYTHON=${PYTHON:-python}
TASK_ID=${TASK_ID:-0}

task_ids_override="cluster.env.env_worker.simulator.libero.task_ids=null"
if [[ -n "$TASK_ID" ]]; then
  task_ids_override="cluster.env.env_worker.simulator.libero.task_ids=[$TASK_ID]"
fi

overrides=(
  "num_episodes=10"
  "resume=False"

  "cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial"
  "$task_ids_override"
  "cluster.env.env_worker.simulator.libero.num_trials_per_task=null"
  "cluster.env.env_worker.simulator.libero.specific_reset_id=null"
  "cluster.env.env_worker.simulator.libero.max_episode_steps=512"

  "cluster.env.env_worker.teleop.devices=[gamepad]"

  "cluster.env.env_worker.recorder.lerobot.root=$PROJECT_ROOT/outputs/record_gamepad/lerobot"
  "cluster.env.env_worker.recorder.lerobot.repo_id=local/verl_vla_libero_gamepad_record"
  "cluster.env.env_worker.recorder.video.root=$PROJECT_ROOT/outputs/record_gamepad/videos"

  "ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa"
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO"
)

"$PYTHON" -m verl_vla.entrypoints.record "${overrides[@]}" "$@"
