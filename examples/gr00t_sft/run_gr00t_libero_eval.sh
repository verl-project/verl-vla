#!/usr/bin/env bash
# Deterministic rollout-only GR00T N1.6 evaluation on all LIBERO Spatial tasks.
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LIBERO_PACKAGE_ROOT=$(python -c \
  'import importlib.util; from pathlib import Path; root = Path(importlib.util.find_spec("libero").origin).parent; nested = root / "libero"; print(nested if (nested / "bddl_files").is_dir() else root)')
DATA_ROOT=${DATA_ROOT:-/data}
MODEL_PATH=${MODEL_PATH:-${DATA_ROOT}/models/gr00t_n1d6_libero_spatial_sft}
NORM_STATS_PATH=${NORM_STATS_PATH:-null}
LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${DATA_ROOT}/libero_config}
LIBERO_ASSETS_PATH=${LIBERO_ASSETS_PATH:-${LIBERO_PACKAGE_ROOT}/assets}
MODEL_GPUS=${MODEL_GPUS:-2}
ENV_WORKERS=${ENV_WORKERS:-4}
NUM_ENVS=${NUM_ENVS:-2}
TASK_SUITE=${TASK_SUITE:-libero_spatial}
TASK_IDS=${TASK_IDS:-null}
TRIALS_PER_TASK=${TRIALS_PER_TASK:-10}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-720}
ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-8}
MAX_INTERACTIONS=${MAX_INTERACTIONS:-$(( (MAX_EPISODE_STEPS + ACTION_CHUNK_SIZE - 1) / ACTION_CHUNK_SIZE ))}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gr00t_n1d6_libero_eval}
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_ROOT}/output/${EXPERIMENT_NAME}}
HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-${OUTPUT_DIR}/hydra}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export LIBERO_CONFIG_PATH

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  mkdir -p "$LIBERO_CONFIG_PATH"
  printf '%s\n' \
    "benchmark_root: ${LIBERO_PACKAGE_ROOT}" \
    "bddl_files: ${LIBERO_PACKAGE_ROOT}/bddl_files" \
    "init_states: ${LIBERO_PACKAGE_ROOT}/init_files" \
    "datasets: ${DATA_ROOT}/datasets" \
    "assets: ${LIBERO_ASSETS_PATH}" \
    > "${LIBERO_CONFIG_PATH}/config.yaml"
fi

python scripts/install_checks/check_gr00t_n1d6.py

python -m verl_vla.entrypoints.eval \
  "model/adapter@cluster.actor_rollout_ref.model.adapter=gr00t" \
  model/override@cluster.actor_rollout_ref.model.override_config=gr00t \
  hydra.run.dir="$HYDRA_RUN_DIR" \
  ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL="$MUJOCO_GL" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYOPENGL_PLATFORM="$PYOPENGL_PLATFORM" \
  cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
  cluster.actor_rollout_ref.model.tokenizer_path=null \
  cluster.actor_rollout_ref.model.load_tokenizer=False \
  cluster.actor_rollout_ref.model.enable_gradient_checkpointing=False \
  cluster.actor_rollout_ref.model.use_remove_padding=False \
  cluster.actor_rollout_ref.model.trust_remote_code=True \
  cluster.actor_rollout_ref.model.adapter.policy_type=libero \
  cluster.actor_rollout_ref.model.adapter.embodiment_tag=libero_panda \
  cluster.actor_rollout_ref.model.adapter.action_dim=7 \
  cluster.actor_rollout_ref.model.adapter.embodiment_id=2 \
  cluster.actor_rollout_ref.model.adapter.critic.enabled=False \
  cluster.actor_rollout_ref.model.adapter.override_modality_configs=True \
  cluster.actor_rollout_ref.model.adapter.use_relative_action=True \
  cluster.actor_rollout_ref.model.adapter.freeze_vision_tower=False \
  cluster.actor_rollout_ref.model.adapter.norm_stats_path="$NORM_STATS_PATH" \
  cluster.actor_rollout_ref.model.adapter.num_action_chunks="$ACTION_CHUNK_SIZE" \
  cluster.actor_rollout_ref.rollout.name=hf \
  cluster.actor_rollout_ref.rollout.mode=async_envloop \
  cluster.resource.model.gpus_per_node="$MODEL_GPUS" \
  cluster.resource.model.device=cuda \
  cluster.resource.env.device=cpu \
  cluster.resource.env.workers_per_node="$ENV_WORKERS" \
  cluster.env.env_worker.num_envs="$NUM_ENVS" \
  cluster.env.env_worker.auto_reset=False \
  cluster.env.env_worker.modes="[eval]" \
  cluster.env.env_worker.simulator.simulator_type=libero \
  cluster.env.env_worker.simulator.libero.simulator_type=libero \
  cluster.env.env_worker.simulator.libero.task_suite_name="$TASK_SUITE" \
  cluster.env.env_worker.simulator.libero.task_ids="$TASK_IDS" \
  cluster.env.env_worker.simulator.libero.num_trials_per_task="$TRIALS_PER_TASK" \
  cluster.env.env_worker.simulator.libero.max_episode_steps="$MAX_EPISODE_STEPS" \
  cluster.env.env_loop.max_interactions="$MAX_INTERACTIONS" \
  cluster.env.env_worker.recorder.video.root="$OUTPUT_DIR/videos" \
  max_episodes=null \
  output_dir="$OUTPUT_DIR"
