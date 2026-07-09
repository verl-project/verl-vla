#!/usr/bin/env bash
# Deterministic GR00T N1.6 validation on all LIBERO Spatial tasks.
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data}
MODEL_PATH=${MODEL_PATH:-${DATA_ROOT}/models/gr00t_n1d6_libero_spatial_sft}
NORM_STATS_PATH=${NORM_STATS_PATH:-null}
LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${DATA_ROOT}/libero_config}
LIBERO_ASSETS_PATH=${LIBERO_ASSETS_PATH:-${HOME}/.cache/libero/assets}
MODEL_GPUS=${MODEL_GPUS:-2}
ENV_WORKERS=${ENV_WORKERS:-4}
NUM_ENVS=${NUM_ENVS:-2}
TASK_SUITE=${TASK_SUITE:-libero_spatial}
TASK_IDS=${TASK_IDS:-null}
TRIALS_PER_TASK=${TRIALS_PER_TASK:-10}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-720}
ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-8}
MAX_INTERACTIONS=${MAX_INTERACTIONS:-$(( (MAX_EPISODE_STEPS + ACTION_CHUNK_SIZE - 1) / ACTION_CHUNK_SIZE ))}
WRAP_CLASSES=${WRAP_CLASSES:-"[Qwen3DecoderLayer,Siglip2EncoderLayer,BasicTransformerBlock]"}
PROJECT_NAME=${PROJECT_NAME:-gr00t-n1d6-libero-eval}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gr00t_n1d6_libero_eval}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export LIBERO_CONFIG_PATH

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  LIBERO_PACKAGE_ROOT=$(python -c \
    'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("libero").origin).parent)')
  mkdir -p "$LIBERO_CONFIG_PATH"
  printf '%s\n' \
    "benchmark_root: ${LIBERO_PACKAGE_ROOT}" \
    "bddl_files: ${LIBERO_PACKAGE_ROOT}/bddl_files" \
    "init_states: ${LIBERO_PACKAGE_ROOT}/init_files" \
    "datasets: ${DATA_ROOT}/datasets" \
    "assets: ${LIBERO_ASSETS_PATH}" \
    > "${LIBERO_CONFIG_PATH}/config.yaml"
fi

python scripts/check_gr00t_n1d6_install.py

python -m verl_vla.trainer.main_sac \
  --config-path "$REPO_ROOT/src/verl_vla/trainer/config" \
  --config-name main_gr00t_eval \
  cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
  cluster.actor_rollout_ref.model.tokenizer_path=null \
  +cluster.actor_rollout_ref.model.load_tokenizer=False \
  cluster.actor_rollout_ref.model.enable_gradient_checkpointing=False \
  cluster.actor_rollout_ref.model.use_remove_padding=False \
  cluster.actor_rollout_ref.model.trust_remote_code=True \
  cluster.actor_rollout_ref.model.override_config.num_attention_heads=32 \
  cluster.actor_rollout_ref.model.override_config.num_key_value_heads=32 \
  cluster.actor_rollout_ref.model.override_config.verl_processor_path="$MODEL_PATH" \
  cluster.actor_rollout_ref.model.override_config.verl_norm_stats_path="$NORM_STATS_PATH" \
  cluster.actor_rollout_ref.model.override_config.verl_action_chunk_size="$ACTION_CHUNK_SIZE" \
  cluster.actor_rollout_ref.actor.strategy=fsdp2 \
  cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  cluster.actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  cluster.actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
  cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="$WRAP_CLASSES" \
  cluster.actor_rollout_ref.rollout.name=hf \
  cluster.actor_rollout_ref.rollout.mode=async_envloop \
  cluster.actor_rollout_ref.rollout.output_critic_value=False \
  cluster.resource.model.gpus_per_node="$MODEL_GPUS" \
  cluster.resource.model.device=cuda \
  cluster.resource.env.device=cpu \
  cluster.resource.env.workers_per_node="$ENV_WORKERS" \
  cluster.env.env_worker.num_envs="$NUM_ENVS" \
  cluster.env.env_worker.modes="[eval]" \
  cluster.env.env_worker.simulator.libero.simulator_type=libero \
  cluster.env.env_worker.simulator.libero.task_suite_name="$TASK_SUITE" \
  cluster.env.env_worker.simulator.libero.task_ids="$TASK_IDS" \
  cluster.env.env_worker.simulator.libero.num_trials_per_task="$TRIALS_PER_TASK" \
  cluster.env.env_worker.simulator.libero.max_episode_steps="$MAX_EPISODE_STEPS" \
  cluster.env.env_loop.max_interactions="$MAX_INTERACTIONS" \
  trainer.logger="['console']" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.val_only=True \
  trainer.val_before_train=True
