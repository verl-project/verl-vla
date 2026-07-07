#!/usr/bin/env bash
# Evaluate a GR00T checkpoint on LIBERO (success rate) via main_sac val-only mode.
# Corrected for the current main_sac config tree (cluster.* nesting; the pi
# libero_sac reference script's flat data.*/trainer.n_gpus_* keys are stale).
# The env self-generates episodes from the LIBERO benchmark (no parquet needed).
#
# Reports val/trajectory_success_rate to the console.
set -x
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-"/data"}
MODEL_PATH=${MODEL_PATH:-"$DATA_ROOT/models/gr00t_n1d7_libero_ft"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"$MODEL_PATH"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DATA_ROOT/output/gr00t_libero_eval"}

MODEL_GPUS=${MODEL_GPUS:-2}          # GPUs for actor/rollout (policy)
ENV_WORKERS=${ENV_WORKERS:-4}        # CPU env workers
NUM_ENVS=${NUM_ENVS:-2}              # envs per worker
TASK_SUITE=${TASK_SUITE:-libero_spatial}
TASK_IDS=${TASK_IDS:-"[0]"}          # limit tasks for a fast success check
TRIALS_PER_TASK=${TRIALS_PER_TASK:-10}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-256}
WRAP_CLASSES=${WRAP_CLASSES:-"[Qwen3VLTextDecoderLayer,Qwen3VLVisionBlock,BasicTransformerBlock]"}

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa VERL_LOGGING_LEVEL=INFO

python -m verl_vla.trainer.main_sac \
    --config-path "$REPO_ROOT/src/verl_vla/trainer/config" --config-name main_sac \
    cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
    cluster.actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
    cluster.actor_rollout_ref.model.enable_gradient_checkpointing=False \
    cluster.actor_rollout_ref.model.override_config.policy_type=libero \
    cluster.actor_rollout_ref.model.override_config.action_chunk_size=10 \
    cluster.actor_rollout_ref.model.override_config.sac_enable=False \
    cluster.actor_rollout_ref.model.override_config.flow_sde_enable=False \
    cluster.actor_rollout_ref.actor.strategy=fsdp2 \
    cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="$WRAP_CLASSES" \
    cluster.actor_rollout_ref.rollout.name=hf \
    cluster.actor_rollout_ref.rollout.mode=async_envloop \
    cluster.actor_rollout_ref.rollout.output_critic_value=False \
    cluster.resource.model.gpus_per_node=$MODEL_GPUS \
    cluster.resource.model.device=cuda \
    cluster.resource.env.device=cpu \
    cluster.resource.env.workers_per_node=$ENV_WORKERS \
    cluster.env.env_worker.num_envs=$NUM_ENVS \
    cluster.env.env_worker.modes=[eval] \
    cluster.env.env_worker.simulator.libero.simulator_type=libero \
    cluster.env.env_worker.simulator.libero.task_suite_name=$TASK_SUITE \
    cluster.env.env_worker.simulator.libero.task_ids="$TASK_IDS" \
    cluster.env.env_worker.simulator.libero.num_trials_per_task=$TRIALS_PER_TASK \
    cluster.env.env_worker.simulator.libero.max_episode_steps=$MAX_EPISODE_STEPS \
    trainer.logger="['console']" \
    trainer.project_name=gr00t_libero_eval \
    trainer.experiment_name=gr00t_libero_eval \
    cluster.env.env_loop.max_interactions=30 \
    trainer.val_only=True \
    trainer.val_before_train=True
