#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

HF_TOKEN=${HF_TOKEN:-}
HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}

RECAP_INITIAL_POLICY_PATH=${RECAP_INITIAL_POLICY_PATH:-/file_system/liujincheng/models/global_step_14_merged_hf}
RECAP_OUTPUT_ROOT=${RECAP_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/recap}
RECAP_SFT_DATASET_NAME=${RECAP_SFT_DATASET_NAME:-verl_vla_libero_sft}
RECAP_SFT_DATASET_ROOT="$RECAP_OUTPUT_ROOT/lerobot/local/$RECAP_SFT_DATASET_NAME"
RECAP_SFT_DATASET_REPO_ID="local/$RECAP_SFT_DATASET_NAME"
RECAP_VALUE_MODEL_CHECKPOINT_DIR="$RECAP_OUTPUT_ROOT/checkpoints/value_model"
RECAP_POLICY_CHECKPOINT_DIR="$RECAP_OUTPUT_ROOT/checkpoints/policy"

RECAP_ENV_LOOP_ENV_WORKERS=${RECAP_ENV_LOOP_ENV_WORKERS:-8}
RECAP_ENV_LOOP_NUM_ENVS=${RECAP_ENV_LOOP_NUM_ENVS:-2}
RECAP_ENV_LOOP_MODEL_GPUS=${RECAP_ENV_LOOP_MODEL_GPUS:-8}
RECAP_SFT_GPUS=${RECAP_SFT_GPUS:-8}

RECAP_ENV_LOOP_MAX_EPISODE_STEPS=${RECAP_ENV_LOOP_MAX_EPISODE_STEPS:-200}
RECAP_ENV_LOOP_TASK_SUITE_NAME=${RECAP_ENV_LOOP_TASK_SUITE_NAME:-libero_spatial}
RECAP_ENV_LOOP_TASK_IDS=${RECAP_ENV_LOOP_TASK_IDS:-[1]}
RECAP_ENV_LOOP_NUM_TRIALS_PER_TASK=${RECAP_ENV_LOOP_NUM_TRIALS_PER_TASK:-null}
RECAP_ENV_LOOP_SPECIFIC_RESET_ID=${RECAP_ENV_LOOP_SPECIFIC_RESET_ID:-null}


overrides=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa"
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYOPENGL_PLATFORM=osmesa"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO"
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_TOKEN=$HF_TOKEN"
  "+ray_kwargs.ray_init.runtime_env.env_vars.HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"

  "recap.num_iterations=100"
  "recap.sft_dataset_path=$RECAP_SFT_DATASET_ROOT"

  "recap.policy_eval.enable=true"
  "recap.collect_data.enable=true"
  "recap.compute_return.enable=true"
  "recap.train_value_model.enable=true"
  "recap.value_infer.enable=true"
  "recap.train_policy.enable=true"

  "recap.policy_eval.model_path=$RECAP_INITIAL_POLICY_PATH"
  "recap.policy_eval.max_episodes=50"
  "recap.policy_eval.cluster.actor_rollout_ref.model.path=$RECAP_INITIAL_POLICY_PATH"
  "recap.policy_eval.cluster.actor_rollout_ref.model.override_config.policy_type=libero"
  "recap.policy_eval.cluster.actor_rollout_ref.rollout.output_critic_value=false"
  "recap.policy_eval.cluster.env.env_loop.pipeline_stage_num=2"
  "recap.policy_eval.cluster.env.env_loop.max_interactions=8"
  "recap.policy_eval.cluster.env.env_worker.async_reset=true"
  "recap.policy_eval.cluster.env.env_worker.num_envs=$RECAP_ENV_LOOP_NUM_ENVS"
  "recap.policy_eval.cluster.env.env_worker.simulator.max_episode_steps=$RECAP_ENV_LOOP_MAX_EPISODE_STEPS"
  "recap.policy_eval.cluster.env.env_worker.simulator.task_suite_name=$RECAP_ENV_LOOP_TASK_SUITE_NAME"
  "recap.policy_eval.cluster.env.env_worker.simulator.task_ids=$RECAP_ENV_LOOP_TASK_IDS"
  "recap.policy_eval.cluster.env.env_worker.simulator.num_trials_per_task=$RECAP_ENV_LOOP_NUM_TRIALS_PER_TASK"
  "recap.policy_eval.cluster.env.env_worker.simulator.specific_reset_id=$RECAP_ENV_LOOP_SPECIFIC_RESET_ID"
  "recap.policy_eval.cluster.resource.env.device=cpu"
  "recap.policy_eval.cluster.resource.env.workers_per_node=$RECAP_ENV_LOOP_ENV_WORKERS"
  "recap.policy_eval.cluster.resource.model.gpus_per_node=$RECAP_ENV_LOOP_MODEL_GPUS"

  "recap.collect_data.max_episodes=32"
  "recap.collect_data.cluster.actor_rollout_ref.model.path=$RECAP_INITIAL_POLICY_PATH"
  "recap.collect_data.cluster.actor_rollout_ref.model.override_config.policy_type=libero"
  "recap.collect_data.cluster.actor_rollout_ref.rollout.output_critic_value=false"
  "recap.collect_data.cluster.env.env_loop.pipeline_stage_num=2"
  "recap.collect_data.cluster.env.env_loop.max_interactions=32"
  "recap.collect_data.cluster.env.env_worker.async_reset=true"
  "recap.collect_data.cluster.env.env_worker.num_envs=$RECAP_ENV_LOOP_NUM_ENVS"
  "recap.collect_data.cluster.env.env_worker.simulator.max_episode_steps=$RECAP_ENV_LOOP_MAX_EPISODE_STEPS"
  "recap.collect_data.cluster.env.env_worker.simulator.task_suite_name=$RECAP_ENV_LOOP_TASK_SUITE_NAME"
  "recap.collect_data.cluster.env.env_worker.simulator.task_ids=$RECAP_ENV_LOOP_TASK_IDS"
  "recap.collect_data.cluster.env.env_worker.simulator.num_trials_per_task=$RECAP_ENV_LOOP_NUM_TRIALS_PER_TASK"
  "recap.collect_data.cluster.env.env_worker.simulator.specific_reset_id=$RECAP_ENV_LOOP_SPECIFIC_RESET_ID"
  "recap.collect_data.cluster.env.env_worker.recorder.lerobot.root=/tmp/verl_vla_lerobot_records"
  "recap.collect_data.cluster.env.env_worker.recorder.lerobot.repo_id=local/verl_vla_libero"
  "recap.collect_data.cluster.resource.env.device=cpu"
  "recap.collect_data.cluster.resource.env.workers_per_node=$RECAP_ENV_LOOP_ENV_WORKERS"
  "recap.collect_data.cluster.resource.model.gpus_per_node=$RECAP_ENV_LOOP_MODEL_GPUS"

  "recap.compute_return.dataset.root=$RECAP_SFT_DATASET_ROOT"
  "recap.compute_return.dataset.repo_id=$RECAP_SFT_DATASET_REPO_ID"

  "recap.train_value_model.data.root=$RECAP_SFT_DATASET_ROOT"
  "recap.train_value_model.data.repo_id=$RECAP_SFT_DATASET_REPO_ID"
  "recap.train_value_model.data.num_workers=16"
  "recap.train_value_model.data.prefetch_factor=8"
  "recap.train_value_model.cluster.actor_rollout_ref.model.path=$PROJECT_ROOT/assets/hf_models/recap_value_critic"
  "recap.train_value_model.cluster.checkpoint.resume_mode=auto"
  "recap.train_value_model.cluster.checkpoint.default_local_dir=$RECAP_VALUE_MODEL_CHECKPOINT_DIR"
  "recap.train_value_model.cluster.checkpoint.max_actor_ckpt_to_keep=3"
  "recap.train_value_model.cluster.resource.model.gpus_per_node=$RECAP_SFT_GPUS"
  "recap.train_value_model.cluster.resource.model.nnodes=1"
  "recap.train_value_model.cluster.actor_rollout_ref.actor.mini_batch_size=128"
  "recap.train_value_model.cluster.actor_rollout_ref.actor.micro_batch_size=16"
  "recap.train_value_model.trainer.total_epochs=1000"
  "recap.train_value_model.trainer.save_freq=500"
  "recap.train_value_model.trainer.logger=[console,tensorboard]"

  "recap.value_infer.dataset.root=$RECAP_SFT_DATASET_ROOT"
  "recap.value_infer.dataset.repo_id=$RECAP_SFT_DATASET_REPO_ID"
  "recap.value_infer.model_path=$RECAP_VALUE_MODEL_CHECKPOINT_DIR/global_step_125/actor/huggingface"
  "recap.value_infer.num_gpus=$RECAP_SFT_GPUS"
  "recap.value_infer.data.batch_size=64"
  "recap.value_infer.data.num_workers=16"
  "recap.value_infer.data.prefetch_factor=8"

  "recap.train_policy.dataset.root=$RECAP_SFT_DATASET_ROOT"
  "recap.train_policy.dataset.repo_id=$RECAP_SFT_DATASET_REPO_ID"
  "recap.train_policy.cluster.actor_rollout_ref.model.path=$RECAP_INITIAL_POLICY_PATH"
  "recap.train_policy.cluster.actor_rollout_ref.model.override_config.policy_type=libero"
  "recap.train_policy.cluster.checkpoint.resume_mode=auto"
  "recap.train_policy.cluster.checkpoint.default_local_dir=$RECAP_POLICY_CHECKPOINT_DIR"
  "recap.train_policy.cluster.checkpoint.max_actor_ckpt_to_keep=3"
  "recap.train_policy.cluster.resource.model.gpus_per_node=$RECAP_SFT_GPUS"
  "recap.train_policy.cluster.resource.model.nnodes=1"
  "recap.train_policy.data.batch_size=128"
  "recap.train_policy.data.action_delta_steps=50"
  "recap.train_policy.data.num_workers=16"
  "recap.train_policy.data.prefetch_factor=8"
  "recap.train_policy.cluster.actor_rollout_ref.actor.mini_batch_size=128"
  "recap.train_policy.cluster.actor_rollout_ref.actor.micro_batch_size=16"
  "recap.train_policy.trainer.total_epochs=1000"
  "recap.train_policy.trainer.save_freq=500"
  "recap.train_policy.trainer.logger=[console,tensorboard]"
)

python -m verl_vla.trainer.main_recap "${overrides[@]}" "$@"
