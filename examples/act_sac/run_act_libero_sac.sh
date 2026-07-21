set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

OUTPUT_DIR="/file_system/liujincheng/output/act_libero_sac"
MODEL_PATH="/file_system/liujincheng/output/act_libero_sft/global_step_1000/actor/huggingface"
if [ -f "$MODEL_PATH" ]; then
    MODEL_PATH="$(dirname "$MODEL_PATH")"
fi
TOKENIZER_PATH="$MODEL_PATH"

NUM_NODES=1
NUM_GPUS=2

SEPARATE_ROLLOUT_MODEL=False
ROLLOUT_NUM_NODES=$NUM_NODES
ROLLOUT_NUM_GPUS=1

ENV_DEVICE=cpu
ENV_WORKERS_PER_NODE=8
ENV_NUM_ENVS=1

PIPELINE_STAGE_NUM=2
MAX_EPISODE_STEPS=256
ACTION_CHUNK_STEPS=10
MAX_INTERACTIONS=32

LIBERO_TASK_SUITE=libero_spatial
LIBERO_TASK_IDS="[0]"
LIBERO_NUM_TRIALS_PER_TASK=null

TOTAL_TRAINING_STEPS=50
ROLLOUT_INTERVAL=10
ROLLOUT_TIMES=1
WARM_ROLLOUT_STEPS=1
MINI_BATCH_SIZE=256
MICRO_BATCH_SIZE=16
LR=1e-4
CRITIC_LR=1e-4
ACTOR_UPDATE_INTERVAL=2
CRITIC_WARMUP_STEPS=0
TEST_FREQ=-1
SAVE_FREQ=-1
ASYNC_ROLLOUT=False
VAL_BEFORE_TRAIN=False
ENV_AUTO_RESET=False
PROJECT_NAME="act-libero-sac"
EXPERIMENT_NAME="libero_sac_preview"
MAX_ACTOR_CKPT_TO_KEEP=3
SAVE_VIDEO=True
VIDEO_DIR="${OUTPUT_DIR}/videos"

PYTHON=python

if [ "$ENV_DEVICE" = "cpu" ]; then
    export MUJOCO_GL=osmesa
else
    export MUJOCO_GL=egl
fi
export VERL_LOGGING_LEVEL=INFO

"$PYTHON" -m verl_vla.entrypoints.train.sac \
    model/override@cluster.actor_rollout_ref.model.override_config=act \
    model/adapter@cluster.actor_rollout_ref.model.adapter=act \
    cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
    cluster.actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
    cluster.actor_rollout_ref.model.trust_remote_code=True \
    cluster.actor_rollout_ref.model.load_tokenizer=False \
    cluster.actor_rollout_ref.model.enable_gradient_checkpointing=False \
    cluster.actor_rollout_ref.model.use_remove_padding=False \
    cluster.actor_rollout_ref.model.override_config.attn_implementation=eager \
    cluster.actor_rollout_ref.model.adapter.policy_type=libero \
    cluster.actor_rollout_ref.model.override_config.chunk_size=$ACTION_CHUNK_STEPS \
    cluster.actor_rollout_ref.model.adapter.n_action_steps=$ACTION_CHUNK_STEPS \
    cluster.actor_rollout_ref.model.adapter.critic.enabled=True \
    cluster.actor_rollout_ref.model.adapter.critic.type=mean_pool \
    cluster.actor_rollout_ref.data_keys.action_mask=null \
    cluster.actor_rollout_ref.actor.strategy=fsdp2 \
    cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[ACTEncoderLayer,ACTDecoderLayer] \
    cluster.actor_rollout_ref.actor.optim.lr=$LR \
    cluster.actor_rollout_ref.actor.critic.lr=$CRITIC_LR \
    cluster.actor_rollout_ref.actor.mini_batch_size=$MINI_BATCH_SIZE \
    cluster.actor_rollout_ref.actor.micro_batch_size=$MICRO_BATCH_SIZE \
    cluster.actor_rollout_ref.actor.actor_update_interval=$ACTOR_UPDATE_INTERVAL \
    cluster.actor_rollout_ref.actor.critic.warmup_steps=$CRITIC_WARMUP_STEPS \
    'cluster.actor_rollout_ref.actor.checkpoint.save_contents=["model", "optimizer", "extra", "hf_model"]' \
    'cluster.actor_rollout_ref.actor.checkpoint.load_contents=["model", "optimizer", "extra"]' \
    cluster.actor_rollout_ref.rollout.mode=async_envloop \
    cluster.actor_rollout_ref.rollout.output_critic_value=False \
    cluster.resource.model.nnodes=$NUM_NODES \
    cluster.resource.model.gpus_per_node=$NUM_GPUS \
    cluster.resource.separate_rollout_model.enabled=$SEPARATE_ROLLOUT_MODEL \
    cluster.resource.separate_rollout_model.nnodes=$ROLLOUT_NUM_NODES \
    cluster.resource.separate_rollout_model.gpus_per_node=$ROLLOUT_NUM_GPUS \
    cluster.resource.env.device=$ENV_DEVICE \
    cluster.resource.env.workers_per_node=$ENV_WORKERS_PER_NODE \
    cluster.env.env_loop.pipeline_stage_num=$PIPELINE_STAGE_NUM \
    cluster.env.env_loop.max_interactions=$MAX_INTERACTIONS \
    cluster.env.env_worker.simulator.libero.max_episode_steps=$MAX_EPISODE_STEPS \
    cluster.env.env_worker.modes="[train]" \
    +cluster.env.env_worker.device=$ENV_DEVICE \
    cluster.env.env_worker.num_envs=$ENV_NUM_ENVS \
    cluster.env.env_worker.auto_reset=$ENV_AUTO_RESET \
    cluster.env.env_worker.simulator.libero.task_suite_name="$LIBERO_TASK_SUITE" \
    cluster.env.env_worker.simulator.libero.task_ids=$LIBERO_TASK_IDS \
    cluster.env.env_worker.simulator.libero.num_trials_per_task=$LIBERO_NUM_TRIALS_PER_TASK \
    cluster.env.env_worker.recorder.enable=$SAVE_VIDEO \
    cluster.env.env_worker.recorder.recorders="[video]" \
    cluster.env.env_worker.recorder.video.root="$VIDEO_DIR" \
    trainer.async_rollout=$ASYNC_ROLLOUT \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.rollout_interval=$ROLLOUT_INTERVAL \
    trainer.rollout_times=$ROLLOUT_TIMES \
    trainer.warm_rollout_steps=$WARM_ROLLOUT_STEPS \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_freq=$SAVE_FREQ \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger="['console']" \
    ray_kwargs.ray_init.logging_level=INFO \
    cluster.checkpoint.default_local_dir="$OUTPUT_DIR" \
    cluster.checkpoint.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
    "$@"
