set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

CONFIG_DIR="${REPO_ROOT}/src/verl_vla/workflows/config"
CONFIG_NAME="rob_sac_trainer.yaml"

libero_train_path=${TRAIN_FILE:-"/root/data/libero_rl/libero_spatial/task_2_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate/train.parquet"}
libero_test_path=${TEST_FILE:-"/root/data/libero_rl/libero_spatial/task_2_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate/test.parquet"}

train_files=$libero_train_path
test_files=$libero_test_path

OUTPUT_DIR=${OUTPUT_DIR:-${MLP_MODEL_OUTPUT:-"/file_system/liujincheng/output/pi05_lerobot_sft233"}}
VIDEO_OUTPUT=${VIDEO_OUTPUT:-"/home/miical/videos"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"/file_system/liujincheng/models/pi05_lerobot_sft_global_step_128000_hf"}
TOKENIZER_PATH="${TOKENIZER_PATH:-$SFT_MODEL_PATH}"

# Physical Node Config
NUM_NODES=${NUM_NODES:-1}                      # number of nodes
NUM_GPUS=${NUM_GPUS:-2}                        # total number of gpus per node

# Role Config
SIM_NODES=${SIM_NODES:-1}                      # number of nodes for sim
ENV_DEVICE=${ENV_DEVICE:-cpu}                  # env worker device: cpu or cuda
NUM_ENV_WORKERS=${NUM_ENV_WORKERS:-1}          # set >0 even when NUM_ENV_GPUS=0 for CPU-only sim
NUM_ROLLOUT_GPUS=${NUM_ROLLOUT_GPUS:-2}        # number of gpus for rollout workers per node

# Rollout Config
# NOTE: BATCH_SIZE * ROLLOUT_N == NUM_ENV_WORKERS * NUM_STAGE * NUM_ENV
BATCH_SIZE=${BATCH_SIZE:-1}                    # batch size for dataloaders per step
ROLLOUT_N=${ROLLOUT_N:-8}                      # response number for each prompt; pads one real env across rollout GPUs
NUM_STAGE=${NUM_STAGE:-1}                      # number of pipeline stages
NUM_ENV=${NUM_ENV:-1}                          # number of envs per env worker

NUM_ACTION_CHUNKS=${NUM_ACTION_CHUNKS:-10}     # number of action chunks
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-250}    # max episode steps for each env

# Training Config
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-16}         # mini batch size
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}        # micro batch size
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10000}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:--1}
ROLLOUT_INTERVAL=${ROLLOUT_INTERVAL:-1}

# libero, isaac, or lerobot
SIM_TYPE=${SIM_TYPE:-"lerobot"}
LEROBOT_CONFIG_PATH=${LEROBOT_CONFIG_PATH:-"/home/miical/Projects/myrob/hilserl/env_config.json"}
PROJECT_NAME=${PROJECT_NAME:-"pi05-lerobot-sac"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${SIM_TYPE}_reinforce_plus_plus"}

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# avoiding warnings
mkdir -p /root/LIBERO/libero/libero/../datasets
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)

# osmesa if env workers are on cpu or Hopper GPUs
if [ "$ENV_DEVICE" = "cpu" ] || echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa"
    export MUJOCO_GL=osmesa
fi

export VERL_LOGGING_LEVEL=INFO

$PYTHON -m verl_vla.entrypoints.train.sac \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.shuffle=True \
    data.seed=55 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$NUM_ENV \
    +env.train.single_env_rollout=True \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    env.disagg_sim.enable=True \
    env.disagg_sim.nnodes=$SIM_NODES \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.train.simulator_type=$SIM_TYPE \
    +env.train.lerobot_config_path=$LEROBOT_CONFIG_PATH \
    env.actor.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    env.actor.model.action_dim=7 \
    env.train.device=$ENV_DEVICE \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=True \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.seed=42 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SiglipEncoderLayer,GemmaDecoderLayerWithExpert] \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.sac_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.sac_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.replay_pool_save_interval=500 \
    actor_rollout_ref.actor.replay_pool_single_size=1000 \
    actor_rollout_ref.actor.replay_pool_save_dir="$OUTPUT_DIR/replay_pool" \
    actor_rollout_ref.actor.critic_warmup_steps=200 \
    actor_rollout_ref.actor.actor_update_interval=1 \
    actor_rollout_ref.actor.warm_rollout_steps=60 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.checkpoint.load_contents=[model] \
    actor_rollout_ref.actor.actor_ema_dynamic_enabled=True \
    actor_rollout_ref.actor.actor_ema_strength_initial=0.8 \
    actor_rollout_ref.actor.actor_ema_strength_final=0.98 \
    actor_rollout_ref.actor.actor_ema_schedule_method=square \
    actor_rollout_ref.actor.sac.critic_target_ema_dynamic_enabled=True \
    actor_rollout_ref.actor.sac.critic_target_ema_strength_initial=0 \
    actor_rollout_ref.actor.sac.critic_target_ema_strength_final=0.8 \
    actor_rollout_ref.actor.sac.critic_target_ema_schedule_method=square \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.adapter.embodiment=lerobot \
    actor_rollout_ref.model.adapter.critic.type=cross_attn \
    actor_rollout_ref.model.adapter.critic.prefix_embed_dim=2048 \
    actor_rollout_ref.model.adapter.critic.input_dim=2140 \
    actor_rollout_ref.model.adapter.critic.hidden_dims=[1024,512,256] \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.n_env_workers_per_node=$NUM_ENV_WORKERS \
    trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.rollout_interval=$ROLLOUT_INTERVAL \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_only=False \
    trainer.val_before_train=False
