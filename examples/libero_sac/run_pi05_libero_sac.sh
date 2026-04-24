set -x
libero_train_path=$HOME/data/libero_rl/libero_spatial/train.parquet
libero_test_path=$HOME/data/libero_rl/libero_spatial/test.parquet

train_files=$libero_train_path
test_files=$libero_test_path

OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"$HOME/models/vla_libero_grpo"}
VIDEO_OUTPUT=${MLP_MODEL_OUTPUT:-"$HOME"}/video
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/data/pi05_libero_torch"}
TOKENIZER_PATH="$SFT_MODEL_PATH"

# Physical Node Config
NUM_NODES=1                                    # number of nodes
NUM_GPUS=4                                     # total number of gpus per node

# Role Config
ENV_DEVICE=cpu                                 # env worker device: cpu or cuda
NUM_ENV_WORKERS=2                              # number of CPU env workers per node
NUM_ROLLOUT_GPUS=4                             # number of gpus for actor/rollout workers per node

# Rollout Config
# NOTE: TRAIN_BATCH_SIZE * ROLLOUT_N == NUM_ENV_WORKERS * NUM_STAGE * NUM_ENV
TRAIN_BATCH_SIZE=32                            # batch size for dataloaders per step
ROLLOUT_N=1                                    # response number for each prompt (for GRPO)
NUM_STAGE=2                                    # number of pipeline stages
NUM_ENV=8                                      # number of envs per env worker

NUM_ACTION_CHUNKS=10                           # number of action chunks
MAX_EPISODE_STEPS=40                           # max episode steps for each env
                                               # max_interactions = MAX_EPISODE_STEPS / num_action_chunks

# Training Config
MINI_BATCH_SIZE=1024                           # mini batch size (batch size per GPU, automatically multiplied by ROLLOUT_N)
MICRO_BATCH_SIZE=8                             # micro batch size (per GPU, for gradient accumulation, should divide MINI_BATCH_SIZE)

# simulator config
PYTHON=python
SIM_TYPE=${SIM_TYPE:-"libero"}
PROJECT_NAME="vla_libero_RL"
EXPERIMENT_NAME="sac_libero_pi05"

# avoiding warnings
mkdir /root/LIBERO/libero/libero/../datasets
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)

# force cpu in Hopper
if echo "$gpu_name" | grep "NVIDIA H"; then
    echo "enable MUJOCO_GL=osmesa in Hopper"
    export MUJOCO_GL=osmesa
    if [ "$ENV_DEVICE" != "cpu" ]; then
        echo "ENV_DEVICE must be cpu on Hopper GPUs"
        exit 1
    fi
fi

# force osmesa if env workers are on cpu
if [ "$ENV_DEVICE" = "cpu" ]; then
    export MUJOCO_GL=osmesa
fi
export VERL_LOGGING_LEVEL=INFO


$PYTHON -m verl_vla.trainer.main_sac \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=4 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$NUM_ENV \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.train.simulator_type=$SIM_TYPE \
    env.actor.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    env.actor.model.action_dim=7 \
    env.train.device=$ENV_DEVICE \
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
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
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
    trainer.rollout_interval=20 \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=300 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    trainer.val_before_train=False
