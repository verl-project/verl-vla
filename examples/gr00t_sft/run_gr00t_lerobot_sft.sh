#!/usr/bin/env bash
# GR00T N1.7 LeRobot SFT on LIBERO — mirrors the pi SFT recipe but corrected for
# the current main_sft config tree (the reference pi script used stale key paths
# from a removed rob_sft_trainer.yaml).
#
# Correct key paths in main_sft (verified via `--cfg job`):
#   cluster.actor_rollout_ref.model.*        (path, tokenizer_path, override_config)
#   cluster.actor_rollout_ref.actor.*        (strategy, fsdp_config, optim, mini/micro_batch_size)
#   cluster.resource.model.{gpus_per_node,nnodes}
#   cluster.checkpoint.{default_local_dir,max_actor_ckpt_to_keep}
#   data.*                                   (top-level LeRobotDataLoaderConfig)
#   trainer.*                                (top-level SFTTrainerConfig)
#
# The trainer auto-selects the GR00T model because MODEL_PATH's config.json has
# "model_type": "gr00t_torch".
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="${REPO_ROOT}/src/verl_vla/trainer/config"
CONFIG_NAME="main_sft"

DATA_ROOT=${DATA_ROOT:-"/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${DATA_ROOT}/output/gr00t_lerobot_sft"}
MODEL_PATH=${MODEL_PATH:-"${DATA_ROOT}/models/gr00t_n1d7_libero_base"}
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"

# LeRobotDataset needs a valid HF repo_id + a local `root`; with a complete
# local root it loads offline (no download).
SFT_REPO_ID=${SFT_REPO_ID:-"lerobot/libero_spatial_image"}
SFT_ROOT=${SFT_ROOT:-"${DATA_ROOT}/datasets/libero_spatial_image"}
SFT_REVISION=${SFT_REVISION:-"main"}
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-32}
SFT_NUM_WORKERS=${SFT_NUM_WORKERS:-2}
SFT_VIDEO_BACKEND=${SFT_VIDEO_BACKEND:-"pyav"}
# Number of future action steps to fetch per sample (produces the action chunk +
# action_is_pad mask the SFT loss needs). Match GR00T's sft_action_horizon=16.
SFT_ACTION_DELTA_STEPS=${SFT_ACTION_DELTA_STEPS:-16}

NUM_GPUS=${NUM_GPUS:-2}
NUM_NODES=${NUM_NODES:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
SAVE_FREQ=${SAVE_FREQ:-500}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}

# GR00T FSDP wrap classes (verified against transformers 4.57.3):
#   Qwen3-VL backbone -> Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock
#   DiT action head   -> BasicTransformerBlock
WRAP_CLASSES=${WRAP_CLASSES:-"[Qwen3VLTextDecoderLayer,Qwen3VLVisionBlock,BasicTransformerBlock]"}

PROJECT_NAME=${PROJECT_NAME:-"gr00t-lerobot-sft"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"gr00t_libero_sft"}

PYTHON=python

$PYTHON -m verl_vla.trainer.main_sft \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
    cluster.actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
    cluster.actor_rollout_ref.model.enable_gradient_checkpointing=False \
    cluster.actor_rollout_ref.model.use_remove_padding=False \
    cluster.actor_rollout_ref.model.trust_remote_code=False \
    cluster.actor_rollout_ref.model.override_config.policy_type=libero \
    cluster.actor_rollout_ref.model.override_config.action_chunk_size=10 \
    cluster.actor_rollout_ref.model.override_config.sac_enable=False \
    cluster.actor_rollout_ref.model.override_config.flow_sde_enable=False \
    cluster.actor_rollout_ref.actor.strategy=fsdp2 \
    cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="$WRAP_CLASSES" \
    cluster.actor_rollout_ref.actor.mini_batch_size=$MINI_BATCH_SIZE \
    cluster.actor_rollout_ref.actor.micro_batch_size=$MICRO_BATCH_SIZE \
    cluster.actor_rollout_ref.actor.optim.lr=$LR \
    cluster.actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    cluster.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$WARMUP_RATIO \
    cluster.resource.model.gpus_per_node=$NUM_GPUS \
    cluster.resource.model.nnodes=$NUM_NODES \
    cluster.checkpoint.default_local_dir="$OUTPUT_DIR" \
    cluster.checkpoint.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
    data.repo_id="$SFT_REPO_ID" \
    data.root="$SFT_ROOT" \
    data.revision="$SFT_REVISION" \
    data.batch_size=$SFT_BATCH_SIZE \
    data.drop_last=True \
    data.num_workers=$SFT_NUM_WORKERS \
    data.video_backend="$SFT_VIDEO_BACKEND" \
    data.action_delta_steps=$SFT_ACTION_DELTA_STEPS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger="['console']"
