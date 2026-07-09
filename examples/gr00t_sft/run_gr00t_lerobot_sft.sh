#!/usr/bin/env bash
# GR00T N1.6 SFT on lerobot/libero_spatial_image using verl Ray/FSDP.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data}
MODEL_PATH=${MODEL_PATH:-nvidia/GR00T-N1.6-3B}
NORM_STATS_PATH=${NORM_STATS_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_ROOT}/output/gr00t_n1d6_libero_spatial_sft}

if [[ -z "$NORM_STATS_PATH" ]]; then
  echo "NORM_STATS_PATH is required. Generate it with scripts/compute_norm_stats.py --include-min-max." >&2
  exit 2
fi

SFT_REPO_ID=${SFT_REPO_ID:-lerobot/libero_spatial_image}
SFT_ROOT=${SFT_ROOT:-${DATA_ROOT}/datasets/libero_spatial_image}
SFT_REVISION=${SFT_REVISION:-main}
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-32}
SFT_NUM_WORKERS=${SFT_NUM_WORKERS:-4}
SFT_VIDEO_BACKEND=${SFT_VIDEO_BACKEND:-pyav}

NUM_GPUS=${NUM_GPUS:-2}
NUM_NODES=${NUM_NODES:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
SAVE_FREQ=${SAVE_FREQ:-500}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
WRAP_CLASSES=${WRAP_CLASSES:-"[Qwen3DecoderLayer,Siglip2EncoderLayer,BasicTransformerBlock]"}
PROJECT_NAME=${PROJECT_NAME:-gr00t-n1d6-libero-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gr00t_n1d6_libero_spatial_sft}

python scripts/check_gr00t_n1d6_install.py

python -m verl_vla.trainer.main_sft \
  --config-path "$REPO_ROOT/src/verl_vla/trainer/config" \
  --config-name main_gr00t_sft \
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
  cluster.actor_rollout_ref.model.override_config.verl_action_chunk_size=8 \
  cluster.actor_rollout_ref.actor.strategy=fsdp2 \
  cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  cluster.actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  cluster.actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
  cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="$WRAP_CLASSES" \
  cluster.actor_rollout_ref.actor.mini_batch_size="$MINI_BATCH_SIZE" \
  cluster.actor_rollout_ref.actor.micro_batch_size="$MICRO_BATCH_SIZE" \
  cluster.actor_rollout_ref.actor.optim.lr="$LR" \
  cluster.actor_rollout_ref.actor.optim.weight_decay="$WEIGHT_DECAY" \
  cluster.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$WARMUP_RATIO" \
  cluster.resource.model.gpus_per_node="$NUM_GPUS" \
  cluster.resource.model.nnodes="$NUM_NODES" \
  cluster.checkpoint.default_local_dir="$OUTPUT_DIR" \
  cluster.checkpoint.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
  data.repo_id="$SFT_REPO_ID" \
  data.root="$SFT_ROOT" \
  data.revision="$SFT_REVISION" \
  data.batch_size="$SFT_BATCH_SIZE" \
  data.drop_last=True \
  data.num_workers="$SFT_NUM_WORKERS" \
  data.video_backend="$SFT_VIDEO_BACKEND" \
  data.action_delta_steps=16 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="['console']"
