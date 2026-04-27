set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

CONFIG_DIR="${REPO_ROOT}/src/verl_vla/trainer/config"
CONFIG_NAME="rob_sft_trainer.yaml"

OUTPUT_DIR=${OUTPUT_DIR:-"/file_system/liujincheng/output/pi05_lerobot_sft"}
MODEL_PATH=${MODEL_PATH:-"/file_system/liujincheng/models/torch_pi05_base"}
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"

SFT_REPO_ID=${SFT_REPO_ID:-"/root/.cache/huggingface/hub/datasets--Miical--so101-30episodes-augmented/snapshots/464f30f26c8f13ec75310845a2ccba012c857b27"}
SFT_REVISION=${SFT_REVISION:-"main"}
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-32}
SFT_NUM_WORKERS=${SFT_NUM_WORKERS:-0}
SFT_VIDEO_BACKEND=${SFT_VIDEO_BACKEND:-"pyav"}
NUM_GPUS=${NUM_GPUS:-2}
NUM_NODES=${NUM_NODES:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10000}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
LR=${LR:-1e-4}
SAVE_FREQ=${SAVE_FREQ:-1000}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}

PROJECT_NAME=${PROJECT_NAME:-"pi05-lerobot-sft"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"lerobot_sft_preview"}

PYTHON=python

$PYTHON -m verl_vla.trainer.main_sft \
    --config-path $CONFIG_DIR \
    --config-name $CONFIG_NAME \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SiglipEncoderLayer,GemmaDecoderLayerWithExpert] \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.model.override_config.attn_implementation=eager \
    data.sft.enable=True \
    data.sft.repo_id="$SFT_REPO_ID" \
    data.sft.revision="$SFT_REVISION" \
    data.sft.batch_size=$SFT_BATCH_SIZE \
    data.sft.drop_last=True \
    data.sft.num_workers=$SFT_NUM_WORKERS \
    data.sft.video_backend="$SFT_VIDEO_BACKEND" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.total_epochs=$TOTAL_EPOCHS \
    actor_rollout_ref.actor.sft_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.sft_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.optim.lr=$LR \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger="['console']" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP
