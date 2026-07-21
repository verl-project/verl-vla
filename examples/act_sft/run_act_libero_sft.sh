set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

OUTPUT_DIR="/file_system/liujincheng/output/act_libero_sft"
MODEL_PATH="$REPO_ROOT/models/act_libero_sft_init"
TOKENIZER_PATH="$MODEL_PATH"

SFT_REPO_ID="$REPO_ROOT/outputs/record_gamepad/lerobot/local/verl_vla_libero_gamepad_record"
SFT_REVISION="main"
SFT_BATCH_SIZE=32
SFT_NUM_WORKERS=8
SFT_PREFETCH_FACTOR=8
SFT_PERSISTENT_WORKERS=True
SFT_PIN_MEMORY=True
SFT_VIDEO_BACKEND="pyav"
SFT_ACTION_DELTA_STEPS=10
ACT_ACTION_STEPS=10

NUM_GPUS=2
NUM_NODES=1

TOTAL_EPOCHS=10000
MINI_BATCH_SIZE=32
MICRO_BATCH_SIZE=16
LR=5e-5
FREEZE_VISION_TOWER=True
BACKBONE_LR=1e-5
SAVE_FREQ=500
MAX_ACTOR_CKPT_TO_KEEP=3

PROJECT_NAME="act-libero-sft"
EXPERIMENT_NAME="libero_sft_current_state_stats"

PYTHON=python

$PYTHON -m verl_vla.entrypoints.train.sft \
    +model/override@cluster.actor_rollout_ref.model.override_config=act \
    model/adapter@cluster.actor_rollout_ref.model.adapter=act \
    cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
    cluster.actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
    cluster.actor_rollout_ref.actor.strategy=fsdp2 \
    cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[ACTEncoderLayer,ACTDecoderLayer] \
    cluster.actor_rollout_ref.actor.optim.clip_grad=1.0 \
    cluster.actor_rollout_ref.model.enable_gradient_checkpointing=False \
    cluster.actor_rollout_ref.model.use_remove_padding=False \
    cluster.actor_rollout_ref.model.trust_remote_code=True \
    +cluster.actor_rollout_ref.model.load_tokenizer=False \
    cluster.actor_rollout_ref.model.override_config.attn_implementation=eager \
    cluster.actor_rollout_ref.model.adapter.policy_type=libero \
    cluster.actor_rollout_ref.model.override_config.chunk_size=$ACT_ACTION_STEPS \
    cluster.actor_rollout_ref.model.adapter.n_action_steps=$ACT_ACTION_STEPS \
    cluster.actor_rollout_ref.model.adapter.freeze_vision_tower=$FREEZE_VISION_TOWER \
    cluster.actor_rollout_ref.model.adapter.optimizer_lr_backbone=$BACKBONE_LR \
    cluster.actor_rollout_ref.data_keys.action_mask=action_is_pad \
    data.repo_id="$SFT_REPO_ID" \
    data.revision="$SFT_REVISION" \
    data.batch_size=$SFT_BATCH_SIZE \
    data.drop_last=True \
    data.num_workers=$SFT_NUM_WORKERS \
    data.prefetch_factor=$SFT_PREFETCH_FACTOR \
    data.persistent_workers=$SFT_PERSISTENT_WORKERS \
    data.pin_memory=$SFT_PIN_MEMORY \
    data.video_backend="$SFT_VIDEO_BACKEND" \
    data.action_delta_steps=$SFT_ACTION_DELTA_STEPS \
    cluster.resource.model.nnodes=$NUM_NODES \
    cluster.resource.model.gpus_per_node=$NUM_GPUS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    cluster.actor_rollout_ref.actor.mini_batch_size=$MINI_BATCH_SIZE \
    cluster.actor_rollout_ref.actor.micro_batch_size=$MICRO_BATCH_SIZE \
    cluster.actor_rollout_ref.actor.optim.lr=$LR \
    'cluster.actor_rollout_ref.actor.checkpoint.save_contents=["model", "optimizer", "extra", "hf_model"]' \
    'cluster.actor_rollout_ref.actor.checkpoint.load_contents=["model", "optimizer", "extra"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger="['console','tensorboard']" \
    cluster.checkpoint.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq=$SAVE_FREQ \
    cluster.checkpoint.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
    "$@"
