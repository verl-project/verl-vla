#!/usr/bin/env bash
#
# Run GR00T N1.6 DSRL (latent-noise steering, arXiv:2506.15799) on an Arena
# task via verl_vla.entrypoints.train.sac.
#
# DSRL keeps the WHOLE GR00T policy frozen and trains only a small SAC noise
# actor over the flow-matching initial noise x0 (plus the SAC critic). The
# steering noise is the SAC action: it seeds the frozen flow head, which
# deterministically (Euler ODE) decodes it into the env action. Follows the
# RLinf DSRL recipe (libero_spatial_dsrl_openpi.yaml) adapted to this repo's
# SAC trainer. Pick the task with ARENA_TASK:
#
#   ARENA_TASK=gr1     (default)  GR1 fridge (put_item_in_fridge_and_close_door),
#                                 gr1_joint 26-DOF, embodiment_tag=gr1.
#   ARENA_TASK=libero             Franka Abs-IK LIBERO, eef_pose 7-DOF (rel_rotvec),
#                                 embodiment_tag=new_embodiment; task via TASK_SUITE/TASK_ID.
#
# Notes vs. plain SAC (run_gr00t_arena_sac.sh):
#   * adapter.dsrl.enabled=true freezes the VLA; only ~0.5M params train.
#   * critic action defaults switch to the noise space (max_action_dim, one
#     shared step) — do NOT override adapter.critic.action_horizon here.
#   * actor lr is the noise-actor lr (3e-4, RLinf parity), not a VLA lr.
#   * auto entropy tuning is on; TARGET_ENTROPY defaults to -(noise_dim/2)
#     with GR00T's padded noise_dim=128 → -64 (RLinf uses -16 for pi0's 32).
#   * BACKUP_ENTROPY=False keeps -alpha*log_pi out of the critic TD target
#     (RLinf parity; the 128-dim summed log-prob would dominate the bootstrap).
#   * The SAC launcher's FREEZE_ACTION_IO / FLOW_SDE_* knobs are intentionally
#     absent: DSRL freezes everything and owns the exploration noise
#     (flow_sde_enable=true would raise at model init).
#   * TD3+BC and offline RLPD prefill are incompatible with DSRL (demos are
#     env actions, not steering noise) — keep them disabled.
#
# Must run inside the GR00T docker (isaaclab_arena:cuda_gr00t_gn16). Launch from
# the host with:
#
#   ARENA_TASK=gr1 INNER_SCRIPT=examples/gr00t_arena_dsrl/run_gr00t_arena_dsrl.sh \
#     OUTPUT_ROOT=/eval/outputs/arena_gr00t_gr1_dsrl \
#     examples/gr00t_arena_sac/run_docker.sh
#
# ─────────────────────────────────────────────────────────────────────────────
# Overridable via env vars (see knobs below). Extra Hydra overrides: "$@"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/isaac-sim/python.sh}"

ARENA_TASK="${ARENA_TASK:-gr1}"

# ── Common paths ─────────────────────────────────────────────────────────────
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/models/checkpoint-10000}"
NUM_ACTION_CHUNKS="${NUM_ACTION_CHUNKS:-16}"

# ── Task-specific defaults ───────────────────────────────────────────────────
# EXTRA_OVERRIDES holds hydra overrides that differ per task.
EXTRA_OVERRIDES=()
case "$ARENA_TASK" in
  gr1)
    GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-gr1}"
    GROOT_EMBODIMENT_ID="${GROOT_EMBODIMENT_ID:-20}"
    ACTION_DIM="${ACTION_DIM:-26}"
    ARENA_ENVIRONMENT="gr1"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_gr00t_gr1_dsrl}"
    PROJECT_NAME="${PROJECT_NAME:-gr00t-arena-gr1-dsrl}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-arena_gr00t_gr1_fridge_dsrl}"
    EPISODIC_REPLAY="${EPISODIC_REPLAY:-True}"
    # 32 interactions × 16 action chunks = 512 env steps (episode_length_s≈10 @ 50 Hz).
    MAX_INTERACTIONS="${MAX_INTERACTIONS:-32}"
    export ARENA_GR1_JOINT_SPACE_DIR="${ARENA_GR1_JOINT_SPACE_DIR:-/workspaces/isaaclab_arena/isaaclab_arena_gr00t/embodiments/gr1}"
    EXTRA_RAY_ENV=(
      "+ray_kwargs.ray_init.runtime_env.env_vars.ARENA_GR1_JOINT_SPACE_DIR=$ARENA_GR1_JOINT_SPACE_DIR"
    )
    ;;
  libero)
    GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-new_embodiment}"
    GROOT_EMBODIMENT_ID="${GROOT_EMBODIMENT_ID:-10}"
    ACTION_DIM="${ACTION_DIM:-7}"
    ARENA_ENVIRONMENT="libero"
    TASK_SUITE="${TASK_SUITE:-libero_spatial}"
    TASK_ID="${TASK_ID:-3}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_gr00t_libero_dsrl}"
    PROJECT_NAME="${PROJECT_NAME:-gr00t-arena-libero-dsrl}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-arena_gr00t_libero_${TASK_SUITE}_task${TASK_ID}_dsrl}"
    # 10 interactions × 16 action chunks = 160 env steps (matches LIBERO eval default).
    MAX_INTERACTIONS="${MAX_INTERACTIONS:-10}"
    # Episodes run up to max_episode_steps=512 env steps but a rollout window is only
    # 160, so episodes span 3-4 windows. Collect them episodically so the early/middle
    # transitions are not dropped (docs/reinforcement-learning/episodic-replay.md).
    EPISODIC_REPLAY="${EPISODIC_REPLAY:-True}"

    export LIBERO_IN_LAB_ROOT="${LIBERO_IN_LAB_ROOT:-/libero_in_lab}"
    if [[ ! -d "$LIBERO_IN_LAB_ROOT" ]]; then
      echo "[warn] LIBERO_IN_LAB_ROOT='$LIBERO_IN_LAB_ROOT' missing — Arena LIBERO may fail to resolve USD/hdf5"
    fi
    ARENA_LIBERO_DATA_DIR="${ARENA_LIBERO_DATA_DIR:-/workspaces/isaaclab_arena/isaaclab_arena_examples/external_environments/libero/data}"
    export LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-$ARENA_LIBERO_DATA_DIR/config}"
    EXTRA_RAY_ENV=(
      "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_IN_LAB_ROOT=$LIBERO_IN_LAB_ROOT"
      "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_CONFIG_DIR=$LIBERO_CONFIG_DIR"
    )
    EXTRA_OVERRIDES+=(
      "cluster.env.env_worker.simulator.arena.libero.libero_task_suite=$TASK_SUITE"
      "cluster.env.env_worker.simulator.arena.libero.libero_task_id=$TASK_ID"
    )
    ;;
  *)
    echo "Unknown ARENA_TASK='$ARENA_TASK' (expected: gr1 | libero)" >&2
    exit 1
    ;;
esac

# ── Experiment identity ──────────────────────────────────────────────────────
REPLAY_POOL_DIR="${REPLAY_POOL_DIR:-$OUTPUT_ROOT/replay_pools}"

# ── Topology (Ray resource pools under cluster.resource) ─────────────────────
# Default: co-located 4 env workers + 4 model workers, 8 Isaac envs per env GPU.
# Scale workers with NUM_ENV_GPUS / NUM_MODEL_GPUS; override NUM_ENV for denser sims.
NUM_NODES="${NUM_NODES:-1}"
NUM_ENV_GPUS="${NUM_ENV_GPUS:-4}"
NUM_MODEL_GPUS="${NUM_MODEL_GPUS:-4}"
NUM_ENV="${NUM_ENV:-8}"
NUM_STAGE="${NUM_STAGE:-2}"

# ── SAC batch / schedule ─────────────────────────────────────────────────────
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5000}"
ROLLOUT_INTERVAL="${ROLLOUT_INTERVAL:-20}"
WARM_ROLLOUT_STEPS="${WARM_ROLLOUT_STEPS:-5}"
CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-100}"
ACTOR_UPDATE_INTERVAL="${ACTOR_UPDATE_INTERVAL:-1}"
SAVE_FREQ="${SAVE_FREQ:-500}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
# Trajectories aggregated per eval; Arena has no fixed benchmark, so without
# this the eval SR is a single 0/1 episode instead of an average.
EVAL_EPISODES="${EVAL_EPISODES:-$((NUM_ENV_GPUS * NUM_ENV))}"

# ── Episodic replay collection (requires auto_reset=true, which this script sets) ──
# Task branches may set a task-specific default above (libero: True); fall back off.
EPISODIC_REPLAY="${EPISODIC_REPLAY:-False}"
EPISODIC_MAX_OPEN_LEN="${EPISODIC_MAX_OPEN_LEN:-128}"

# ── DSRL noise actor / critic optimisation (RLinf DSRL parity) ───────────────
NOISE_ACTOR_LR="${NOISE_ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
CRITIC_TAU="${CRITIC_TAU:-0.005}"

# ── SAC entropy (auto-tuned alpha over the 128-dim steering noise) ───────────
AUTO_ENTROPY="${AUTO_ENTROPY:-True}"
ALPHA_TYPE="${ALPHA_TYPE:-softplus}"
INITIAL_ALPHA="${INITIAL_ALPHA:-1.0}"
TARGET_ENTROPY="${TARGET_ENTROPY:--64.0}"
# RLinf DSRL parity: no -alpha*log_pi term in the critic TD target (the summed
# noise log-prob would dominate the bootstrap before alpha anneals).
BACKUP_ENTROPY="${BACKUP_ENTROPY:-False}"

# ── SAC stability / replay knobs (shared with the SAC launcher ablations) ────
# Actor EMA acts on the tiny noise actor under DSRL; null disables (default).
EMA_DECAY="${EMA_DECAY:-null}"
# Critic capacity knobs — the critic trains as usual under DSRL (it scores the
# steering noise), so these remain meaningful. FREEZE_ACTION_IO / FLOW_SDE_*
# are intentionally absent (see header notes).
CRITIC_POOL_PROJ_DIM="${CRITIC_POOL_PROJ_DIM:-0}"                  # SAC baseline 256
CRITIC_LAYERNORM="${CRITIC_LAYERNORM:-True}"                      # SAC baseline True
ACTOR_POSITIVE_SAMPLE_RATIO="${ACTOR_POSITIVE_SAMPLE_RATIO:-0.8}"

# ── Logging ──────────────────────────────────────────────────────────────────
TRAINER_LOGGER="${TRAINER_LOGGER:-[console]}"

mkdir -p "$OUTPUT_ROOT/videos" "$OUTPUT_ROOT/checkpoints" "$REPLAY_POOL_DIR" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# GR00T docker runtime env.
# ─────────────────────────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL=INFO
export TORCH_CUDNN_SDPA_ENABLED="${TORCH_CUDNN_SDPA_ENABLED:-0}"
export PYTHONPATH="/opt/groot_deps:$REPO_ROOT/src:/workspaces/isaaclab_arena:${PYTHONPATH:-}"

# ─────────────────────────────────────────────────────────────────────────────
# main_sac launch (DSRL flavour).
#
# Hydra overrides:
#   * adapter.dsrl.enabled=true  -> freeze the VLA, train the noise actor
#   * critic action defaults auto-switch to the steering-noise space
#     (action_dim=max_action_dim, action_horizon=1) — no override needed.
# ─────────────────────────────────────────────────────────────────────────────
"$PYTHON" -m verl_vla.entrypoints.train.sac \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDNN_SDPA_ENABLED="0"' \
  "${EXTRA_RAY_ENV[@]}" \
  "model/adapter@cluster.actor_rollout_ref.model.adapter=gr00t" \
  "model/override@cluster.actor_rollout_ref.model.override_config=gr00t" \
  "cluster.env.env_worker.simulator.arena.environment=$ARENA_ENVIRONMENT" \
  "cluster.actor_rollout_ref.model.path=$GROOT_MODEL_PATH" \
  "cluster.actor_rollout_ref.model.tokenizer_path=$GROOT_MODEL_PATH" \
  "cluster.actor_rollout_ref.model.trust_remote_code=True" \
  "cluster.actor_rollout_ref.model.load_tokenizer=False" \
  "cluster.actor_rollout_ref.model.use_remove_padding=False" \
  "cluster.actor_rollout_ref.model.adapter.embodiment_tag=$GROOT_EMBODIMENT_TAG" \
  "cluster.actor_rollout_ref.model.adapter.embodiment_id=$GROOT_EMBODIMENT_ID" \
  "cluster.actor_rollout_ref.model.adapter.action_dim=$ACTION_DIM" \
  "cluster.actor_rollout_ref.model.adapter.num_action_chunks=$NUM_ACTION_CHUNKS" \
  "cluster.actor_rollout_ref.model.adapter.dsrl.enabled=true" \
  "cluster.actor_rollout_ref.model.adapter.critic.pool_proj_dim=$CRITIC_POOL_PROJ_DIM" \
  "cluster.actor_rollout_ref.model.adapter.critic.layernorm=$CRITIC_LAYERNORM" \
  "cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" \
  "cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer,Siglip2EncoderLayer,BasicTransformerBlock,MultiEmbodimentActionEncoder,CategorySpecificMLP]" \
  "cluster.actor_rollout_ref.actor.optim.lr=$NOISE_ACTOR_LR" \
  "cluster.actor_rollout_ref.actor.optim.warmup_style=constant" \
  "cluster.actor_rollout_ref.actor.mini_batch_size=$MINI_BATCH_SIZE" \
  "cluster.actor_rollout_ref.actor.micro_batch_size=$MICRO_BATCH_SIZE" \
  "cluster.actor_rollout_ref.actor.actor_update_interval=$ACTOR_UPDATE_INTERVAL" \
  "cluster.actor_rollout_ref.actor.ema_decay=$EMA_DECAY" \
  "cluster.actor_rollout_ref.actor.sac.auto_entropy=$AUTO_ENTROPY" \
  "cluster.actor_rollout_ref.actor.sac.initial_alpha=$INITIAL_ALPHA" \
  "cluster.actor_rollout_ref.actor.sac.alpha_type=$ALPHA_TYPE" \
  "cluster.actor_rollout_ref.actor.sac.target_entropy=$TARGET_ENTROPY" \
  "cluster.actor_rollout_ref.actor.sac.backup_entropy=$BACKUP_ENTROPY" \
  "cluster.actor_rollout_ref.actor.critic.lr=$CRITIC_LR" \
  "cluster.actor_rollout_ref.actor.critic.tau=$CRITIC_TAU" \
  "cluster.actor_rollout_ref.actor.critic.warmup_steps=$CRITIC_WARMUP_STEPS" \
  "cluster.actor_rollout_ref.actor.replay.actor_positive_sample_ratio=$ACTOR_POSITIVE_SAMPLE_RATIO" \
  "cluster.actor_rollout_ref.actor.replay.save_dir=$REPLAY_POOL_DIR" \
  "cluster.actor_rollout_ref.actor.replay.online_single_size=20000" \
  "cluster.actor_rollout_ref.rollout.name=hf" \
  "cluster.actor_rollout_ref.rollout.output_critic_value=false" \
  "cluster.actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "cluster.env.env_loop.pipeline_stage_num=$NUM_STAGE" \
  "cluster.env.env_loop.max_interactions=$MAX_INTERACTIONS" \
  "cluster.env.env_worker.auto_reset=true" \
  "cluster.env.env_worker.num_envs=$NUM_ENV" \
  "cluster.env.env_worker.simulator_start_timeout_s=600" \
  "cluster.env.env_worker.simulator.simulator_type=arena" \
  "${EXTRA_OVERRIDES[@]}" \
  "cluster.env.env_worker.modes=[train]" \
  "cluster.env.env_worker.teleop.enable=false" \
  "cluster.env.env_worker.recorder.enable=true" \
  "cluster.env.env_worker.recorder.recorders=[video]" \
  "cluster.env.env_worker.recorder.video.root=$OUTPUT_ROOT/videos" \
  "cluster.resource.env.nnodes=$NUM_NODES" \
  "cluster.resource.env.gpus_per_node=$NUM_ENV_GPUS" \
  "cluster.resource.env.workers_per_node=$NUM_ENV_GPUS" \
  "cluster.resource.model.nnodes=$NUM_NODES" \
  "cluster.resource.model.gpus_per_node=$NUM_MODEL_GPUS" \
  "cluster.resource.model.workers_per_node=$NUM_MODEL_GPUS" \
  "cluster.checkpoint.resume_mode=${RESUME_MODE:-disable}" \
  "cluster.checkpoint.resume_from_path=${RESUME_FROM_PATH:-null}" \
  "cluster.checkpoint.default_local_dir=$OUTPUT_ROOT/checkpoints" \
  "trainer.project_name=$PROJECT_NAME" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.logger=$TRAINER_LOGGER" \
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
  "trainer.rollout_interval=$ROLLOUT_INTERVAL" \
  "trainer.warm_rollout_steps=$WARM_ROLLOUT_STEPS" \
  "trainer.save_freq=$SAVE_FREQ" \
  "trainer.test_freq=$TEST_FREQ" \
  "trainer.eval_episodes=$EVAL_EPISODES" \
  "trainer.val_before_train=$VAL_BEFORE_TRAIN" \
  "trainer.val_only=False" \
  "trainer.episodic_replay=$EPISODIC_REPLAY" \
  "trainer.episodic_max_open_len=$EPISODIC_MAX_OPEN_LEN" \
  "$@"
