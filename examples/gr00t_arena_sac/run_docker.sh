#!/usr/bin/env bash
#
# GR00T Arena container launcher for verl-vla.
#
# It (re)creates the GR00T container, mounts the right host dirs, then runs an
# inner eval / sac script inside it. The inner script is selected with
# INNER_SCRIPT (relative to the repo root as seen inside the container).
#
#   isaaclab_arena:cuda_gr00t_gn16  (non-root; host-matching user via the image
#   entrypoint, same as the Arena docker/run_docker.sh)
#       Has the GR00T / Eagle / /opt/groot_deps stack. Mounts:
#         host verl-vla repo  -> /eval
#         host checkpoint dir -> /models            (MODELS_HOST)
#         host IsaacLab-Arena -> /workspaces/isaaclab_arena   (ARENA_HOST)
#         host libero_in_lab  -> /libero_in_lab      (LIBERO_IN_LAB_HOST, optional)
#
# See README.md in this folder for the full path / variable reference and
# copy-paste command recipes.
#
# ─────────────────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   # GR00T GR1 fridge eval (defaults):
#   examples/gr00t_arena_sac/run_docker.sh
#
#   # GR00T LIBERO spatial task 3 eval:
#   INNER_SCRIPT=examples/gr00t_arena_sac/run_gr00t_arena_eval.sh ARENA_TASK=libero \
#     GROOT_MODEL_PATH=/models/checkpoint-10000 \
#     examples/gr00t_arena_sac/run_docker.sh
#
#   # GR00T GR1 SAC train:
#   INNER_SCRIPT=examples/gr00t_arena_sac/run_gr00t_arena_sac.sh ARENA_TASK=gr1 \
#     GROOT_MODEL_PATH=/models/checkpoint-10000 \
#     OUTPUT_ROOT=/eval/outputs/arena_gr00t_gr1_sac \
#     examples/gr00t_arena_sac/run_docker.sh
#
#   # Just (re)start the container / drop into a shell:
#   examples/gr00t_arena_sac/run_docker.sh --shell
#   examples/gr00t_arena_sac/run_docker.sh --no-run
#
# ─────────────────────────────────────────────────────────────────────────────
# Common overrides (env vars) — see README.md for the full table
# ─────────────────────────────────────────────────────────────────────────────
#   IMAGE              docker image                (default: isaaclab_arena:cuda_gr00t_gn16)
#   CONTAINER_NAME     container name              (default: isaaclab_arena-cuda_gr00t_gn16)
#   RECREATE=1         force remove + recreate the container
#   DIRECT_RUN=1       run the inner script in a one-shot container (default for root)
#   INNER_SCRIPT       inner script (relative to the repo inside the container)
#                      (EVAL_SCRIPT: deprecated alias, still honoured)
#   MAX_EPISODES       episodes to evaluate        (default 10; ignored by train)
#   ARENA_TASK         gr1 | libero                (forwarded to gr00t inner scripts)
#   OUTPUT_ROOT        eval/train output root inside the container
#   MODELS_HOST        host checkpoint parent      -> /models
#   GROOT_MODEL_PATH   checkpoint path inside the container
#   ARENA_HOST         host IsaacLab-Arena checkout -> /workspaces/isaaclab_arena
#   LIBERO_IN_LAB_HOST host libero_in_lab checkout -> /libero_in_lab
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="run"
case "${1:-}" in
  --shell)  MODE="shell" ;;
  --no-run) MODE="none" ;;
  "")       MODE="run" ;;
  *) echo "Unknown option: $1" >&2; exit 1 ;;
esac

log() { echo -e "\033[1;35m[run_docker:gr00t]\033[0m $*"; }

RECREATE="${RECREATE:-0}"
MAX_EPISODES="${MAX_EPISODES:-10}"
if [[ "$(id -u)" == "0" ]]; then
  DIRECT_RUN="${DIRECT_RUN:-1}"
else
  DIRECT_RUN="${DIRECT_RUN:-0}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# GR00T backend configuration.
# ─────────────────────────────────────────────────────────────────────────────
DOCKER_MOUNT_ARGS=()
DOCKER_ENV_ARGS=(-e PYTHONDONTWRITEBYTECODE=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y)

# Non-root: like the Arena docker/run_docker.sh, work runs as a container user that
# matches the host uid/gid (with passwordless sudo), so outputs are written with host
# ownership instead of root. We create that user with a small root bootstrap after
# the container starts — the image entrypoint's own useradd does not create a home
# dir for out-of-range (domain/LDAP) uids and would abort. The container is kept
# alive with `sleep infinity`; all real work is `docker exec`ed as this user.
RUN_UID="$(id -u)"; RUN_GID="$(id -g)"
RUN_USER="$(id -un)"; RUN_GROUP="$(id -gn)"

IMAGE="${IMAGE:-isaaclab_arena:cuda_gr00t_gn16}"
CONTAINER_NAME="${CONTAINER_NAME:-isaaclab_arena-cuda_gr00t_gn16}"
WORKDIR="${WORKDIR:-/eval}"
# INNER_SCRIPT selects the eval / train script run inside the container.
# EVAL_SCRIPT is still honoured as a deprecated alias for backward compatibility.
INNER_SCRIPT="${INNER_SCRIPT:-${EVAL_SCRIPT:-examples/gr00t_arena_sac/run_gr00t_arena_eval.sh}}"

# Checkpoint parent on the host -> /models. Put your GR00T HF-format export dir(s)
# under <repo>/checkpoints/ (e.g. <repo>/checkpoints/checkpoint-10000), or override.
MODELS_HOST="${MODELS_HOST:-$HOST_REPO/checkpoints}"
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/models/checkpoint-10000}"
# IsaacLab-Arena checkout -> /workspaces/isaaclab_arena. Defaults to a checkout next
# to this repo; its env code / patches win over the image's (often stale) baked copy.
ARENA_HOST="${ARENA_HOST:-$HOST_REPO/IsaacLab-Arena}"
ARENA_WORKDIR="${ARENA_WORKDIR:-/workspaces/isaaclab_arena}"
# The bind-mounted Arena provides the editable isaaclab install; point the image at it.
DOCKER_ENV_ARGS+=(-e "ISAACLAB_PATH=$ARENA_WORKDIR/submodules/IsaacLab")
# LIBERO USD assets (required for the Arena LIBERO environment; harmless for GR1).
LIBERO_IN_LAB_HOST="${LIBERO_IN_LAB_HOST:-$HOST_REPO/libero_in_lab}"
# Container mount dest == the env var name the inner scripts read (LIBERO_IN_LAB_ROOT).
LIBERO_IN_LAB_ROOT="${LIBERO_IN_LAB_ROOT:-/libero_in_lab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/outputs/arena_gr00t_gr1_eval}"
# Ray embeds the session name below this path in an AF_UNIX socket; keeping the
# root short avoids Linux's 107-byte socket-path limit.
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"

# Inner train/eval scripts read their knobs from env, but docker exec/run do not
# inherit the host env — only what we -e forward. The bare `-e NAME` form (no
# value) forwards the process-env value and is silently dropped by docker when
# the var is unset, so unset knobs fall through to inner-script defaults — no
# guard needed. Caller pass-throughs arrive already exported (prefix assignment);
# this launcher's own derived defaults (paths etc.) are computed above as plain
# shell vars, so export them here to carry our value through the same mechanism.
GR00T_COMPAT_PATCHES="${GR00T_COMPAT_PATCHES-all}"
export GROOT_MODEL_PATH MAX_EPISODES OUTPUT_ROOT RAY_TMPDIR \
  LIBERO_IN_LAB_ROOT GR00T_COMPAT_PATCHES
INNER_FORWARD_VARS=(
  # Caller pass-throughs (inner scripts default anything left unset).
  EMA_DECAY FREEZE_ACTION_IO CRITIC_POOL_PROJ_DIM CRITIC_LAYERNORM
  FLOW_SDE_ENABLE ACTOR_POSITIVE_SAMPLE_RATIO
  FLOW_SDE_NOISE_LEVEL FLOW_SDE_ROLLOUT_NOISE_SCALE FLOW_SDE_TRAIN_NOISE_SCALE FLOW_SDE_INITIAL_BETA
  CRITIC_TAU INITIAL_ALPHA ALPHA_TYPE AUTO_ENTROPY CRITIC_WARMUP_STEPS
  ACTOR_UPDATE_INTERVAL MINI_BATCH_SIZE MICRO_BATCH_SIZE
  MAX_INTERACTIONS TOTAL_TRAINING_STEPS ROLLOUT_INTERVAL WARM_ROLLOUT_STEPS
  SAVE_FREQ TEST_FREQ VAL_BEFORE_TRAIN EVAL_EPISODES RESUME_MODE RESUME_FROM_PATH
  TRAINER_LOGGER PROJECT_NAME EXPERIMENT_NAME REPLAY_POOL_DIR
  NUM_NODES NUM_ENV_GPUS NUM_MODEL_GPUS NUM_ENV NUM_STAGE
  NUM_ACTION_CHUNKS GROOT_EMBODIMENT_TAG GROOT_EMBODIMENT_ID ACTION_DIM
  WANDB_MODE WANDB_ENTITY WANDB_PROJECT WANDB_BASE_URL WANDB_DIR
  ARENA_TASK TASK_SUITE TASK_ID
  # Derived defaults exported by this launcher (see export above).
  GROOT_MODEL_PATH MAX_EPISODES OUTPUT_ROOT RAY_TMPDIR
  LIBERO_IN_LAB_ROOT GR00T_COMPAT_PATCHES
)
INNER_ENV_ARGS=()
for _v in "${INNER_FORWARD_VARS[@]}"; do
  INNER_ENV_ARGS+=(-e "$_v")
done

mkdir -p "$HOST_REPO/outputs"
chmod 777 "$HOST_REPO/outputs" 2>/dev/null || true
DOCKER_MOUNT_ARGS+=(-v "$HOST_REPO:$WORKDIR")

if [[ -d "$ARENA_HOST" ]]; then
  log "Mounting Arena '$ARENA_HOST' -> $ARENA_WORKDIR"
  DOCKER_MOUNT_ARGS+=(-v "$ARENA_HOST:$ARENA_WORKDIR")
else
  log "ERROR: ARENA_HOST='$ARENA_HOST' does not exist."
  log "  Set ARENA_HOST to your local IsaacLab-Arena checkout (needed for wrist cam / env code)."
  exit 1
fi

if [[ -d "$MODELS_HOST" ]]; then
  log "Mounting models '$MODELS_HOST' -> /models"
  DOCKER_MOUNT_ARGS+=(-v "$MODELS_HOST:/models")
else
  log "WARNING: MODELS_HOST='$MODELS_HOST' does not exist; /models will be empty."
fi

if [[ -d "$LIBERO_IN_LAB_HOST" ]]; then
  log "Mounting libero_in_lab '$LIBERO_IN_LAB_HOST' -> $LIBERO_IN_LAB_ROOT"
  DOCKER_MOUNT_ARGS+=(-v "$LIBERO_IN_LAB_HOST:$LIBERO_IN_LAB_ROOT")
else
  log "WARNING: LIBERO_IN_LAB_HOST='$LIBERO_IN_LAB_HOST' missing; Arena LIBERO evals need it."
fi

# Do NOT bind-mount host /tmp -> /tmp: host files owned by another uid cause
# "Permission denied" inside the container. Use the image's own /tmp.

# Root environments (including many managed GPU jobs) can run the workload
# directly. This avoids docker-exec/privileged-container restrictions imposed by
# nested Docker services while preserving the long-lived non-root mode below.
if [[ "$MODE" == "run" && "$DIRECT_RUN" == "1" ]]; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  log "Running $INNER_SCRIPT in a one-shot container"
  INNER_COMMAND="bash '$INNER_SCRIPT'"
  docker run --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host --network=host \
    "${DOCKER_ENV_ARGS[@]}" \
    "${DOCKER_MOUNT_ARGS[@]}" \
    "${INNER_ENV_ARGS[@]}" \
    -w "$WORKDIR" \
    --entrypoint bash \
    "$IMAGE" \
    -lc "$INNER_COMMAND"
  HOST_OUTPUT="${OUTPUT_ROOT/#$WORKDIR/$HOST_REPO}"
  log "Outputs on host: $HOST_OUTPUT"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# (Re)create the container with all GPUs.
# Bypass the image entrypoint and keep a long-lived bash so we can docker exec
# repeatedly.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$RECREATE" == "1" ]]; then
  log "Removing existing container '$CONTAINER_NAME' (RECREATE=1)"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# Recreate if a running container is missing any mount we requested.
if [[ "$RECREATE" != "1" ]] && docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  mounts="$(docker inspect -f '{{range .Mounts}}{{println .Source .Destination}}{{end}}' "$CONTAINER_NAME" 2>/dev/null || true)"
  need_recreate=0
  for i in "${!DOCKER_MOUNT_ARGS[@]}"; do
    [[ "${DOCKER_MOUNT_ARGS[$i]}" == "-v" ]] || continue
    spec="${DOCKER_MOUNT_ARGS[$((i + 1))]}"          # host:dest[:ro]
    host="${spec%%:*}"
    rest="${spec#*:}"
    dest="${rest%%:*}"
    if ! grep -qx "$host $dest" <<<"$mounts"; then
      log "Running container is missing mount ($host -> $dest); recreating"
      need_recreate=1
      break
    fi
  done
  if [[ "$need_recreate" == "1" ]]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  log "Starting container '$CONTAINER_NAME' from image '$IMAGE' (will run work as $RUN_USER, uid $RUN_UID)"
  # Keep-alive as a long-lived bash (bypass the entrypoint). --privileged mirrors the
  # Arena launcher (Isaac Sim device access as non-root).
  docker run -d --name "$CONTAINER_NAME" \
    --privileged \
    --gpus all \
    --ipc=host --network=host \
    --ulimit memlock=-1 --ulimit stack=-1 \
    "${DOCKER_ENV_ARGS[@]}" \
    "${DOCKER_MOUNT_ARGS[@]}" \
    --entrypoint bash \
    "$IMAGE" \
    -c 'sleep infinity' >/dev/null
else
  log "Container '$CONTAINER_NAME' already running, reusing it"
fi

# Bootstrap the host-matching non-root user (uid/gid, passwordless sudo, home) so all
# work runs as it — outputs get host ownership instead of root. Same effect as the
# Arena entrypoint, but works for out-of-range domain/LDAP uids. Idempotent.
docker exec -u 0 "$CONTAINER_NAME" bash -c "
  set -e
  getent group '$RUN_GID' >/dev/null || groupadd --gid '$RUN_GID' '$RUN_GROUP' || true
  if ! getent passwd '$RUN_UID' >/dev/null; then
    useradd --no-log-init -m -o --uid '$RUN_UID' --gid '$RUN_GID' \
      --groups sudo,isaac-sim --shell /bin/bash '$RUN_USER'
  fi
  echo '$RUN_USER ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/99-runuser
  chmod 0440 /etc/sudoers.d/99-runuser
  chown '$RUN_UID:$RUN_GID' /models /libero_in_lab 2>/dev/null || true
  # /isaac-sim is 0751 (drwxr-x--x isaac-sim). 'docker exec -u uid:gid' below drops
  # supplementary groups, so the run user is NOT in the isaac-sim group at runtime and
  # cannot read it as 'other' -> isaacsim fails to expose SimulationApp ('NoneType' is
  # not callable in AppLauncher). Grant others read+traverse on the top dir (subdirs are
  # already o+rx) so the host-uid user can load the isaacsim.simulation_app extension.
  chmod o+rx /isaac-sim 2>/dev/null || true
" >/dev/null 2>&1 || log "WARNING: non-root user bootstrap reported an error"

# Run inside the container as the host-matching user.
DOCKER_USER_ARGS=(-u "$RUN_UID:$RUN_GID" -e "HOME=/home/$RUN_USER")

# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks.
# ─────────────────────────────────────────────────────────────────────────────
log "Container user: $(docker exec "${DOCKER_USER_ARGS[@]}" "$CONTAINER_NAME" whoami)"
log "GPUs visible inside container:"
docker exec "$CONTAINER_NAME" nvidia-smi --query-gpu=index,name,memory.total --format=csv

if docker exec "$CONTAINER_NAME" test -d /opt/groot_deps; then
  log "Found /opt/groot_deps (GR00T deps OK)"
else
  log "WARNING: /opt/groot_deps missing — is IMAGE really the cuda_gr00t_gn16 build?"
fi
# Confirm the bind-mounted Arena (not the baked image copy) has the wrist-cam patch.
ARENA_ENV_FILE="$ARENA_WORKDIR/isaaclab_arena_environments/gr1_put_and_close_door_environment.py"
if docker exec "$CONTAINER_NAME" test -f "$ARENA_ENV_FILE"; then
  if docker exec "$CONTAINER_NAME" grep -q 'GR1T2WristCameraCfg' "$ARENA_ENV_FILE"; then
    log "Arena mount OK: wrist-cam patch present at $ARENA_WORKDIR"
  else
    log "WARNING: Arena at $ARENA_WORKDIR has no GR1T2WristCameraCfg — check ARENA_HOST branch"
  fi
else
  log "WARNING: Arena env file missing under $ARENA_WORKDIR"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Run inner script / shell / nothing.
# ─────────────────────────────────────────────────────────────────────────────
# Allocate a TTY only when stdin is a terminal (avoids "the input device is not a
# TTY" when this helper is driven by CI / nohup / pipes).
DOCKER_TTY_ARGS=()
if [[ -t 0 ]]; then
  DOCKER_TTY_ARGS+=(-t)
fi

case "$MODE" in
  none)
    log "Container ready. Attach with: docker exec -ti -u $RUN_UID:$RUN_GID $CONTAINER_NAME bash"
    ;;
  shell)
    log "Dropping into an interactive shell as $RUN_USER (workdir=$WORKDIR)"
    docker exec -ti "${DOCKER_USER_ARGS[@]}" -w "$WORKDIR" "$CONTAINER_NAME" bash
    ;;
  run)
    # Non-root user == host uid, so outputs under the bind-mounted repo are written
    # with correct host ownership (no post-hoc chmod needed).
    log "Running $INNER_SCRIPT (GROOT_MODEL_PATH=$GROOT_MODEL_PATH, MAX_EPISODES=$MAX_EPISODES, OUTPUT_ROOT=$OUTPUT_ROOT)"
    # Resolve the wandb API key for [console,wandb] runs: the container HOME is a fresh
    # useradd -m dir (host ~/.netrc is not mounted), so wandb would prompt for a key and
    # fail under no-tty. Prefer an already-exported WANDB_API_KEY, else pull it from the
    # host ~/.netrc (api.wandb.ai entry). Read at runtime only; never written to a file.
    if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
      WANDB_API_KEY="$(awk '{for(i=1;i<=NF;i++){if($i=="machine")m=($(i+1)=="api.wandb.ai");if(m&&$i=="password"){print $(i+1);exit}}}' "$HOME/.netrc")"
    fi
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      log "Forwarding WANDB_API_KEY into the container (len=${#WANDB_API_KEY})"
    else
      log "WARNING: no WANDB_API_KEY found (env or ~/.netrc); wandb logging will fail if TRAINER_LOGGER includes wandb"
    fi
    docker exec -i "${DOCKER_TTY_ARGS[@]}" "${DOCKER_USER_ARGS[@]}" -w "$WORKDIR" \
      "${INNER_ENV_ARGS[@]}" \
      -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
      "$CONTAINER_NAME" \
      bash "$INNER_SCRIPT"
    HOST_OUTPUT="${OUTPUT_ROOT/#$WORKDIR/$HOST_REPO}"
    log "Outputs on host: $HOST_OUTPUT"
    ;;
esac
