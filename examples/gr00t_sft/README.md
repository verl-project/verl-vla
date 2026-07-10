# GR00T N1.6: LIBERO Spatial SFT and Validation

verl-vla does not vendor the GR00T source code or declare GR00T as a default
dependency. Only users who need GR00T should install the pinned upstream source
package:

```text
GR00T commit: e29d8fc50b0e4745120ae3fb72447986fe638aa6
Community validation checkpoint revision: d690a226ad06e81736786f56cf879d2ed1dd3f0f
```

Each step below provides separate commands for a native environment and Docker.
All examples assume that the current directory is the root of the verl-vla
repository.

## 0. Configure paths and create directories

### Native environment

```bash
export DATA_ROOT="${DATA_ROOT:-$PWD/.data/gr00t_sft}"

mkdir -p \
  "$DATA_ROOT/datasets/libero_spatial_image" \
  "$DATA_ROOT/models" \
  "$DATA_ROOT/output" \
  "$DATA_ROOT/huggingface" \
  "$DATA_ROOT/raytmp" \
  "$DATA_ROOT/tmp"
```

### Docker

```bash
export DATA_HOST="${DATA_HOST:-$PWD/.data/gr00t_sft}"
export REPO_HOST="${REPO_HOST:-$PWD}"
export IMAGE="${IMAGE:-verl-vla-gr00t:n1.6}"

mkdir -p \
  "$DATA_HOST/datasets/libero_spatial_image" \
  "$DATA_HOST/models" \
  "$DATA_HOST/output" \
  "$DATA_HOST/huggingface" \
  "$DATA_HOST/raytmp" \
  "$DATA_HOST/tmp"
```

Override `DATA_ROOT` or `DATA_HOST` before running these commands if you want
to place datasets, caches, and checkpoints on a larger disk.

The Docker commands below bind-mount `REPO_HOST` into `/workspace/verl-vla` and
set `PYTHONPATH=/workspace/verl-vla/src`. This keeps example scripts, configs,
and Python source code live during development without rebuilding the image for
every code change. Rebuild the image only when dependencies or the Dockerfile
change.

The resulting directory layout should look like this:

```text
verl-vla-data/
├── datasets/libero_spatial_image/
│   ├── data/
│   ├── meta/
│   ├── videos/
│   └── norm_stats.json
├── models/
├── output/
└── huggingface/
```

## 1. Install the runtime environment

`--no-deps` only prevents pip from resolving the dependencies declared by
GR00T. It does not mean that GR00T has no runtime dependencies. A native
environment must provide dependency versions compatible with the Dockerfile.

### Native environment

```bash
python -m pip install --no-deps \
  "gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T.git@e29d8fc50b0e4745120ae3fb72447986fe638aa6"
python -m pip install --no-deps .
python scripts/check_gr00t_n1d6_install.py
```

### Docker

The Docker image installs the pinned GR00T commit, adds the Eagle assets missing
from the upstream wheel, and installs LIBERO, MuJoCo, and headless rendering
dependencies.

```bash
docker build \
  -f docker/Dockerfile.gr00t \
  -t "$IMAGE" \
  .
```

The image build runs `scripts/check_gr00t_n1d6_install.py`. There is no need to
install or verify GR00T again on the host.

## 2. Download the LIBERO Spatial training dataset

The dataset is pinned to the verified LeRobot v3 revision.

### Native environment

```bash
hf download lerobot/libero_spatial_image \
  --repo-type dataset \
  --revision d86c0b94922572b3b657e1d1a3d01f0952ddeb46 \
  --local-dir "$DATA_ROOT/datasets/libero_spatial_image"
```

### Docker

```bash
docker run --rm \
  -v "$DATA_HOST:/data" \
  -e HF_HOME=/data/huggingface \
  "$IMAGE" \
  hf download lerobot/libero_spatial_image \
    --repo-type dataset \
    --revision d86c0b94922572b3b657e1d1a3d01f0952ddeb46 \
    --local-dir /data/datasets/libero_spatial_image
```

## 3. Compute SFT normalization statistics

GR00T SFT requires min, max, mean, std, q01, and q99 statistics for state and
action values. This step creates
`datasets/libero_spatial_image/norm_stats.json`.

The dataset stores gripper actions using LIBERO simulator semantics:
`-1=open, +1=close`. The generic statistics script preserves the raw dataset
semantics. The GR00T policy adapter converts both training samples and flat
statistics to the official GR00T semantics, `1=open, 0=close`. Native nested
`libero_panda` statistics from GR00T are already converted and are therefore
left unchanged.

### Native environment

```bash
python scripts/compute_norm_stats.py \
  --repo-id lerobot/libero_spatial_image \
  --root "$DATA_ROOT/datasets/libero_spatial_image" \
  --output-path "$DATA_ROOT/datasets/libero_spatial_image/norm_stats.json" \
  --include-min-max \
  --batch-size 32 \
  --num-workers 8
```

### Docker

This step is CPU-only and does not require `--gpus`. Increase `/dev/shm` for
the multi-worker PyTorch dataloader.

```bash
docker run --rm \
  --shm-size 64g \
  -v "$DATA_HOST:/data" \
  -v "$REPO_HOST:/workspace/verl-vla:ro" \
  -e HF_HOME=/data/huggingface \
  -e PYTHONPATH=/workspace/verl-vla/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$IMAGE" \
  python scripts/compute_norm_stats.py \
    --repo-id lerobot/libero_spatial_image \
    --root /data/datasets/libero_spatial_image \
    --output-path /data/datasets/libero_spatial_image/norm_stats.json \
    --include-min-max \
    --batch-size 32 \
    --num-workers 8
```

## 4. Launch multi-GPU SFT

`SFT_BATCH_SIZE` is the global dataloader batch size, while
`MICRO_BATCH_SIZE` is the per-rank batch size for each forward pass. The
configuration below is the exact setup that achieved a 90% success rate over 50
LIBERO Spatial validation trials: 8 GPUs, global/mini batch size 64, micro batch
size 8, 13 epochs, FP32 master weights, and BF16 mixed precision.

With the pinned dataset and this batch size, each epoch contains 827 optimizer
steps, so 13 epochs covers `global_step_10000`. The reported validation uses
`global_step_10000`. No generic trainer or worker code is modified for GR00T.

### Native environment

```bash
export NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
export GLOBAL_BATCH_SIZE=64
export MICRO_BATCH_SIZE=8

MODEL_PATH=nvidia/GR00T-N1.6-3B \
NORM_STATS_PATH="$DATA_ROOT/datasets/libero_spatial_image/norm_stats.json" \
SFT_ROOT="$DATA_ROOT/datasets/libero_spatial_image" \
OUTPUT_DIR="$DATA_ROOT/output/gr00t_n1d6_libero_spatial_sft" \
NUM_GPUS="$NUM_GPUS" \
SFT_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MINI_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
SFT_NUM_WORKERS=8 \
TOTAL_EPOCHS=13 \
LR=1e-4 \
WEIGHT_DECAY=1e-5 \
SAVE_FREQ=500 \
MAX_ACTOR_CKPT_TO_KEEP=3 \
RESUME_DATALOADER_STATE=true \
bash examples/gr00t_sft/run_gr00t_lerobot_sft.sh
```

`TOTAL_EPOCHS=13` produces 10,751 steps. When
`$DATA_ROOT/output/gr00t_n1d6_libero_spatial_sft/latest_checkpointed_iteration.txt`
becomes `10000`, the step 10000 FSDP and Hugging Face checkpoints have been
fully written and training can be stopped. All validation commands below use
step 10000.

### Docker

```bash
export NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
export GLOBAL_BATCH_SIZE=64
export MICRO_BATCH_SIZE=8

docker run -d \
  --name gr00t_n1d6_sft \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$DATA_HOST:/data" \
  -v "$REPO_HOST:/workspace/verl-vla:ro" \
  -e HF_HOME=/data/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e PYTHONPATH=/workspace/verl-vla/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RAY_TMPDIR=/data/raytmp \
  -e TMPDIR=/data/tmp \
  -e NO_ALBUMENTATIONS_UPDATE=1 \
  -e MODEL_PATH=nvidia/GR00T-N1.6-3B \
  -e SFT_REPO_ID=lerobot/libero_spatial_image \
  -e SFT_ROOT=/data/datasets/libero_spatial_image \
  -e NORM_STATS_PATH=/data/datasets/libero_spatial_image/norm_stats.json \
  -e OUTPUT_DIR=/data/output/gr00t_n1d6_libero_spatial_sft \
  -e NUM_GPUS="$NUM_GPUS" \
  -e NUM_NODES=1 \
  -e SFT_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
  -e MINI_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
  -e MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
  -e SFT_NUM_WORKERS=8 \
  -e TOTAL_EPOCHS=13 \
  -e LR=1e-4 \
  -e WEIGHT_DECAY=1e-5 \
  -e SAVE_FREQ=500 \
  -e MAX_ACTOR_CKPT_TO_KEEP=3 \
  -e RESUME_DATALOADER_STATE=true \
  "$IMAGE" \
  bash -lc 'nvidia-smi && bash examples/gr00t_sft/run_gr00t_lerobot_sft.sh'

docker logs -f gr00t_n1d6_sft
```

Docker training continues to 10,751 steps. To reproduce the step 10000
checkpoint exactly, run the following monitor in another terminal. The marker
is updated only after the FSDP shards, Hugging Face weights, and dataloader state
have all been written:

```bash
TARGET_STEP=10000
MARKER="$DATA_HOST/output/gr00t_n1d6_libero_spatial_sft/latest_checkpointed_iteration.txt"

while true; do
  SAVED_STEP=$(cat "$MARKER" 2>/dev/null || echo 0)
  if (( SAVED_STEP >= TARGET_STEP )); then
    docker stop -t 30 gr00t_n1d6_sft
    break
  fi
  sleep 30
done
```

## 5. Use the directly exported Hugging Face checkpoint

The GR00T SFT configuration saves both the FSDP shards required to resume
training and a complete Hugging Face checkpoint. No additional merge step is
required. The directory includes the model weights, GR00T processor
configuration, normalization statistics, and embodiment metadata.

### Native environment

```text
$DATA_ROOT/output/gr00t_n1d6_libero_spatial_sft/global_step_10000/actor/huggingface
```

### Docker

```text
/data/output/gr00t_n1d6_libero_spatial_sft/global_step_10000/actor/huggingface
```

## 6. Run LIBERO Spatial validation rollouts

The commands below reproduce the protocol used for the reported 90% success
rate: 10 tasks with 5 trials per task, for a total of 50 rollouts. Two
environment workers run 25 environments each. `MAX_INTERACTIONS=90` corresponds
to 720 simulator steps with an action chunk size of 8.

### Native environment

```bash
MODEL_PATH="$DATA_ROOT/output/gr00t_n1d6_libero_spatial_sft/global_step_10000/actor/huggingface" \
NORM_STATS_PATH=null \
MODEL_GPUS=1 \
ENV_WORKERS=2 \
NUM_ENVS=25 \
TASK_IDS=null \
TRIALS_PER_TASK=5 \
MAX_EPISODE_STEPS=720 \
ACTION_CHUNK_SIZE=8 \
MAX_INTERACTIONS=90 \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
bash examples/gr00t_sft/run_gr00t_libero_eval.sh
```

### Docker

Docker requires the NVIDIA Container Toolkit. Replace `device=7` with the GPU
to use, or use `--gpus all`.

```bash
docker run --rm \
  --gpus '"device=7"' \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$DATA_HOST:/data" \
  -v "$REPO_HOST:/workspace/verl-vla:ro" \
  -v "$DATA_HOST/libero_cache/assets:/root/.cache/libero/assets:ro" \
  -e HF_HOME=/data/huggingface \
  -e PYTHONPATH=/workspace/verl-vla/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RAY_TMPDIR=/data/raytmp_eval \
  -e TMPDIR=/data/tmp_eval \
  -e LIBERO_CONFIG_PATH=/data/libero_config \
  -e MUJOCO_GL=osmesa \
  -e PYOPENGL_PLATFORM=osmesa \
  -e MODEL_PATH=/data/output/gr00t_n1d6_libero_spatial_sft/global_step_10000/actor/huggingface \
  -e NORM_STATS_PATH=null \
  -e MODEL_GPUS=1 \
  -e ENV_WORKERS=2 \
  -e NUM_ENVS=25 \
  -e TASK_IDS=null \
  -e TRIALS_PER_TASK=5 \
  -e MAX_EPISODE_STEPS=720 \
  -e ACTION_CHUNK_SIZE=8 \
  -e MAX_INTERACTIONS=90 \
  "$IMAGE" \
  bash -lc 'bash examples/gr00t_sft/run_gr00t_libero_eval.sh'
```

## 7. Validate the community LIBERO checkpoint

`validate_community_checkpoint.sh` downloads and validates the following
checkpoint by default:

```text
0xAnkitSingh/GR00T-N1.6-LIBERO
revision d690a226ad06e81736786f56cf879d2ed1dd3f0f
```

The default configuration evaluates 3 trials of `libero_spatial` task 0. This
is a community checkpoint, not an official NVIDIA LIBERO checkpoint.

### Native environment

```bash
DATA_ROOT="$DATA_ROOT" \
MODEL_GPUS=1 \
ENV_WORKERS=1 \
NUM_ENVS=3 \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
bash examples/gr00t_sft/validate_community_checkpoint.sh
```

### Docker

```bash
docker run --rm \
  --gpus '"device=7"' \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$DATA_HOST:/data" \
  -v "$REPO_HOST:/workspace/verl-vla:ro" \
  -v "$DATA_HOST/libero_cache/assets:/root/.cache/libero/assets:ro" \
  -e HF_HOME=/data/huggingface \
  -e PYTHONPATH=/workspace/verl-vla/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RAY_TMPDIR=/data/raytmp_community_eval \
  -e TMPDIR=/data/tmp_community_eval \
  -e LIBERO_CONFIG_PATH=/data/libero_config \
  -e MUJOCO_GL=osmesa \
  -e PYOPENGL_PLATFORM=osmesa \
  -e DATA_ROOT=/data \
  -e MODEL_GPUS=1 \
  -e ENV_WORKERS=1 \
  -e NUM_ENVS=3 \
  "$IMAGE" \
  bash -lc 'bash examples/gr00t_sft/validate_community_checkpoint.sh'
```

A successful validation prints the following metrics:

```text
val/trajectory_count
val/success_trajectory_count
val/failed_trajectory_count
val/trajectory_success_rate
val/per_task_success_rate/task_0
```

This integration covers GR00T policy SFT, upstream-compatible checkpoint export,
and validation-only LIBERO rollouts. GR00T SAC, RECAP, and Flow-SDE training are
outside the scope of this integration.
