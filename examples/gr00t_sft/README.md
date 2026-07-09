# GR00T N1.6：LIBERO Spatial SFT 与验证

verl-vla 不内置 GR00T 源码，也不把 GR00T 声明为默认依赖。只有需要使用
GR00T 的用户才安装固定的上游 source package：

```text
GR00T commit: e29d8fc50b0e4745120ae3fb72447986fe638aa6
Community validation checkpoint revision: d690a226ad06e81736786f56cf879d2ed1dd3f0f
```

下文每一步都分别提供原生环境和 Docker 命令。示例假设当前目录是
verl-vla 仓库根目录。

## 0. 设置路径并创建目录

### 原生环境

```bash
export DATA_ROOT=/raid/lancel/Projects/verl-vla-data

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
export DATA_HOST=/raid/lancel/Projects/verl-vla-data
export IMAGE=verl-vla-gr00t:n1.6

mkdir -p \
  "$DATA_HOST/datasets/libero_spatial_image" \
  "$DATA_HOST/models" \
  "$DATA_HOST/output" \
  "$DATA_HOST/huggingface" \
  "$DATA_HOST/raytmp" \
  "$DATA_HOST/tmp"
```

最终数据目录类似：

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

## 1. 安装运行环境

`--no-deps` 只表示 pip 不解析 GR00T 自身声明的依赖，并不表示 GR00T
不需要运行时依赖。原生环境需要自行提供与 Dockerfile 对齐的依赖版本。

### 原生环境

```bash
python -m pip install --no-deps \
  "gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T.git@e29d8fc50b0e4745120ae3fb72447986fe638aa6"
python -m pip install --no-deps .
python scripts/check_gr00t_n1d6_install.py
```

### Docker

Docker 镜像会安装固定 GR00T commit、补齐上游 wheel 缺失的 Eagle assets，
并安装 LIBERO、MuJoCo 和 headless rendering 依赖。

```bash
docker build \
  -f docker/Dockerfile.gr00t \
  -t "$IMAGE" \
  .
```

构建结束时已经执行 `scripts/check_gr00t_n1d6_install.py`，不需要在宿主机
再次安装或检查 GR00T。

## 2. 下载 LIBERO Spatial 训练数据

数据集固定到已验证的 LeRobot v3 revision。

### 原生环境

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

## 3. 计算 SFT normalization statistics

GR00T SFT 需要 state/action 的 min、max、mean、std、q01 和 q99。生成文件
为 `datasets/libero_spatial_image/norm_stats.json`。

### 原生环境

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

这一步只使用 CPU，不需要 `--gpus`。

```bash
docker run --rm \
  -v "$DATA_HOST:/data" \
  -e HF_HOME=/data/huggingface \
  "$IMAGE" \
  python scripts/compute_norm_stats.py \
    --repo-id lerobot/libero_spatial_image \
    --root /data/datasets/libero_spatial_image \
    --output-path /data/datasets/libero_spatial_image/norm_stats.json \
    --include-min-max \
    --batch-size 32 \
    --num-workers 8
```

## 4. 启动多 GPU SFT

`SFT_BATCH_SIZE` 是全局 dataloader batch；`MICRO_BATCH_SIZE` 是每个 rank
每次 forward 的 batch。下面的配置使用本机全部 GPU，每张卡一个样本。

### 原生环境

```bash
export NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
export PER_GPU_BATCH=1
export GLOBAL_BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))

MODEL_PATH=nvidia/GR00T-N1.6-3B \
NORM_STATS_PATH="$DATA_ROOT/datasets/libero_spatial_image/norm_stats.json" \
SFT_ROOT="$DATA_ROOT/datasets/libero_spatial_image" \
OUTPUT_DIR="$DATA_ROOT/output/gr00t_n1d6_libero_spatial_sft" \
NUM_GPUS="$NUM_GPUS" \
SFT_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MINI_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MICRO_BATCH_SIZE="$PER_GPU_BATCH" \
SFT_NUM_WORKERS=8 \
TOTAL_EPOCHS=4 \
bash examples/gr00t_sft/run_gr00t_lerobot_sft.sh
```

### Docker

```bash
export NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
export PER_GPU_BATCH=1
export GLOBAL_BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))

docker run -d \
  --name gr00t_n1d6_sft \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$DATA_HOST:/data" \
  -e HF_HOME=/data/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
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
  -e MICRO_BATCH_SIZE="$PER_GPU_BATCH" \
  -e SFT_NUM_WORKERS=8 \
  -e TOTAL_EPOCHS=4 \
  -e SAVE_FREQ=500 \
  -e MAX_ACTOR_CKPT_TO_KEEP=3 \
  "$IMAGE" \
  bash -lc 'nvidia-smi && bash examples/gr00t_sft/run_gr00t_lerobot_sft.sh'

docker logs -f gr00t_n1d6_sft
```

## 5. 合并 FSDP checkpoint

合并结果保持上游 GR00T 的 state-dict key，并包含 processor config、statistics
和 embodiment metadata。将 `global_step_1000` 替换为实际 checkpoint。

### 原生环境

```bash
python scripts/merge_gr00t_fsdp_checkpoint.py \
  --local-dir "$DATA_ROOT/output/gr00t_n1d6_libero_spatial_sft/global_step_1000/actor" \
  --base-model nvidia/GR00T-N1.6-3B \
  --norm-stats "$DATA_ROOT/datasets/libero_spatial_image/norm_stats.json" \
  --target-dir "$DATA_ROOT/models/gr00t_n1d6_libero_spatial_step1000" \
  --verify
```

### Docker

```bash
docker run --rm \
  -v "$DATA_HOST:/data" \
  -e HF_HOME=/data/huggingface \
  "$IMAGE" \
  python scripts/merge_gr00t_fsdp_checkpoint.py \
    --local-dir /data/output/gr00t_n1d6_libero_spatial_sft/global_step_1000/actor \
    --base-model nvidia/GR00T-N1.6-3B \
    --norm-stats /data/datasets/libero_spatial_image/norm_stats.json \
    --target-dir /data/models/gr00t_n1d6_libero_spatial_step1000 \
    --verify
```

## 6. 运行 LIBERO Spatial validation rollout

`run_gr00t_libero_eval.sh` 默认评测 10 个任务、每个任务 10 个 trial。快速验证
可以设置 `TASK_IDS='[0]'` 和 `TRIALS_PER_TASK=3`。

### 原生环境

```bash
MODEL_PATH="$DATA_ROOT/models/gr00t_n1d6_libero_spatial_step1000" \
NORM_STATS_PATH=null \
MODEL_GPUS=1 \
ENV_WORKERS=1 \
NUM_ENVS=3 \
TASK_IDS='[0]' \
TRIALS_PER_TASK=3 \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
bash examples/gr00t_sft/run_gr00t_libero_eval.sh
```

### Docker

Docker 中需要 NVIDIA Container Toolkit。`device=7` 可以替换为希望使用的 GPU，
也可以改成 `--gpus all`。

```bash
docker run --rm \
  --gpus '"device=7"' \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$DATA_HOST:/data" \
  -v "$DATA_HOST/libero_cache/assets:/root/.cache/libero/assets:ro" \
  -e HF_HOME=/data/huggingface \
  -e RAY_TMPDIR=/data/raytmp_eval \
  -e TMPDIR=/data/tmp_eval \
  -e LIBERO_CONFIG_PATH=/data/libero_config \
  -e MUJOCO_GL=osmesa \
  -e PYOPENGL_PLATFORM=osmesa \
  -e MODEL_PATH=/data/models/gr00t_n1d6_libero_spatial_step1000 \
  -e NORM_STATS_PATH=null \
  -e MODEL_GPUS=1 \
  -e ENV_WORKERS=1 \
  -e NUM_ENVS=3 \
  -e TASK_IDS='[0]' \
  -e TRIALS_PER_TASK=3 \
  "$IMAGE" \
  bash -lc 'bash examples/gr00t_sft/run_gr00t_libero_eval.sh'
```

## 7. 验证社区 LIBERO checkpoint

`validate_community_checkpoint.sh` 默认下载并验证：

```text
0xAnkitSingh/GR00T-N1.6-LIBERO
revision d690a226ad06e81736786f56cf879d2ed1dd3f0f
```

默认配置评测 `libero_spatial task 0` 的 3 个 trial。该 checkpoint 是社区
checkpoint，不是 NVIDIA 官方发布的 LIBERO checkpoint。

### 原生环境

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
  -v "$DATA_HOST/libero_cache/assets:/root/.cache/libero/assets:ro" \
  -e HF_HOME=/data/huggingface \
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

成功的 validation 会输出：

```text
val/trajectory_count
val/success_trajectory_count
val/failed_trajectory_count
val/trajectory_success_rate
val/per_task_success_rate/task_0
```

当前集成范围是 GR00T policy SFT、上游兼容 checkpoint export 和 val-only
LIBERO rollout。GR00T SAC、RECAP 和 Flow-SDE 训练不在本集成范围内。
