# README_LANCEL — GR00T on Libero (verl-vla) 使用说明

本机记录的命令。目标硬件:**8×H20(96G/卡)**。数据/权重放在挂载到容器的 `/data`。

> 说明:GPU 用 `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all`(别用 `--gpus all`,会走 CDI 报错)。训练用 `MUJOCO_GL=egl`,评测/仿真用 `MUJOCO_GL=osmesa`。
> **换机器时**把下面 host 路径改成你的:`DATA_HOST`(挂 `/data`)、`REPO_HOST`(挂 `/workspace/verl-vla`)、`STAGING_HOST`(挂 `/staging`,放 Cosmos backbone 等大文件)。

---

## 0. 前置(一次性)

```bash
REPO_HOST=/home/lancel/Projects/verl-vla
DATA_HOST=/mnt/dockerdisk/data
STAGING_HOST=/mnt/workspace8T/lancel_gr00t

# 0.1 构建运行镜像(verl + lerobot + transformers4.57 + GR00T + LIBERO)
cd $REPO_HOST && docker build -f docker/Dockerfile.gr00t -t verl-vla-gr00t:dev .

# 0.2 关键资产(容器内路径)
#   GR00T base(gr00t_torch 格式):            /data/models/gr00t_n1d7_libero_base
#   Cosmos-Reason2-2B(backbone,本地):        /staging/models/Cosmos-Reason2-2B
#   SFT 数据集(标准 Mujoco libero,已验证匹配): /data/datasets/libero_spatial_image
#   LIBERO 仿真资产缓存(首次自动下载):        /data/libero_cache , /data/libero_home

# 8×H20 通用参数
GPUS=8
```

---

## 1. GR00T SFT —— 在 libero_spatial 上训练(8×H20)

脚本 `examples/gr00t_sft/run_gr00t_lerobot_sft.sh`(`main_sft`,已修正 `cluster.*` 键路径)。

```bash
docker run -d --name gr00t_sft --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host --shm-size=32g \
  -v $REPO_HOST:/workspace/verl-vla -v $DATA_HOST:/data -v $STAGING_HOST:/staging \
  -e PYTHONPATH=/workspace/verl-vla/src -e MUJOCO_GL=egl \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e RAY_TMPDIR=/data/raytmp -e TMPDIR=/data/raytmp \
  -e MODEL_PATH=/data/models/gr00t_n1d7_libero_base \
  -e OUTPUT_DIR=/data/output/gr00t_libero_sft \
  -e NUM_GPUS=8 -e MICRO_BATCH_SIZE=16 -e MINI_BATCH_SIZE=256 \
  -e LR=1e-4 -e WEIGHT_DECAY=1e-5 -e WARMUP_RATIO=0.05 \
  -e TOTAL_EPOCHS=10 -e SAVE_FREQ=250 -e SFT_NUM_WORKERS=8 \
  verl-vla-gr00t:dev bash -lc 'cd /workspace/verl-vla && bash examples/gr00t_sft/run_gr00t_lerobot_sft.sh'

docker logs -f gr00t_sft 2>&1 | grep -E 'global_step:[0-9]+ - training'
```

要点:
- checkpoint 存到 `OUTPUT_DIR/global_step_N/`(FSDP2 分片 + `actor/huggingface/`)。
- **可无损续训**:同一个 `OUTPUT_DIR` 重跑即可(`resume_mode=auto`)。
- H20 96G 显存充裕,可再把 `MICRO_BATCH_SIZE` 调到 32、`MINI_BATCH_SIZE` 调到 512 提吞吐(注意 `MINI` 要能被 `NUM_GPUS×MICRO` 整除)。
- 数据/超参用环境变量覆盖:`SFT_REPO_ID` / `SFT_ROOT` / `LR` / `SFT_ACTION_DELTA_STEPS`(默认 16)。

---

## 2. GR00T SFT —— 在 libero_spatial 上验证(成功率)

SFT 存的是 FSDP2 分片,**不能直接被 HF 加载**;先 merge 成 HF 目录,再用 `main_sac` val-only 在 LiberoEnv 评成功率。

```bash
STEP=1000    # 要评的 checkpoint 步数

# 2.1 合并 FSDP 分片 -> HF checkpoint(CPU)
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=none \
  -v $REPO_HOST:/workspace/verl-vla -v $DATA_HOST:/data -e PYTHONPATH=/workspace/verl-vla/src \
  verl-vla-gr00t:dev bash -lc "python scripts/merge_gr00t_fsdp_checkpoint.py \
    --actor-dir /data/output/gr00t_libero_sft/global_step_${STEP}/actor \
    --base-dir  /data/models/gr00t_n1d7_libero_base \
    --output-dir /data/models/gr00t_sft_step${STEP}_hf"

# 2.2 评测(10 task × 10 = 100 条轨迹),报 val/trajectory_success_rate
docker run --rm --name gr00t_eval --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host --shm-size=32g \
  -v $REPO_HOST:/workspace/verl-vla -v $DATA_HOST:/data -v $STAGING_HOST:/staging \
  -v $DATA_HOST/libero_cache:/root/.cache/libero -v $DATA_HOST/libero_home:/root/.libero \
  -e PYTHONPATH=/workspace/verl-vla/src -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e RAY_TMPDIR=/data/raytmp -e TMPDIR=/data/raytmp \
  -e MODEL_PATH=/data/models/gr00t_sft_step${STEP}_hf \
  -e MODEL_GPUS=4 -e TASK_IDS=null -e TRIALS_PER_TASK=10 -e ENV_WORKERS=16 -e NUM_ENVS=4 \
  verl-vla-gr00t:dev bash -lc 'cd /workspace/verl-vla && bash examples/gr00t_sft/run_gr00t_libero_eval.sh' \
  2>&1 | grep -E 'val/trajectory_success_rate|val/success_trajectory_count|per_task_success_rate'
```

要点:
- 评测受限于 env rollout(CPU),GPU 不是瓶颈 → actor 用 `MODEL_GPUS=4` 足够,把并行度堆在 `ENV_WORKERS`(16)× `NUM_ENVS`(4)。
- 常用:`TASK_IDS`(`null`=全 10 个 / `[0]`=单 task)、`TRIALS_PER_TASK`、`MAX_EPISODE_STEPS`(默认 256)。
- 快速自检数据↔环境约定:`python /staging/replay_test.py`(专家动作回放进 LiberoEnv,应大部分成功)。

---

## 3. RECAP —— GR00T 接入 RECAP(逐阶段单独跑)

RECAP 是 `main_recap` 内部依次跑 6 个阶段:
`1 policy_eval → 2 collect_data → 3 compute_return → 4 train_value_model → 5 value_infer → 6 train_policy`。
每个阶段有 `recap.<stage>.enable` 开关;**逐步跑 = 每次只把目标阶段 enable=True、其余 =False**,阶段间靠 `/data` 上的数据集/checkpoint 衔接。

**GR00T 适配(不改模型代码)**:policy 三处(policy_eval / collect_data / train_policy)指向 GR00T;value 模型(train_value_model / value_infer)用独立的 `recap_value_critic`(pi0/Gemma3),保持不变;`train_policy` 的 ACP 只是给 task 拼文本 tag,GR00T 天然支持。

### 3.0 公共变量(6 步都用)
```bash
# 起点:一个"在 libero 成功率 > 0"的 GR00T policy(先用第 1、2 节训到有成功率)
GR00T_POLICY=/data/models/gr00t_sft_stepXXXX_hf
# recap_value_critic 的 base(Gemma3+SigLIP;需单独准备)
VALUE_BASE=/data/models/recap_value_critic_base
# GR00T 的 FSDP wrap 类
WRAP='[Qwen3VLTextDecoderLayer,Qwen3VLVisionBlock,BasicTransformerBlock]'
# RECAP 工作区(必须在 /data,跨 docker 持久化)
COLLECT_REPO=local/gr00t_libero_collected
COLLECT_ROOT=/data/recap_gr00t/collected
VALUE_DIR=/data/recap_gr00t/value_model
POLICY_DIR=/data/recap_gr00t/policy

# 统一的 docker run 前缀(每步把它接上 `python -m verl_vla.trainer.main_recap ...`)
DRUN="docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host --shm-size=32g \
  -v $REPO_HOST:/workspace/verl-vla -v $DATA_HOST:/data -v $STAGING_HOST:/staging \
  -v $DATA_HOST/libero_cache:/root/.cache/libero -v $DATA_HOST/libero_home:/root/.libero \
  -e PYTHONPATH=/workspace/verl-vla/src -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e RAY_TMPDIR=/data/raytmp -e TMPDIR=/data/raytmp \
  verl-vla-gr00t:dev bash -lc"

# 六个阶段的 enable 开关组合(只开目标阶段);EN() 生成这组 override
EN() { # 用法: EN policy_eval  -> 只开 policy_eval
  for s in policy_eval collect_data compute_return train_value_model value_infer train_policy; do
    if [ "$s" = "$1" ]; then echo -n "recap.$s.enable=True "; else echo -n "recap.$s.enable=False "; fi
  done
}
BASE="--config-path src/verl_vla/trainer/config --config-name main_recap recap.env_loop.task_suite_name=libero_spatial"
```

### 3.1 逐阶段命令

**① policy_eval(基线成功率,可选)** — GR00T policy 在 libero 里评一遍。
```bash
$DRUN "python -m verl_vla.trainer.main_recap $BASE $(EN policy_eval) \
  recap.policy_eval.model_path=$GR00T_POLICY \
  recap.policy_eval.cluster.actor_rollout_ref.model.path=$GR00T_POLICY \
  recap.policy_eval.cluster.actor_rollout_ref.model.override_config.policy_type=libero \
  recap.policy_eval.cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=$WRAP \
  recap.policy_eval.cluster.resource.model.gpus_per_node=4"
```

**② collect_data** — 用 GR00T policy rollout,把轨迹录成 LeRobot 数据集到 `$COLLECT_ROOT`。
```bash
$DRUN "python -m verl_vla.trainer.main_recap $BASE $(EN collect_data) \
  recap.collect_data.cluster.actor_rollout_ref.model.path=$GR00T_POLICY \
  recap.collect_data.cluster.actor_rollout_ref.model.override_config.policy_type=libero \
  recap.collect_data.cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=$WRAP \
  recap.collect_data.cluster.resource.model.gpus_per_node=4 \
  recap.collect_data.cluster.env.env_worker.recorder.lerobot.repo_id=$COLLECT_REPO \
  recap.compute_return.dataset.root=$COLLECT_ROOT recap.compute_return.dataset.repo_id=$COLLECT_REPO"
```

**③ compute_return** — 从轨迹算 return/advantage,写回数据集。
```bash
$DRUN "python -m verl_vla.trainer.main_recap $BASE $(EN compute_return) \
  recap.compute_return.dataset.root=$COLLECT_ROOT recap.compute_return.dataset.repo_id=$COLLECT_REPO"
```

**④ train_value_model** — 用采集数据 SFT 训 `recap_value_critic`(target=return)→ 存到 `$VALUE_DIR`。
```bash
$DRUN "python -m verl_vla.trainer.main_recap $BASE $(EN train_value_model) \
  recap.train_value_model.data.root=$COLLECT_ROOT recap.train_value_model.data.repo_id=$COLLECT_REPO \
  recap.train_value_model.cluster.actor_rollout_ref.model.path=$VALUE_BASE \
  recap.train_value_model.cluster.checkpoint.default_local_dir=$VALUE_DIR \
  recap.train_value_model.cluster.resource.model.gpus_per_node=8 \
  recap.train_value_model.cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  recap.train_value_model.cluster.actor_rollout_ref.actor.mini_batch_size=256"
```

**⑤ value_infer** — 用训好的 value 模型给数据打 value/advantage。
```bash
# VALUE_CKPT: 上一步产出的 value 模型 HF 目录(在 $VALUE_DIR/.../global_step_N/actor/huggingface)
VALUE_CKPT=$(ls -d $VALUE_DIR/*/global_step_*/actor/huggingface 2>/dev/null | tail -1)
$DRUN "python -m verl_vla.trainer.main_recap $BASE $(EN value_infer) \
  recap.value_infer.dataset.root=$COLLECT_ROOT recap.value_infer.dataset.repo_id=$COLLECT_REPO \
  recap.value_infer.model_path=$VALUE_CKPT"
```

**⑥ train_policy** — 用 advantage 做 ACP 条件 SFT 重训 GR00T policy → 存到 `$POLICY_DIR`。
```bash
$DRUN "python -m verl_vla.trainer.main_recap $BASE $(EN train_policy) \
  recap.train_policy.dataset.root=$COLLECT_ROOT recap.train_policy.dataset.repo_id=$COLLECT_REPO \
  recap.train_policy.cluster.actor_rollout_ref.model.path=$GR00T_POLICY \
  recap.train_policy.cluster.actor_rollout_ref.model.override_config.policy_type=libero \
  recap.train_policy.cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=$WRAP \
  recap.train_policy.cluster.actor_rollout_ref.actor.acp.enable=True \
  recap.train_policy.cluster.checkpoint.default_local_dir=$POLICY_DIR \
  recap.train_policy.cluster.resource.model.gpus_per_node=8 \
  recap.train_policy.cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  recap.train_policy.cluster.actor_rollout_ref.actor.mini_batch_size=256"
```

### 3.2 评测改进后的 policy
`train_policy` 产出的还是 FSDP2 分片(在 `$POLICY_DIR/.../global_step_N/actor`)→ 按**第 2 节**先 merge 再 eval,对比 RECAP 前后的 `val/trajectory_success_rate`。

> ⚠️ **验证状态**:第 1、2 节已在本机(2×A6000)跑通;**第 3 节 RECAP-GR00T 是基于代码结构推导的配方,尚未端到端验证**。跑前务必:(a) 有一个成功率 >0 的 GR00T policy,(b) 有 `recap_value_critic` base。每个 stage 的确切 key 以 `docker run ... verl-vla-gr00t:dev bash -lc 'python -m verl_vla.trainer.main_recap --cfg job'` dump 的真实配置树为准(尤其 collect_data 的 recorder 输出路径、value_infer 的 model_path 字段名)。
