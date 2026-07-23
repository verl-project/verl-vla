# GR00T Arena SAC Post-Training & Eval

Everything you need to run **GR00T N1.6** policies on **IsaacLab-Arena** tasks —
policy evaluation and SAC post-training — from the host, in one Docker image.

Two tasks are supported, selected with `ARENA_TASK`:

| `ARENA_TASK` | Simulator / embodiment | Task |
| --- | --- | --- |
| `gr1` (default) | GR1 humanoid, 26-DOF joint space, `embodiment_tag=gr1` | Put bottle in fridge & close door |
| `libero` | Franka Abs-IK, 7-DOF eef-pose, `embodiment_tag=new_embodiment` | LIBERO suites (needs LIBERO USD assets) |

---

## Hardware requirements

- **NVIDIA GPU with recent driver** (CUDA >=12.8 runtime).
- **≥ 2 GPUs for eval and SAC.** The env-loop cluster runs the Isaac Sim **env
  worker** and the GR00T **model/rollout worker** as separate Ray workers, each
  reserving its own GPU. A single-GPU host fails at Ray resource allocation
  (`Total available GPUs 1 is less than total desired GPUs 2`).

---

## How it works

```
run_docker.sh   →   (re)creates the GR00T container + mounts   →   runs an inner script
```

- **`run_docker.sh`** is the container launcher. It picks the GR00T image, mounts
  the right host dirs, and runs an inner script directly for root jobs or via
  `docker exec` for non-root jobs.
- The inner script (selected with `INNER_SCRIPT`) builds the actual Hydra command
  and runs it with the in-container `python` (`/isaac-sim/python.sh`).

| File | Role |
| --- | --- |
| `run_docker.sh` | GR00T container launcher. Mounts image/repo/Arena/models, runs the inner script named by `INNER_SCRIPT`. |
| `run_gr00t_arena_eval.sh` | GR00T rollout through the shared `eval` workflow. `ARENA_TASK=gr1\|libero`. |
| `run_gr00t_arena_sac.sh` | GR00T SAC training. `ARENA_TASK=gr1\|libero`. |

Inner scripts are meant to run *inside* the container, but `run_docker.sh`
launches them for you — you normally never call them directly.

---

## Setup

Do steps 1–3 once. Step 4 is only for the LIBERO task.

All commands assume you are in the **verl-vla repo root**.

### 1. Clone IsaacLab-Arena **and sync its submodules**

`run_docker.sh` bind-mounts a host IsaacLab-Arena checkout into the container at
`/workspaces/isaaclab_arena`. The image installs `isaaclab` **and** `gr00t` as
*editable* packages that point back into this checkout's submodules — so if the
submodules are empty you get `ModuleNotFoundError: No module named 'isaaclab'` /
`'gr00t'` at runtime. Syncing them is mandatory.

Clone into `./IsaacLab-Arena` (the default `ARENA_HOST`):

```bash
git clone -b reb/arena-verl \
  https://github.com/rebeccazhang0707/IsaacLab-Arena.git \
  IsaacLab-Arena

git -C IsaacLab-Arena submodule update --init --recursive   # submodules/IsaacLab + Isaac-GR00T
```

Verify both are populated (non-empty):

```bash
ls IsaacLab-Arena/submodules/IsaacLab/source     # → isaaclab, isaaclab_tasks, ...
ls IsaacLab-Arena/submodules/Isaac-GR00T/gr00t   # → configs, model, policy, ...
```

### 2. Build the GR00T image

The Dockerfile lives in **this repo** (`docker/Dockerfile.isaaclab_arena`), but its
`COPY` paths (`submodules/…`, `isaaclab_arena*`, `docker/setup/…`) resolve against the
**build context**, not against the Dockerfile's location — and Docker `COPY` cannot
read host paths outside the context. So build from the verl-vla repo root, point `-f`
at the in-repo Dockerfile, and pass your local Arena checkout (`ARENA_HOST`, the same
one the launcher mounts) as the build context. Pass `INSTALL_GROOT=true` to install
the GR00T / Eagle / CUDA 12.8 stack into `/opt/groot_deps`, and tag it exactly as the
launcher expects:

```bash
ARENA_HOST=/path/to/IsaacLab-Arena   # your local Arena checkout (submodules populated)
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.isaaclab_arena \
  --build-arg INSTALL_GROOT=true \
  -t isaaclab_arena:cuda_gr00t_gn16 \
  "$ARENA_HOST"
```


### 3. Prepare a GR00T N1.6 checkpoint

Pull the checkpoint(s) you need from HuggingFace (`--include "checkpoint-10000/*"`
preserves the `checkpoint-10000/` subdir; `--local-dir` sets the parent that gets
mounted to `/models`).

**GR1 fridge checkpoint** → `./checkpoints/checkpoint-10000` (the default path — the
GR1 commands then need no override):

```bash
pip install -U "huggingface_hub[cli]"
hf download china-sae-robotics/gr00t_n16_gr1_sequential_task \
  --include "checkpoint-10000/*" \
  --local-dir checkpoints
```

- Source: <https://huggingface.co/china-sae-robotics/gr00t_n16_gr1_sequential_task/tree/main/checkpoint-10000>

**LIBERO checkpoint** → a separate parent (`./checkpoints_libero`, since it is also
named `checkpoint-10000`); pass `MODELS_HOST=checkpoints_libero` when running LIBERO:

```bash
hf download china-sae-robotics/gr00t_n16_arena_libero_all_suites_rel_rotvec \
  --include "checkpoint-10000/*" \
  --local-dir checkpoints_libero
```

- Source: <https://huggingface.co/china-sae-robotics/gr00t_n16_arena_libero_all_suites_rel_rotvec/tree/main/checkpoint-10000>

To use a checkpoint elsewhere, set `MODELS_HOST=/abs/parent/dir` and
`GROOT_MODEL_PATH=/models/<your-ckpt-dir>`.

The checkpoint's `embodiment_id.json` must include the embodiment you evaluate
(`gr1` → id 20 by default).

### 4. (LIBERO only) provide the LIBERO assets

The GR1 task needs **no** manual assets. The **LIBERO** task resolves three
directories under `LIBERO_IN_LAB_ROOT` (the `/libero_in_lab` mount), all under
`benchmarks/datasets/libero/`:

| Subdir | Resolves env var | Source |
| --- | --- | --- |
| `USD/` | `LIBERO_ASSETS_DATA_DIR` (scene / object USDs) | HF `RobotLearningLab_Dataset` |
| `assembled_hdf5/` | `LIBERO_ASSEMBLED_DATASET_DIR` (demo HDF5 for state reset) | HF `RobotLearningLab_Dataset` |


Download the USD + HDF5 assets so they land at
`libero_in_lab/benchmarks/datasets/libero/{USD,assembled_hdf5}` (pointing
`--local-dir` at `…/benchmarks/datasets` preserves the repo-relative
`libero/USD` and `libero/assembled_hdf5` paths):

```bash
pip install -U "huggingface_hub[cli]"
hf download china-sae-robotics/RobotLearningLab_Dataset \
  --repo-type dataset \
  --include "libero/USD/*" "libero/assembled_hdf5/*" \
  --local-dir libero_in_lab/benchmarks/datasets
```

The task-config JSONs are **not** hosted on the HF dataset — copy them from the
`libero_in_lab` / `isaaclab_playground` source into
`libero_in_lab/benchmarks/datasets/libero/config/`.

Source: <https://huggingface.co/datasets/china-sae-robotics/RobotLearningLab_Dataset/tree/main/libero>

### 5. Scene assets (automatic)

The GR1 fridge scene's floorplan / object USDs are downloaded from the Arena asset
registry on the **first** run and cached in the container — no manual step. The
first eval therefore takes longer while assets download.

---
## Typical commands

Run all of these from the **verl-vla repo root** on the host.

Defaults assume the checkpoint is at `./checkpoints/checkpoint-10000`. Override
`GROOT_MODEL_PATH` (container path under `/models`) to pick a different one.

### GR1 fridge task eval (verified default)

```bash
examples/gr00t_arena_sac/run_docker.sh
```

### LIBERO spatial task 3 eval

```bash
INNER_SCRIPT=examples/gr00t_arena_sac/run_gr00t_arena_eval.sh \
ARENA_TASK=libero \
MODELS_HOST=checkpoints_libero GROOT_MODEL_PATH=/models/checkpoint-10000 \
  examples/gr00t_arena_sac/run_docker.sh
```

### GR1 fridge task SAC training

```bash
INNER_SCRIPT=examples/gr00t_arena_sac/run_gr00t_arena_sac.sh \
ARENA_TASK=gr1 \
GROOT_MODEL_PATH=/models/checkpoint-10000 \
OUTPUT_ROOT=/eval/outputs/arena_gr00t_gr1_sac \
  examples/gr00t_arena_sac/run_docker.sh
```

### LIBERO SAC training (use a separate container)

```bash
INNER_SCRIPT=examples/gr00t_arena_sac/run_gr00t_arena_sac.sh \
ARENA_TASK=libero \
MODELS_HOST=checkpoints_libero GROOT_MODEL_PATH=/models/checkpoint-10000 \
OUTPUT_ROOT=/eval/outputs/arena_gr00t_libero_sac \
CONTAINER_NAME=isaaclab_arena-cuda_gr00t_gn16_sac \
  examples/gr00t_arena_sac/run_docker.sh
```

> SAC reconstructs complete episodes per environment lane before adding them to
> replay. A LIBERO episode runs up to 512 env steps but a rollout window covers
> only `10 × 16 = 160`, so incomplete suffixes remain buffered across windows.
> One-slot terminal segments are discarded as non-auto-reset padding.

### Start a container / shell only

```bash
examples/gr00t_arena_sac/run_docker.sh --no-run   # (re)start GR00T container only
examples/gr00t_arena_sac/run_docker.sh --shell    # GR00T interactive shell
```

Extra Hydra overrides append to any inner script and are forwarded via `"$@"`,
e.g. inside `--shell`:

```bash
GROOT_MODEL_PATH=/models/checkpoint-10000 \
  bash examples/gr00t_arena_sac/run_gr00t_arena_eval.sh max_episodes=1
```

---

## Code layout

The GR00T policy and the Arena environment live in the verl-vla source tree.

### `src/verl_vla/models/gr00t_n1d6/` — GR00T N1.6 model wrapper

```
gr00t_n1d6/
├── __init__.py          # package doc; pins GR00T_N1D6_COMMIT (the Isaac-GR00T commit this wraps)
├── trainable_model.py   # Gr00tN1d6TrainableModel: main trainable wrapper around the native
│                        #   GR00T policy + optional SAC critic / Flow-SDE actor sampling;
│                        #   load_gr00t_n1d6_policy() loader (entered from build_vla_model)
├── gr00t_adapter.py     # GR00TN16Adapter: raw obs → checkpoint's Gr00tN1d6Processor → model
│                        #   inputs, and action de-normalisation (canonical Gr00tPolicy path)
├── adapter_config.py    # Gr00tAdapterConfig / Gr00tCriticConfig: framework-side settings
│                        #   (policy IO, critic, Flow-SDE). NOT a HF PretrainedConfig
├── compat.py            # process-wide load-time shims (Eagle/transformers 4.51.3, cuDNN SDPA,
│                        #   FSDP2 interpolate); opt-in via GR00T_COMPAT_PATCHES (default off)
├── utils.py             # shared helpers: fallback constants, flat-state→joint-group geometry,
│                        #   checkpoint state-dict remap / critic extraction
├── critic/              # SAC critic backends (used only when adapter.critic.enabled)
│   ├── base.py          #   CriticBackend ABC (scores the normalised full_action)
│   ├── backends.py      #   cross-attention and mean-pool backend adapters
│   ├── group.py         #   shared critic module group (heads + optional cross-attn / pool proj)
│   └── mlp.py           #   critic MLP (SiLU + optional LayerNorm)
└── policy/              # per-embodiment IO adapters for the external GR00T policy
    ├── base.py          #   policy IO contract (mirrors pi0_torch/policy/base.py)
    ├── arena_policy.py  #   Arena / GR1 IO adapter (26-DOF joint space)
    └── libero_policy.py #   LIBERO IO adapter (7-DOF eef-pose)
```

### `src/verl_vla/envs/arena/` — IsaacLab-Arena environment

```
arena/
├── __init__.py     # exports IsaacLabArenaEnv
├── arena_env.py    # IsaacLabArenaEnv: Arena env on the shared BaseEnv interface —
│                   #   launches Isaac Sim (AppLauncher), reset/step, video recorders
├── config.py       # Arena backend config with G1, GR1, and LIBERO environment subconfigs
├── embodiment.py   # embodiment adapters: obs/action mapping per embodiment (GR1 / Franka),
│                   #   joint-space YAMLs, LIBERO asset-root resolution
└── utils.py        # helpers for IsaacLabArenaEnv
```
---

## Reference

### Image / container / mounts

| | GR00T |
| --- | --- |
| Image | `isaaclab_arena:cuda_gr00t_gn16` |
| Container name | `isaaclab_arena-cuda_gr00t_gn16` |
| Container user | non-root — a host-matching user (your uid/gid) with passwordless sudo |
| Repo mount | host verl-vla repo → `/eval` |
| GR00T deps | `/opt/groot_deps` (Eagle / transformers 4.51.3) |
| Extra mounts | Arena, `/models`, `/libero_in_lab` |
| Default inner script | `run_gr00t_arena_eval.sh` |

#### Host → container mounts

Defaults are relative to the verl-vla repo root (`<repo>`).

| Host path (default) | Container path | Set via |
| --- | --- | --- |
| `<repo>` | `/eval` | (this repo) |
| `<repo>/checkpoints` | `/models` | `MODELS_HOST` |
| `<repo>/IsaacLab-Arena` | `/workspaces/isaaclab_arena` | `ARENA_HOST` |
| `<repo>/libero_in_lab` | `/libero_in_lab` | `LIBERO_IN_LAB_HOST` |

> `ARENA_HOST` **must** exist with submodules synced (step 1). The launcher checks
> that the bind-mounted Arena has the env code it needs after mounting.

### Path / checkpoint defaults

| What | Default (container path unless noted) | Env var |
| --- | --- | --- |
| GR00T checkpoint | `/models/checkpoint-10000` | `GROOT_MODEL_PATH` |
| GR00T checkpoint parent (host) | `$MODELS_HOST` | `MODELS_HOST` |
| GR1 joint-space YAMLs | `/workspaces/isaaclab_arena/isaaclab_arena_gr00t/embodiments/gr1` | `ARENA_GR1_JOINT_SPACE_DIR` |
| LIBERO assets root | `/libero_in_lab` | `LIBERO_IN_LAB_ROOT` |

### Output directories

Outputs land under `<repo>/outputs/…` on the host (the repo is bind-mounted).
`OUTPUT_ROOT` is a *container* path; it defaults under `/eval/outputs`.

| Run | Default `OUTPUT_ROOT` basename | Contents |
| --- | --- | --- |
| GR00T GR1 eval | `arena_gr00t_gr1_eval` | `videos/`, `metrics.json`, `hydra/` |
| GR00T LIBERO eval | `arena_gr00t_<suite>_task<id>_eval` | `videos/`, `metrics.json`, `hydra/` |
| GR00T GR1 SAC | `arena_gr00t_gr1_sac` | `videos/`, `checkpoints/`, `replay_pools/` |
| GR00T LIBERO SAC | `arena_gr00t_libero_sac` | `videos/`, `checkpoints/`, `replay_pools/` |

### Environment variables — `run_docker.sh` (launcher)

| Var | Default | Meaning |
| --- | --- | --- |
| `INNER_SCRIPT` | `run_gr00t_arena_eval.sh` | Inner script (path relative to the repo inside the container). `EVAL_SCRIPT` is a deprecated alias. |
| `IMAGE` | `isaaclab_arena:cuda_gr00t_gn16` | Override the docker image. |
| `CONTAINER_NAME` | `isaaclab_arena-cuda_gr00t_gn16` | Override the container name. |
| `RECREATE` | `0` | `1` forces remove + recreate of the container. |
| `DIRECT_RUN` | `1` for root, otherwise `0` | Use a one-shot container instead of the long-lived `docker exec` mode. |
| `RAY_TMPDIR` | `/tmp/ray` | Short Ray session path; avoids the AF_UNIX 107-byte path limit. |
| `MAX_EPISODES` | `10` | Episodes to evaluate (ignored by SAC training). |
| `OUTPUT_ROOT` | inner-script default | Eval/train output root (container path). |
| `ARENA_TASK` | `gr1` | Forwarded to GR00T inner scripts (`gr1`/`libero`). |
| `MODELS_HOST` | `<repo>/checkpoints` | Checkpoint parent → `/models`. |
| `GROOT_MODEL_PATH` | `/models/checkpoint-10000` | Checkpoint path inside the container. |
| `ARENA_HOST` | `<repo>/IsaacLab-Arena` | Arena checkout → `/workspaces/isaaclab_arena`. |
| `LIBERO_IN_LAB_HOST` | `<repo>/libero_in_lab` | LIBERO assets → `/libero_in_lab`. |
| `TASK_SUITE` / `TASK_ID` | (unset → inner default) | (libero) LIBERO suite / task id. |

### Environment variables — inner scripts

| Var | GR1 default | LIBERO default | Meaning |
| --- | --- | --- | --- |
| `ARENA_TASK` | `gr1` | `libero` | Selects simulator + embodiment. |
| `GROOT_EMBODIMENT_TAG` | `gr1` | `new_embodiment` | Embodiment tag. |
| `GROOT_EMBODIMENT_ID` | `20` | `10` | Projector index. |
| `ACTION_DIM` | `26` | `7` | Real (unpadded) env action width. |
| `NUM_ACTION_CHUNKS` | `16` | `16` | Executed action-chunk length (must match training). |
| `MAX_INTERACTIONS` | `32` | `10` | `env_loop` interactions per rollout. |
| `TASK_SUITE` / `TASK_ID` | — | `libero_spatial` / `3` | LIBERO task (libero only). |
| `MAX_EPISODES` (eval) | `10` | `10` | Episodes to evaluate. |

`run_gr00t_arena_sac.sh` additionally exposes SAC knobs (all overridable):
`NUM_ENV` (8), `NUM_ENV_GPUS`/`NUM_MODEL_GPUS` (1), `NUM_STAGE` (2),
`MINI_BATCH_SIZE` (128), `MICRO_BATCH_SIZE` (32), `TOTAL_EPOCHS` (1000),
`ROLLOUT_INTERVAL` (20), `WARM_ROLLOUT_STEPS` (5), `CRITIC_WARMUP_STEPS` (200),
`SAVE_FREQ` (500), `INITIAL_ALPHA` (0.01), `ALPHA_TYPE` (softplus),
`AUTO_ENTROPY` (False), `CRITIC_TAU` (0.01), `RESUME_MODE`/`RESUME_FROM_PATH`,
`PROJECT_NAME`, `EXPERIMENT_NAME`, `TRAINER_LOGGER` (`[console]`).

---
