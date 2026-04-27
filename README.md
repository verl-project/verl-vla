# verl-vla

Experimental VLA training support built on top of `verl`, currently focused on PI0.5 workflows for Libero, Isaac, and LeRobot-style SFT.

## Supported Simulators

| Simulator | Env Name | Difference | Benchmark data source |
| --- | --- | --- | --- |
| Mujoco | `LiberoEnv` | 1. Initialize task from `init_states` in Libero dataset. 2. Each env can have different tasks. | https://github.com/Lifelong-Robot-Learning/LIBERO |
| IsaacSim | `IsaacEnv` | 1. Initialize from randomized states with more variety than dataset `init_states`. 2. Each sim process must use the same task for its envs. | https://huggingface.co/datasets/china-sae-robotics/IsaacLabPlayGround_Dataset |

## Hardware Requirements

- Simulator GPU: NVIDIA L20 or L40 with 48GB memory and RT Cores

Notes:

1. Mujoco can fall back to CPU mode with degraded performance if RT Cores are unavailable.
2. IsaacSim requires GPUs with RT Cores.
3. RTX GPU support is planned, but it does not work well with colocated mode under current memory limits.

## Docker Image

Isaac Lab support for Libero depends on RobotLearningLab from The Isaac Lab Project Developers team. It is currently bundled into the preview image below.

Example image:

```text
vemlp-demo-cn-beijing.cr.volces.com/verl/pi05-libero-sac:v0.2
```

## Dataset Preparation

Libero parquet generation script:

```bash
python scripts/prepare_libero_dataset.py
```

Adjust paths inside the script or your environment before generating data.

## Training Entry

The current default training entry is SAC, and the repo now also includes a LeRobot-based SFT entry.

Main Python entry:

```bash
python -m verl_vla.trainer.main_sac
```

Recommended launcher script:

```bash
bash examples/libero_sac/run_pi05_libero_sac.sh
```

Disaggregated launcher:

```bash
bash examples/libero_sac/run_pi05_libero_sac_disagg.sh
```

SFT entry:

```bash
python -m verl_vla.trainer.main_sft
```

Recommended SFT launcher:

```bash
bash examples/lerobot_sft/run_pi05_lerobot_sft.sh
```

## Disaggregation Mode

Train-rollout workers and simulation workers can be placed on different nodes.

Start Ray on the main train-rollout node:

```bash
ray start --head --dashboard-host=0.0.0.0 --resources='{"train_rollout": 1}'
```

Start Ray on each simulation node:

```bash
ray start --address='<main_node_ip>:6379' --resources='{"sim": 1}'
```

Then launch training on the main node only.
