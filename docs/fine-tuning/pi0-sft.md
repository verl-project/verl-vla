# Fine-tune PI0.5 on LIBERO Spatial

This guide shows how to fine-tune `Miical/pi05-base` on the
`lerobot/libero_spatial_image` dataset using supervised fine-tuning (SFT). We
provide a Docker-based launcher configured for a single node with eight NVIDIA
GPUs. If you prefer to use a local Python environment, install the dependencies
by following the versions and installation order in
[`docker/Dockerfile.pi0`](https://github.com/verl-project/verl-vla/blob/main/docker/Dockerfile.pi0).

## Build the image

Run from the repository root:

```bash
docker build \
  -f docker/Dockerfile.pi0 \
  -t verl-vla-pi0:dev \
  .
```

The image contains the verified PI0.5, verl-vla, LeRobot, and LIBERO runtime,
including the LIBERO assets required for OSMesa rendering.

## Start training

```bash
bash examples/pi05_sft/run_train.sh
```

> **Optional: LoRA fine-tuning**
>
> To train only a LoRA adapter on the native PI0.5 policy, append the LoRA
> overrides to the same launcher:
>
> ```bash
> bash examples/pi05_sft/run_train.sh \
>   cluster.actor_rollout_ref.model.lora.rank=32 \
>   cluster.actor_rollout_ref.model.lora.alpha=32 \
>   cluster.actor_rollout_ref.model.lora.target_modules=all-linear
> ```
>
> The full verl checkpoint retains the adapter and optimizer state for resume.
> The `huggingface/` export contains only the merged native PI0.5 policy and
> remains loadable through the upstream policy implementation. A standard PEFT
> adapter is exported alongside it under `lora_adapter/` for adapter-only
> distribution or continued LoRA training.
>
> To initialize from an existing PEFT adapter directory, also set
> `cluster.actor_rollout_ref.model.lora.adapter_path` and keep `lora.rank` equal
> to the rank recorded by that adapter.

The launcher uses `.data/pi05_sft` as the persistent data directory shared by
the host and the container. Source code is bind-mounted from the repository,
so Python changes are available without rebuilding the image.

On the first run, the launcher automatically:

1. downloads the LIBERO Spatial dataset through LeRobot;
2. computes the dataset normalization statistics;
3. downloads the PI0.5 checkpoint from Hugging Face; and
4. starts distributed SFT on all eight GPUs.

Downloaded files and training outputs remain under `.data/pi05_sft` and are
reused by later runs.

## Default configuration

| Setting | Value |
| --- | --- |
| Model | `Miical/pi05-base` |
| Dataset | `lerobot/libero_spatial_image` |
| Nodes | 1 |
| GPUs | 8 |
| Global batch size | 256 |
| Micro-batch size | 16 |
| DataLoader workers | 8 |
| Action horizon | 10 |
| Epochs | 25 (approximately 5,150 steps) |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Warmup ratio | `0.05` |
| Distributed strategy | FSDP2 |
| Model dtype | BF16 |
| Output | `.data/pi05_sft/output/pi05_libero_spatial_sft` |

## Monitor training

A running job reports loss and gradient metrics in the console:

```text
Training Progress: 1/5150 ... grad_pre=... sft_loss=...
```

The Docker launcher also starts TensorBoard automatically. Event files are
written to:

```text
.data/pi05_sft/output/pi05_libero_spatial_sft/tensorboard
```

Open the following address in a browser to view the training metrics:

```text
http://localhost:6006
```

When training on a remote machine, replace `localhost` with the machine address
or forward port `6006` over SSH.

GPU utilization can be inspected from another terminal:

```bash
watch -n 1 nvidia-smi
```

The loss curve from a reference run is shown below:

![PI0.5 LIBERO Spatial SFT loss](../_static/images/pi05-libero-spatial-sft-loss.png)

## Evaluate the checkpoint

The evaluation launcher reads the latest saved checkpoint and runs the full
LIBERO Spatial benchmark:

```bash
bash examples/pi05_sft/run_eval.sh
```

The reference run evaluated all 10 tasks with 10 trials per task:

| Metric | Value |
| --- | ---: |
| Tasks | 10 |
| Trials per task | 10 |
| Successful trajectories | 100 / 100 |
| Success rate | 100% |
| Average return | 1.0 |
| Average successful trajectory length | 98.72 |
| Average successful trajectory chunk length | 10.34 |
| Total evaluation time | 51.69 seconds |

Every task achieved a 100% success rate. The
[complete evaluation metrics](../_static/results/pi05-libero-spatial-eval.json)
include per-task trajectory lengths, counts, timing, and throughput.
