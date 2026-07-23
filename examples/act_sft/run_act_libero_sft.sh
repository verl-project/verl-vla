#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

vvla-train-sft \
  --config-dir "./examples/act_sft" \
  --config-name act_sft \
  cluster.actor_rollout_ref.model.path="./assets/hf_models/act_libero" \
  cluster.actor_rollout_ref.model.adapter.processor_dataset_root="./outputs/record/lerobot/local/libero_spatial" \
  data.repo_id=local/libero_spatial \
  data.root="./outputs/record/lerobot/local/libero_spatial" \
  data.batch_size=32 \
  cluster.resource.model.gpus_per_node=1 \
  cluster.actor_rollout_ref.actor.mini_batch_size=32 \
  cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  cluster.actor_rollout_ref.actor.optim.lr=1e-4 \
  'trainer.logger=[console,tensorboard]' \
  trainer.total_epochs=100 \
  "$@"
