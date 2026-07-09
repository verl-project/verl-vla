# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Merge a verl FSDP GR00T N1.6 checkpoint into an upstream HF directory.

The verl adapter does not add parameters or prefixes. This script reuses verl's
standard FSDP shard merger, selects ``AutoModel`` for N1.6, and then saves the
official processor with the explicit LIBERO normalization statistics.

Example:
    python scripts/merge_gr00t_fsdp_checkpoint.py \
        --local-dir /data/output/gr00t_sft/global_step_1000/actor \
        --base-model nvidia/GR00T-N1.6-3B \
        --norm-stats /data/datasets/libero_spatial_image/norm_stats.json \
        --target-dir /data/models/gr00t_sft_step1000_hf \
        --verify
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoModel
from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

from verl_vla.models.gr00t_n1d6.policy.libero_policy import load_gr00t_processor
from verl_vla.models.register_vla_models import register_gr00t_n1d6_model


class Gr00tN1d6FSDPModelMerger(FSDPModelMerger):
    """Select AutoModel and skip verl's tokenizer-only save path for GR00T."""

    def get_transformers_auto_model_class(self):
        return AutoModel

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        with init_empty_weights():
            model = AutoModel.from_config(
                self.model_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=self.config.trust_remote_code,
            )
        model.to_empty(device="cpu")
        print(f"Saving GR00T model to {self.config.target_dir}")
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)


def find_latest_actor_dir(checkpoint_root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("global_step_*/actor"):
        match = re.fullmatch(r"global_step_(\d+)", path.parent.name)
        if match and (path / "fsdp_config.json").exists():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No global_step_*/actor FSDP checkpoints found under {checkpoint_root}")
    return max(candidates, key=lambda item: item[0])[1]


def merge_checkpoint(local_dir: Path, target_dir: Path, base_model: str, norm_stats: Path) -> None:
    register_gr00t_n1d6_model(required=True)
    config_dir = local_dir / "huggingface"
    if not (config_dir / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint is missing {config_dir / 'config.json'}")

    merger_config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(target_dir),
        trust_remote_code=True,
        local_dir=str(local_dir),
        hf_model_config_path=str(config_dir),
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    Gr00tN1d6FSDPModelMerger(merger_config).merge_and_save()

    processor = load_gr00t_processor(base_model, str(norm_stats), training=False)
    processor.save_pretrained(target_dir)

    # Never persist machine-local processor/stat paths. AutoProcessor loads the
    # files saved immediately above after the checkpoint is relocated.
    config_path = target_dir / "config.json"
    with config_path.open(encoding="utf-8") as file:
        model_config = json.load(file)
    model_config["architectures"] = ["Gr00tN1d6"]
    model_config["verl_processor_path"] = None
    model_config["verl_norm_stats_path"] = None
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(model_config, file, indent=2, sort_keys=True)
        file.write("\n")


def verify_in_fresh_process(target_dir: Path) -> None:
    code = """
import sys
import gr00t.model  # registers the upstream N1.6 classes
from transformers import AutoModel, AutoProcessor
path = sys.argv[1]
model = AutoModel.from_pretrained(path, low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(path)
print(f'upstream load verified: {model.__class__.__name__}, {processor.__class__.__name__}')
"""
    subprocess.run([sys.executable, "-c", code, str(target_dir)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local-dir", type=Path, help="Checkpoint actor directory.")
    source.add_argument("--checkpoint-root", type=Path, help="Root containing global_step_*/actor.")
    parser.add_argument("--base-model", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Load with upstream classes in a fresh process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = args.local_dir if args.local_dir is not None else find_latest_actor_dir(args.checkpoint_root)
    local_dir = local_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    norm_stats = args.norm_stats.expanduser().resolve()
    print(f"Merging checkpoint: {local_dir}")
    print(f"Writing upstream checkpoint: {target_dir}")
    merge_checkpoint(local_dir, target_dir, args.base_model, norm_stats)
    if args.verify:
        verify_in_fresh_process(target_dir)


if __name__ == "__main__":
    main()
