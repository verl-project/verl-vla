# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compute the explicit LIBERO statistics required by GR00T N1.6.

Unlike the framework's generic normalization script, GR00T's default
StateActionProcessor requires min/max in addition to mean/std/q01/q99. This
dedicated entry also accepts a local LeRobot dataset root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from compute_norm_stats import RunningStats, _resolve_batch_key, _to_numpy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm


class Gr00tRunningStats(RunningStats):
    def get_statistics(self) -> dict[str, list[float]]:
        stats = super().get_statistics()
        if self.min_ is None or self.max_ is None:
            raise ValueError("Running min/max are not initialized.")
        return {
            "min": self.min_.astype(np.float32).tolist(),
            "max": self.max_.astype(np.float32).tolist(),
            **stats,
        }


def create_dataloader(args: argparse.Namespace) -> tuple[StatefulDataLoader, int]:
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        revision=args.revision,
        video_backend=args.video_backend,
    )
    if args.shuffle:
        generator = torch.Generator()
        if args.seed is not None:
            generator.manual_seed(int(args.seed))
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        sampler=sampler,
    )
    total_frames = len(dataset)
    if args.max_frames is not None:
        total_frames = min(total_frames, int(args.max_frames))
    num_batches = (
        total_frames // args.batch_size
        if args.drop_last
        else (total_frames + args.batch_size - 1) // args.batch_size
    )
    return dataloader, max(1, num_batches)


def compute(args: argparse.Namespace) -> None:
    dataloader, num_batches = create_dataloader(args)
    state_stats = Gr00tRunningStats()
    action_stats = Gr00tRunningStats()
    processed_frames = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Computing GR00T stats")):
        if args.max_frames is not None and processed_frames >= args.max_frames:
            break
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dataloader batch to be dict, got {type(batch)}")

        state_key = _resolve_batch_key(batch, ["observation.state", "state"], "state")
        action_key = _resolve_batch_key(batch, ["action.full_action", "action.action", "action"], "action")
        state = _to_numpy(batch[state_key])
        action = _to_numpy(batch[action_key])
        if state is None or action is None:
            raise TypeError("State/action values must be torch.Tensor or numpy.ndarray.")

        if args.max_frames is not None:
            remaining = int(args.max_frames) - processed_frames
            state = state[:remaining]
            action = action[:remaining]
        state_stats.update(state)
        action_stats.update(action)
        processed_frames += int(state.shape[0])
        if batch_idx + 1 >= num_batches:
            break

    output = {
        "state": state_stats.get_statistics(),
        "action": action_stats.get_statistics(),
        "meta": {
            "repo_id": args.repo_id,
            "root": args.root,
            "processed_frames": processed_frames,
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(f"Saved GR00T normalization statistics to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--drop-last", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    compute(parse_args())
