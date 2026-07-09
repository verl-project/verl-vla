# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Compute normalization statistics for LeRobot SFT datasets.

This script follows the same dataset loading style as the current VLA SFT pipeline:
`LeRobotDataset` + `StatefulDataLoader`.

It computes mean/std/q01/q99 for state and action tensors, with optional min/max,
and writes a JSON file like:
{
  "state": {"min": [...], "max": [...], "mean": [...], "std": [...], "q01": [...], "q99": [...]},
  "action": {"min": [...], "max": [...], "mean": [...], "std": [...], "q01": [...], "q99": [...]}
}

``min`` and ``max`` are emitted only when ``--include-min-max`` is set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm


@dataclass
class RunningStats:
    """Streaming mean/std/q01/q99 over the last dimension of incoming tensors."""

    sum_: np.ndarray | None = None
    sq_sum_: np.ndarray | None = None
    min_: np.ndarray | None = None
    max_: np.ndarray | None = None
    histograms_: list[np.ndarray] | None = None
    bin_edges_: list[np.ndarray] | None = None
    count: int = 0
    num_quantile_bins: int = 5000

    def _as_2d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 0:
            return x.reshape(1, 1)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x.reshape(-1, x.shape[-1])

    def _init_histograms(self, x: np.ndarray) -> None:
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)

        vector_length = x.shape[1]
        self.histograms_ = [np.zeros(self.num_quantile_bins, dtype=np.float64) for _ in range(vector_length)]
        self.bin_edges_ = []
        for i in range(vector_length):
            lo = float(self.min_[i])
            hi = float(self.max_[i])
            if hi <= lo:
                lo -= 1e-6
                hi += 1e-6
            self.bin_edges_.append(np.linspace(lo, hi, self.num_quantile_bins + 1, dtype=np.float64))

    def _adjust_histograms(self) -> None:
        if self.histograms_ is None or self.bin_edges_ is None or self.min_ is None or self.max_ is None:
            return

        for i in range(len(self.histograms_)):
            lo = float(self.min_[i])
            hi = float(self.max_[i])
            if hi <= lo:
                lo -= 1e-6
                hi += 1e-6

            old_edges = self.bin_edges_[i]
            old_hist = self.histograms_[i]
            new_edges = np.linspace(lo, hi, self.num_quantile_bins + 1, dtype=np.float64)

            # Re-bin existing counts into the updated range.
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=old_hist)
            self.histograms_[i] = new_hist.astype(np.float64)
            self.bin_edges_[i] = new_edges

    def _update_histograms(self, x: np.ndarray) -> None:
        if self.histograms_ is None or self.bin_edges_ is None:
            return
        for i in range(x.shape[1]):
            hist, _ = np.histogram(x[:, i], bins=self.bin_edges_[i])
            self.histograms_[i] += hist

    def _compute_quantiles(self, quantiles: list[float]) -> list[np.ndarray]:
        if self.histograms_ is None or self.bin_edges_ is None or self.count <= 0:
            raise ValueError("Cannot compute quantiles before collecting any data.")

        out = []
        for q in quantiles:
            target = q * self.count
            q_values = []
            for hist, edges in zip(self.histograms_, self.bin_edges_, strict=False):
                cumsum = np.cumsum(hist)
                idx = int(np.searchsorted(cumsum, target, side="left"))
                idx = min(max(idx, 0), len(edges) - 2)
                q_values.append(edges[idx])
            out.append(np.asarray(q_values, dtype=np.float64))
        return out

    def update(self, x: np.ndarray) -> None:
        x = self._as_2d(x)

        x = x.astype(np.float64, copy=False)
        batch_sum = x.sum(axis=0)
        batch_sq_sum = np.square(x).sum(axis=0)
        batch_count = x.shape[0]

        if self.sum_ is None:
            self.sum_ = batch_sum
            self.sq_sum_ = batch_sq_sum
            self._init_histograms(x)
        else:
            if batch_sum.shape[0] != self.sum_.shape[0]:
                raise ValueError("Incoming batch vector length does not match initialized vector length.")

            if self.min_ is None or self.max_ is None:
                raise ValueError("Running min/max are not initialized.")

            batch_min = np.min(x, axis=0)
            batch_max = np.max(x, axis=0)
            min_changed = np.any(batch_min < self.min_)
            max_changed = np.any(batch_max > self.max_)
            if min_changed or max_changed:
                self.min_ = np.minimum(self.min_, batch_min)
                self.max_ = np.maximum(self.max_, batch_max)
                self._adjust_histograms()

            self.sum_ += batch_sum
            self.sq_sum_ += batch_sq_sum

        self._update_histograms(x)
        self.count += int(batch_count)

    def get_statistics(self, *, include_min_max: bool = False) -> dict[str, list[float]]:
        if self.sum_ is None or self.sq_sum_ is None or self.count == 0:
            raise ValueError("No data collected for running stats.")

        mean = self.sum_ / self.count
        var = self.sq_sum_ / self.count - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-12))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        statistics = {
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
        }
        if include_min_max:
            if self.min_ is None or self.max_ is None:
                raise ValueError("Running min/max are not initialized.")
            statistics = {
                "min": self.min_.astype(np.float32).tolist(),
                "max": self.max_.astype(np.float32).tolist(),
                **statistics,
            }
        return statistics


def _to_numpy(value) -> np.ndarray | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _resolve_batch_key(batch: dict, candidates: list[str], name: str) -> str:
    for key in candidates:
        if key in batch:
            return key
    raise KeyError(f"Cannot find {name} key. Tried: {candidates}. Available keys: {sorted(batch.keys())}")


def create_lerobot_dataloader(
    repo_id: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int | None,
    revision: str | None,
    video_backend: str | None,
    drop_last: bool,
    max_frames: int | None,
    root: str | None = None,
) -> tuple[StatefulDataLoader, int]:
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        revision=revision,
        video_backend=video_backend,
    )

    if shuffle:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
        sampler = RandomSampler(data_source=dataset, generator=generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    data_loader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        sampler=sampler,
    )

    total_frames = len(dataset)
    if max_frames is not None:
        total_frames = min(total_frames, int(max_frames))

    num_batches = total_frames // batch_size if drop_last else (total_frames + batch_size - 1) // batch_size
    num_batches = max(1, num_batches)
    return data_loader, num_batches


def compute_norm_stats(
    repo_id: str,
    output_path: str,
    root: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    max_frames: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    revision: str | None = None,
    video_backend: str | None = "pyav",
    drop_last: bool = False,
    include_min_max: bool = False,
) -> None:
    dataloader, num_batches = create_lerobot_dataloader(
        repo_id=repo_id,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        seed=seed,
        revision=revision,
        video_backend=video_backend,
        drop_last=drop_last,
        max_frames=max_frames,
        root=root,
    )

    state_stats = RunningStats()
    action_stats = RunningStats()

    processed_frames = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Computing stats")):
        if max_frames is not None and processed_frames >= max_frames:
            break

        if not isinstance(batch, dict):
            raise TypeError(f"Expected dataloader batch to be dict, got {type(batch)}")

        state_key = _resolve_batch_key(batch, ["observation.state", "state"], "state")
        action_key = _resolve_batch_key(batch, ["action.full_action", "action.action", "action"], "action")

        state_np = _to_numpy(batch[state_key])
        action_np = _to_numpy(batch[action_key])

        if state_np is None or action_np is None:
            raise TypeError(
                "State/action tensors must be torch.Tensor or numpy.ndarray. "
                f"Got state={type(batch[state_key])}, action={type(batch[action_key])}."
            )

        if max_frames is not None:
            remaining = int(max_frames) - processed_frames
            if remaining <= 0:
                break
            state_np = state_np[:remaining]
            action_np = action_np[:remaining]

        state_stats.update(state_np)
        action_stats.update(action_np)

        processed_frames += int(state_np.shape[0])

        if batch_idx + 1 >= num_batches:
            break

    norm_stats = {
        "state": state_stats.get_statistics(include_min_max=include_min_max),
        "action": action_stats.get_statistics(include_min_max=include_min_max),
        "meta": {
            "repo_id": repo_id,
            "root": root,
            "processed_frames": processed_frames,
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)

    print(f"Saved normalization stats to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute state/action normalization statistics for a LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo id, e.g. Miical/record-test")
    parser.add_argument("--root", type=str, default=None, help="Optional local LeRobot dataset root")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSON path for computed stats")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before computing stats")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--video-backend", type=str, default="pyav")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument(
        "--include-min-max",
        action="store_true",
        help="Include per-dimension min/max values required by GR00T normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_norm_stats(
        repo_id=args.repo_id,
        output_path=args.output_path,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        shuffle=args.shuffle,
        seed=args.seed,
        revision=args.revision,
        video_backend=args.video_backend,
        drop_last=args.drop_last,
        include_min_max=args.include_min_max,
    )


if __name__ == "__main__":
    main()
