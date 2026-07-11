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

"""Helpers for collecting LeRobot datasets from env-loop rollouts."""

from __future__ import annotations

import logging
from typing import Any

from verl_vla.recorder.dataset import count_lerobot_episodes, truncate_lerobot_episodes

logger = logging.getLogger(__name__)

CollectedDatasets = dict[str, dict[str, Any]]


def collect_lerobot_rollout_dataset(
    cluster,
    *,
    target_episodes: int,
    initial_completed_episodes: int = 0,
    log_prefix: str,
    max_episodes_name: str = "max_episodes",
    log: logging.Logger | None = None,
) -> CollectedDatasets:
    active_logger = log or logger
    target_episodes = int(target_episodes)
    if target_episodes <= 0:
        raise ValueError(f"{max_episodes_name} must be positive, got {target_episodes}.")
    initial_completed_episodes = int(initial_completed_episodes)
    if initial_completed_episodes < 0:
        raise ValueError(f"initial_completed_episodes must be non-negative, got {initial_completed_episodes}.")

    collected_datasets: CollectedDatasets = {}
    completed_episodes = initial_completed_episodes
    rollout_idx = 0
    while completed_episodes < target_episodes:
        _rollout_output, collected_datasets, metrics = cluster.rollout()
        collected_dataset = collected_datasets.get("collected_dataset")
        if collected_dataset is None:
            active_logger.info(
                "Finished %s %s without completed trajectories: collected_episodes=%s/%s, dataset_keys=%s, metrics=%s",
                log_prefix,
                rollout_idx,
                completed_episodes,
                target_episodes,
                list(collected_datasets.keys()),
                metrics,
            )
            rollout_idx += 1
            continue

        previous_completed_episodes = completed_episodes
        new_completed_episodes = count_lerobot_episodes(collected_dataset["root"])
        completed_episodes = initial_completed_episodes + new_completed_episodes
        if completed_episodes > target_episodes:
            dataset_root = collected_datasets["collected_dataset"]["root"]
            truncate_lerobot_episodes(dataset_root, target_episodes - initial_completed_episodes)
            new_completed_episodes = target_episodes - initial_completed_episodes
            completed_episodes = target_episodes
        active_logger.info(
            "Finished %s %s: collected_episodes=%s/%s, new_episodes=%s, metrics=%s",
            log_prefix,
            rollout_idx,
            completed_episodes,
            target_episodes,
            new_completed_episodes,
            metrics,
        )
        if completed_episodes <= previous_completed_episodes:
            active_logger.info("%s %s did not add completed trajectories; continuing.", log_prefix, rollout_idx)
            rollout_idx += 1
            continue
        rollout_idx += 1

    return collected_datasets
