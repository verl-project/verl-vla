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

import logging

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from .config import LeRobotDataLoaderConfig

logger = logging.getLogger(__name__)


def resolve_multiprocessing_context(data_config: LeRobotDataLoaderConfig, num_workers: int) -> str | None:
    if num_workers <= 0:
        return None
    if data_config.multiprocessing_context is not None:
        return data_config.multiprocessing_context

    try:
        import ray
    except ImportError:
        return None

    if ray.is_initialized():
        # Ray actor handles created before DataLoader construction can be
        # inherited by forked workers, and we observed workers hang while
        # cleaning up those inherited handles. forkserver keeps multiprocessing
        # workers isolated from the trainer's live Ray runtime state.
        return "forkserver"
    return None


def build_lerobot_dataset(
    data_config: LeRobotDataLoaderConfig,
    *,
    repo_id: str | None = None,
    root: str | None = None,
    delta_timestamps: dict[str, list[float]] | None = None,
):
    return LeRobotDataset(
        repo_id=repo_id or data_config.repo_id,
        root=root if root is not None else data_config.root,
        revision=data_config.revision,
        video_backend=data_config.video_backend,
        delta_timestamps=delta_timestamps,
    )


def build_lerobot_sft_dataset(data_config: LeRobotDataLoaderConfig):
    action_delta_steps = int(data_config.action_delta_steps)
    delta_timestamps = None
    if action_delta_steps > 0:
        probe_dataset = build_lerobot_dataset(data_config)
        delta_timestamps = {data_config.action_key: [t / probe_dataset.fps for t in range(action_delta_steps)]}
    return build_lerobot_dataset(data_config, delta_timestamps=delta_timestamps)


def build_lerobot_sft_dataloader(
    data_config: LeRobotDataLoaderConfig,
    *,
    train_world_size: int = 1,
) -> StatefulDataLoader:
    dataset = build_lerobot_sft_dataset(data_config)
    batch_size = int(data_config.batch_size)
    drop_last = bool(data_config.drop_last)
    if train_world_size > 1 and batch_size % train_world_size != 0:
        raise ValueError(
            "data.batch_size must be divisible by trainer world size for DataProto dispatch: "
            f"batch_size={batch_size}, world_size={train_world_size}."
        )
    if train_world_size > 1 and not drop_last:
        logger.warning(
            "Forcing data.drop_last=True because distributed DataProto dispatch requires every batch "
            "to be divisible by trainer world size=%s.",
            train_world_size,
        )
        drop_last = True

    dataset_size = len(dataset)
    if drop_last and dataset_size < batch_size:
        raise ValueError(
            "SFT dataset is smaller than one batch with drop_last=True; no batches can be produced. "
            f"dataset_size={dataset_size}, batch_size={batch_size}, repo_id={data_config.repo_id}, "
            f"root={data_config.root}. Reduce data.batch_size or set drop_last=False for single-GPU runs."
        )
    logger.info(
        "Created SFT dataloader: dataset_size=%s, batch_size=%s, drop_last=%s, steps_per_epoch=%s, repo_id=%s, root=%s",
        dataset_size,
        batch_size,
        drop_last,
        dataset_size // batch_size if drop_last else (dataset_size + batch_size - 1) // batch_size,
        data_config.repo_id,
        data_config.root,
    )

    if data_config.shuffle:
        generator = torch.Generator()
        if data_config.seed is not None:
            generator.manual_seed(int(data_config.seed))
        sampler = RandomSampler(data_source=dataset, generator=generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    num_workers = int(data_config.num_workers)
    multiprocessing_context = resolve_multiprocessing_context(data_config, num_workers)
    return StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        sampler=sampler,
        pin_memory=bool(data_config.pin_memory),
        persistent_workers=bool(data_config.persistent_workers) if num_workers > 0 else False,
        multiprocessing_context=multiprocessing_context,
        prefetch_factor=int(data_config.prefetch_factor) if num_workers > 0 else None,
    )


class RLPDTransitionDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        action_chunk_steps: int,
        transition_stride: int = 1,
        include_final_transition: bool = True,
        root: str | None = None,
        revision: str | None = None,
        video_backend: str | None = None,
        episodes: list[int] | None = None,
        max_transitions: int = 0,
    ):
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            revision=revision,
            video_backend=video_backend,
        )
        self.records = list(
            iter_lerobot_transition_records(
                self.dataset,
                action_chunk_steps=action_chunk_steps,
                transition_stride=transition_stride,
                include_final_transition=include_final_transition,
                episodes=episodes,
                max_transitions=max_transitions,
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        start = int(record["start"])
        next_index = int(record["next"])
        return {
            "start": start,
            "next": next_index,
            "terminal": bool(record["terminal"]),
            "episode_success": bool(record["episode_success"]),
            "t0_item": self.dataset[start],
            "t1_item": self.dataset[next_index],
        }


def iter_lerobot_transition_records(
    dataset,
    action_chunk_steps: int,
    transition_stride: int = 1,
    include_final_transition: bool = True,
    episodes: list[int] | None = None,
    max_transitions: int = 0,
):
    transition_stride = int(transition_stride)
    if transition_stride <= 0:
        raise ValueError(f"transition_stride must be positive, got {transition_stride}")
    selected_episodes = set(int(episode) for episode in episodes) if episodes is not None else None
    transition_window = action_chunk_steps * 2
    emitted = 0

    def as_bool(value) -> bool:
        return bool(value.item()) if torch.is_tensor(value) else bool(value)

    for episode in dataset.meta.episodes:
        episode_index = int(episode["episode_index"])
        if selected_episodes is not None and episode_index not in selected_episodes:
            continue

        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        episode_length = end - start
        episode_success = as_bool(dataset.hf_dataset["next.success"][end - 1]) if episode_length > 0 else False
        num_transitions = end - start - transition_window + 1
        if num_transitions > 0:
            for transition_offset in range(0, num_transitions, transition_stride):
                if max_transitions > 0 and emitted >= max_transitions:
                    return
                transition_start = start + transition_offset
                yield {
                    "start": transition_start,
                    "next": transition_start + action_chunk_steps,
                    "terminal": False,
                    "episode_success": episode_success,
                }
                emitted += 1

        if include_final_transition and episode_length >= action_chunk_steps:
            if max_transitions > 0 and emitted >= max_transitions:
                return
            transition_start = end - action_chunk_steps
            yield {
                "start": transition_start,
                "next": transition_start,
                "terminal": True,
                "episode_success": episode_success,
            }
            emitted += 1
