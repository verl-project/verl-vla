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

from __future__ import annotations

import queue
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, SequentialSampler, Subset
from tqdm import tqdm

from verl_vla.models.recap_value_critic import ReCapValueCriticForPrediction
from verl_vla.trainer.recap.compute_return import (
    RECAP_ADVANTAGE_FIELD,
    RECAP_INDICATOR_FIELD,
    RECAP_RETURN_FIELD,
    RECAP_VALUE_FIELD,
)
from verl_vla.utils.data import dataloader_batch_to_dataproto
from verl_vla.utils.dataloader import LeRobotDataLoaderConfig
from verl_vla.utils.dataloader.lerobot import build_lerobot_dataset
from verl_vla.utils.dtype import precision_to_torch_dtype
from verl_vla.utils.lerobot import iter_lerobot_frame_records, write_lerobot_frame_columns


@dataclass(frozen=True)
class _ValueInferConfig:
    device: torch.device
    torch_dtype: torch.dtype
    precision: str
    num_gpus: int
    n_step: int
    positive_ratio: float
    force_intervention_positive: bool

    @classmethod
    def from_config(cls, config: DictConfig) -> _ValueInferConfig:
        infer_cfg = OmegaConf.select(config, "recap.value_infer", default={})
        precision = str(infer_cfg.get("precision", "bfloat16"))
        return cls(
            device=torch.device(
                str(infer_cfg.get("device", OmegaConf.select(config, "trainer.device", default="cuda")))
            ),
            torch_dtype=precision_to_torch_dtype(precision),
            precision=precision,
            num_gpus=int(infer_cfg.get("num_gpus", 1)),
            n_step=int(infer_cfg.get("n_step", 10)),
            positive_ratio=float(infer_cfg.get("positive_ratio", 0.3)),
            force_intervention_positive=bool(infer_cfg.get("force_intervention_positive", True)),
        )


def infer_recap_values(config: DictConfig, dataset: dict[str, str | Path], model_path: str | Path) -> dict[str, float]:
    infer_cfg = _ValueInferConfig.from_config(config)
    dataset_root = Path(dataset["root"])
    repo_id = str(dataset["repo_id"])
    data_config: LeRobotDataLoaderConfig = instantiate(config.recap.value_infer.data)
    dataset_size = len(
        build_lerobot_dataset(
            data_config,
            repo_id=repo_id,
            root=str(dataset_root),
        )
    )

    value_lookup = _infer_value_lookup(
        config=config,
        dataset=dataset,
        model_path=model_path,
        infer_cfg=infer_cfg,
    )
    value_stats = _write_recap_value_columns(
        dataset_root=dataset_root,
        value_lookup=value_lookup,
        n_step=infer_cfg.n_step,
        positive_ratio=infer_cfg.positive_ratio,
        force_intervention_positive=infer_cfg.force_intervention_positive,
    )
    return {
        "recap/value_infer_num_samples": float(len(value_lookup)),
        "recap/value_infer_dataset_frames": float(dataset_size),
        "recap/value_infer_repo_id": repo_id,
        **value_stats,
    }


# ---------------------------------------------------------------------------
# Value prediction
# ---------------------------------------------------------------------------


def _infer_value_lookup(
    *,
    config: DictConfig,
    dataset: dict[str, str | Path],
    model_path: str | Path,
    infer_cfg: _ValueInferConfig,
) -> dict[int, np.float32]:
    if infer_cfg.device.type != "cuda" or infer_cfg.num_gpus <= 1:
        data_config: LeRobotDataLoaderConfig = instantiate(config.recap.value_infer.data)
        lerobot_dataset = build_lerobot_dataset(
            data_config,
            repo_id=str(dataset["repo_id"]),
            root=str(dataset["root"]),
        )
        return _infer_value_lookup_on_device(
            lerobot_dataset=lerobot_dataset,
            data_config=data_config,
            model_path=model_path,
            infer_cfg=infer_cfg,
            device=infer_cfg.device,
            rank=0,
            world_size=1,
        )

    return _infer_value_lookup_multi_gpu(
        config=config,
        dataset=dataset,
        model_path=model_path,
        infer_cfg=infer_cfg,
    )


def _infer_value_lookup_multi_gpu(
    *,
    config: DictConfig,
    dataset: dict[str, str | Path],
    model_path: str | Path,
    infer_cfg: _ValueInferConfig,
) -> dict[int, np.float32]:
    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        raise RuntimeError("recap.value_infer.num_gpus > 1 requires CUDA devices.")

    world_size = min(infer_cfg.num_gpus, available_gpus)
    if world_size < infer_cfg.num_gpus:
        print(
            f"Requested {infer_cfg.num_gpus} value infer GPUs but only {available_gpus} are visible; "
            f"using {world_size}."
        )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    config_container = OmegaConf.to_container(config, resolve=True)
    dataset_container = {key: str(value) for key, value in dataset.items()}
    processes = [
        ctx.Process(
            target=_infer_value_shard_worker,
            args=(
                rank,
                world_size,
                config_container,
                dataset_container,
                str(model_path),
                infer_cfg.precision,
                result_queue,
            ),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()

    try:
        return _collect_worker_results(processes, result_queue)
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise


def _collect_worker_results(processes: list[mp.Process], result_queue) -> dict[int, np.float32]:
    value_lookup: dict[int, np.float32] = {}
    errors: list[str] = []
    received = 0

    while received < len(processes):
        try:
            rank, shard_lookup, error = result_queue.get(timeout=5.0)
        except queue.Empty:
            dead_without_result = [process for process in processes if process.exitcode not in (None, 0)]
            if dead_without_result:
                errors.extend(
                    f"worker pid={process.pid} exited with code {process.exitcode}" for process in dead_without_result
                )
                break
            continue

        received += 1
        if error is not None:
            errors.append(f"rank {rank}: {error}")
        else:
            value_lookup.update({int(index): np.float32(value) for index, value in shard_lookup.items()})

    for process in processes:
        process.join()
        if process.exitcode != 0:
            errors.append(f"worker pid={process.pid} exited with code {process.exitcode}")

    if errors:
        raise RuntimeError("Multi-GPU value inference failed:\n" + "\n".join(errors))
    return value_lookup


def _infer_value_shard_worker(
    rank: int,
    world_size: int,
    config_container: dict[str, Any],
    dataset: dict[str, str | Path],
    model_path: str,
    precision: str,
    result_queue,
) -> None:
    try:
        config = OmegaConf.create(config_container)
        infer_cfg = _ValueInferConfig(
            device=torch.device(f"cuda:{rank}"),
            torch_dtype=precision_to_torch_dtype(precision),
            precision=precision,
            num_gpus=world_size,
            n_step=0,
            positive_ratio=0.0,
            force_intervention_positive=False,
        )
        torch.cuda.set_device(infer_cfg.device)
        data_config: LeRobotDataLoaderConfig = instantiate(config.recap.value_infer.data)
        lerobot_dataset = build_lerobot_dataset(
            data_config,
            repo_id=str(dataset["repo_id"]),
            root=str(dataset["root"]),
        )
        shard_lookup = _infer_value_lookup_on_device(
            lerobot_dataset=lerobot_dataset,
            data_config=data_config,
            model_path=model_path,
            infer_cfg=infer_cfg,
            device=infer_cfg.device,
            rank=rank,
            world_size=world_size,
        )
        result_queue.put((rank, shard_lookup, None))
    except Exception:
        result_queue.put((rank, {}, traceback.format_exc()))


def _infer_value_lookup_on_device(
    *,
    lerobot_dataset,
    data_config: LeRobotDataLoaderConfig,
    model_path: str | Path,
    infer_cfg: _ValueInferConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[int, np.float32]:
    # STEP 1: build dataloader for the assigned shard of the dataset.
    if world_size > 1:
        infer_dataset = Subset(lerobot_dataset, range(rank, len(lerobot_dataset), world_size))
    else:
        infer_dataset = lerobot_dataset
    loader = DataLoader(
        dataset=infer_dataset,
        batch_size=data_config.batch_size,
        sampler=SequentialSampler(infer_dataset),
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory and device.type == "cuda",
        persistent_workers=data_config.persistent_workers if data_config.num_workers > 0 else False,
        prefetch_factor=data_config.prefetch_factor if data_config.num_workers > 0 else None,
    )

    # STEP 2: load the value model and run inference on the assigned shard.
    model = ReCapValueCriticForPrediction.from_pretrained(
        str(model_path),
        torch_dtype=infer_cfg.torch_dtype,
        trust_remote_code=True,
    )
    model.to(device=device)
    model.eval()

    # STEP 3: run inference and collect the value predictions into a lookup table.
    value_lookup: dict[int, np.float32] = {}
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"Value infer rank {rank}",
            total=len(loader),
            position=rank,
            leave=rank == 0,
            dynamic_ncols=True,
        ):
            batch_proto = dataloader_batch_to_dataproto(batch).to(device)
            with torch.autocast(device_type=device.type, dtype=infer_cfg.torch_dtype, enabled=device.type == "cuda"):
                values = model(batch_proto, tokenizer=None)

            indices = batch_proto.batch["index"].detach().cpu().numpy().astype(np.int64, copy=False)
            values_np = values.detach().float().cpu().numpy().astype(np.float32, copy=False)
            for index, value in zip(indices, values_np, strict=True):
                value_lookup[int(index)] = np.float32(value)
    return value_lookup


# ---------------------------------------------------------------------------
# Writeback and ACP labels
# ---------------------------------------------------------------------------


def _write_recap_value_columns(
    *,
    dataset_root: Path,
    value_lookup: dict[int, np.float32],
    n_step: int,
    positive_ratio: float,
    force_intervention_positive: bool,
) -> dict[str, float]:
    if n_step <= 0:
        raise ValueError("recap.value_infer.n_step must be positive.")
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("recap.value_infer.positive_ratio must be within [0, 1].")

    annotations = _compute_recap_annotations(
        dataset_root=dataset_root,
        value_lookup=value_lookup,
        n_step=n_step,
        positive_ratio=positive_ratio,
        force_intervention_positive=force_intervention_positive,
    )
    write_lerobot_frame_columns(
        dataset_root,
        columns_by_index={
            RECAP_VALUE_FIELD: {index: annotation["value"] for index, annotation in annotations.items()},
            RECAP_ADVANTAGE_FIELD: {index: annotation["advantage"] for index, annotation in annotations.items()},
            RECAP_INDICATOR_FIELD: {index: annotation["indicator"] for index, annotation in annotations.items()},
        },
        dtypes={
            RECAP_VALUE_FIELD: np.dtype(np.float32),
            RECAP_ADVANTAGE_FIELD: np.dtype(np.float32),
            RECAP_INDICATOR_FIELD: np.dtype(np.int64),
        },
    )

    advantages_np = np.asarray([item["advantage"] for item in annotations.values()], dtype=np.float32)
    indicators_np = np.asarray([item["indicator"] for item in annotations.values()], dtype=np.float32)
    return {
        "recap/advantage_mean": float(np.mean(advantages_np)),
        "recap/advantage_std": float(np.std(advantages_np)),
        "recap/indicator_positive_ratio": float(np.mean(indicators_np)),
    }


def _compute_recap_annotations(
    *,
    dataset_root: Path,
    value_lookup: dict[int, np.float32],
    n_step: int,
    positive_ratio: float,
    force_intervention_positive: bool,
) -> dict[int, dict[str, np.float32 | np.int64]]:
    records = _load_recap_records(dataset_root=dataset_root, value_lookup=value_lookup)
    rewards = _compute_dense_rewards(records)
    advantages = _compute_n_step_advantages(records, rewards, n_step=n_step)
    indicators = _binarize_advantages(
        records=records,
        advantages=advantages,
        positive_ratio=positive_ratio,
        force_intervention_positive=force_intervention_positive,
    )

    return {
        int(record["index"]): {
            "value": np.float32(record["value"]),
            "advantage": np.float32(advantage),
            "indicator": np.int64(indicator),
        }
        for record, advantage, indicator in zip(records, advantages, indicators, strict=True)
    }


def _load_recap_records(
    *,
    dataset_root: Path,
    value_lookup: dict[int, np.float32],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    frame_records = iter_lerobot_frame_records(
        dataset_root,
        columns=[
            "index",
            "episode_index",
            "frame_index",
            RECAP_RETURN_FIELD,
        ],
        optional_columns={
            "task_index": 0,
            "info.is_intervention": False,
        },
    )
    for frame_record in frame_records:
        index = int(frame_record["index"])
        records.append(
            {
                "index": index,
                "episode_index": int(frame_record["episode_index"]),
                "frame_index": int(frame_record["frame_index"]),
                "task_index": int(frame_record["task_index"]),
                "return": float(np.asarray(frame_record[RECAP_RETURN_FIELD]).reshape(-1)[0]),
                "value": float(value_lookup[index]),
                "is_intervention": bool(frame_record["info.is_intervention"]),
            }
        )

    records.sort(key=lambda item: (item["episode_index"], item["frame_index"], item["index"]))
    return records


def _compute_dense_rewards(records: list[dict[str, object]]) -> np.ndarray:
    rewards = np.zeros(len(records), dtype=np.float32)
    for i, record in enumerate(records):
        has_next = i + 1 < len(records)
        next_in_episode = (
            has_next
            and records[i + 1]["episode_index"] == record["episode_index"]
            and int(records[i + 1]["frame_index"]) == int(record["frame_index"]) + 1
        )
        if next_in_episode:
            rewards[i] = float(record["return"]) - float(records[i + 1]["return"])
        else:
            rewards[i] = float(record["return"])
    return rewards


def _compute_n_step_advantages(
    records: list[dict[str, object]],
    rewards: np.ndarray,
    *,
    n_step: int,
) -> np.ndarray:
    advantages = np.zeros(len(records), dtype=np.float32)
    for i, record in enumerate(records):
        reward_sum, next_index = _sum_n_step_rewards(records=records, rewards=rewards, start_index=i, n_step=n_step)
        bootstrap = _next_value_if_contiguous(records=records, current=record, next_index=next_index, n_step=n_step)
        advantages[i] = np.float32(reward_sum + bootstrap - float(record["value"]))
    return advantages


def _sum_n_step_rewards(
    *,
    records: list[dict[str, object]],
    rewards: np.ndarray,
    start_index: int,
    n_step: int,
) -> tuple[float, int]:
    current = records[start_index]
    reward_sum = 0.0
    next_index = start_index
    steps = 0
    while steps < n_step and next_index < len(records):
        record = records[next_index]
        same_episode = record["episode_index"] == current["episode_index"]
        contiguous = int(record["frame_index"]) == int(current["frame_index"]) + steps
        if not same_episode or not contiguous:
            break
        reward_sum += float(rewards[next_index])
        steps += 1
        next_index += 1
    return reward_sum, next_index


def _next_value_if_contiguous(
    *,
    records: list[dict[str, object]],
    current: dict[str, object],
    next_index: int,
    n_step: int,
) -> float:
    if (
        next_index < len(records)
        and records[next_index]["episode_index"] == current["episode_index"]
        and int(records[next_index]["frame_index"]) == int(current["frame_index"]) + n_step
    ):
        return float(records[next_index]["value"])
    return 0.0


def _binarize_advantages(
    *,
    records: list[dict[str, object]],
    advantages: np.ndarray,
    positive_ratio: float,
    force_intervention_positive: bool,
) -> np.ndarray:
    task_indices = np.asarray([record["task_index"] for record in records], dtype=np.int64)
    indicators = np.zeros(len(records), dtype=np.int64)

    for task_idx in np.unique(task_indices):
        task_mask = task_indices == task_idx
        task_advantages = advantages[task_mask]
        if task_advantages.size == 0:
            continue
        positive_count = int(np.ceil(float(task_advantages.size) * positive_ratio))
        if positive_count <= 0:
            continue
        task_indicators = np.zeros(task_advantages.size, dtype=np.int64)
        positive_order = np.argsort(task_advantages)[-positive_count:]
        task_indicators[positive_order] = 1
        indicators[task_mask] = task_indicators

    if force_intervention_positive:
        interventions = np.asarray([record["is_intervention"] for record in records], dtype=bool)
        indicators[interventions] = 1

    return indicators
