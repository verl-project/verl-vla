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

import numpy as np
import torch
from verl import DataProto

from verl_vla.utils.keys import ACTION_KEY, OBS_KEY


def reduce_substep_dims(value: torch.Tensor, *, reduction: str) -> torch.Tensor:
    """Reduce chunk/substep dimensions while preserving batch and rollout time."""
    if value.ndim <= 2:
        return value

    while value.ndim > 2:
        if reduction == "any":
            value = value.any(dim=-1)
        elif reduction == "sum":
            value = value.sum(dim=-1)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    return value


def _build_sac_transition_masks(
    done_steps: torch.Tensor, success_steps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build transition masks for rows that may contain zero, one, or many completed episodes."""
    valid_mask = torch.zeros_like(done_steps, dtype=torch.bool)
    success_mask = torch.zeros_like(done_steps, dtype=torch.bool)

    for batch_idx in range(done_steps.shape[0]):
        start_idx = 0
        done_indices = torch.nonzero(done_steps[batch_idx], as_tuple=False).flatten().tolist()
        for done_idx in done_indices:
            if done_idx < start_idx:
                continue

            segment = slice(start_idx, done_idx + 1)
            valid_mask[batch_idx, segment] = True

            if success_steps[batch_idx, segment].any():
                success_mask[batch_idx, segment] = True

            start_idx = done_idx + 1

    return valid_mask, success_mask


def update_progress_trajectory_counts(
    env_result: DataProto,
    *,
    stage_id: int,
    progress_counts: dict[str, int],
    progress_lane_state: dict[int, dict[str, torch.Tensor]],
) -> None:
    del stage_id, progress_lane_state

    batch = env_result.batch
    episode_done = batch["next.terminated"].bool() | batch["next.truncated"].bool()
    success = batch["next.success"].bool()
    first_done_idx = episode_done.float().argmax(dim=1)
    chunk_done = episode_done.any(dim=1)
    step_idx = torch.arange(episode_done.shape[1], device=episode_done.device).unsqueeze(0)
    success_before_done = (success & (step_idx <= first_done_idx.unsqueeze(1))).any(dim=1)

    progress_counts["done_eps"] += int(chunk_done.sum().item())
    progress_counts["succ_eps"] += int((chunk_done & success_before_done).sum().item())


def dataloader_batch_to_dataproto(batch: dict) -> DataProto:
    tensor_batch = {}
    non_tensor_batch = {}
    batch_size = None
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_batch[key] = value
            if batch_size is None:
                batch_size = value.shape[0]
        else:
            non_tensor_batch[key] = np.array(value, dtype=object)
            if batch_size is None and hasattr(value, "__len__"):
                batch_size = len(value)
    if batch_size is None:
        batch_size = 1
    return DataProto.from_dict(
        tensors=tensor_batch,
        non_tensors=non_tensor_batch,
        meta_info={"global_token_num": [0] * batch_size},
    )


def get_dataproto_from_prefix(data: DataProto, prefix: str, separator: str = "") -> DataProto:
    match_prefix = prefix if not separator or prefix.endswith(separator) else f"{prefix}{separator}"
    prefix_length = len(match_prefix)
    tensor_batch = {}
    non_tensor_batch = {}

    if data.batch is not None:
        for key, value in data.batch.items():
            if key.startswith(match_prefix):
                tensor_batch[key[prefix_length:]] = value

    for key, value in data.non_tensor_batch.items():
        if key.startswith(match_prefix):
            non_tensor_batch[key[prefix_length:]] = value

    return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=data.meta_info)


def get_dataproto_keys_by_prefix(data: DataProto, prefixes: tuple[str, ...]) -> tuple[list[str], list[str]]:
    def _key_has_prefix(key: str | tuple) -> bool:
        if isinstance(key, tuple):
            key_name = ".".join(str(part) for part in key)
        else:
            key_name = str(key)
        return key_name.startswith(prefixes)

    batch_keys = []
    if data.batch is not None:
        batch_keys = [key for key in data.batch.keys() if _key_has_prefix(key)]
    non_tensor_batch_keys = [key for key in data.non_tensor_batch.keys() if _key_has_prefix(key)]
    return batch_keys, non_tensor_batch_keys


def slice_dataproto_batch(data: DataProto, start: int, end: int) -> DataProto:
    return DataProto.from_dict(
        tensors={key: value[:, start:end] for key, value in data.batch.items()},
        meta_info=data.meta_info,
    )


def merge_nested_dicts_or_tuples(a: dict | tuple, b: dict | tuple) -> dict | tuple:
    if isinstance(a, dict) and isinstance(b, dict):
        return {key: merge_nested_dicts_or_tuples(a[key], b[key]) for key in a.keys()}
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(merge_nested_dicts_or_tuples(item_a, item_b) for item_a, item_b in zip(a, b, strict=False))
    return torch.cat([a, b], dim=0)


def split_nested_dicts_or_tuples(data: dict | tuple, split_num: int) -> list[dict | tuple]:
    if isinstance(data, torch.Tensor):
        return list(torch.chunk(data, split_num, dim=0))
    if isinstance(data, dict):
        split_dicts = [dict() for _ in range(split_num)]
        for key, value in data.items():
            split_values = split_nested_dicts_or_tuples(value, split_num)
            for i in range(split_num):
                split_dicts[i][key] = split_values[i]
        return split_dicts
    if isinstance(data, tuple):
        split_tuples = [list() for _ in range(split_num)]
        for item in data:
            split_items = split_nested_dicts_or_tuples(item, split_num)
            for i in range(split_num):
                split_tuples[i].append(split_items[i])
        return [tuple(split_tuple) for split_tuple in split_tuples]
    raise TypeError("Input data must be a torch.Tensor, dict, or tuple.")


def valid_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    x = x.squeeze(-1)
    valid_f = valid.float().to(x.device)
    denom = valid_f.sum().clamp_min(1.0)
    return (x * valid_f).sum() / denom


def stack_dataproto_with_padding(data_protos: list[DataProto], prefix: str) -> dict[str, torch.Tensor | np.ndarray]:
    merged = {}

    tensor_keys = sorted({key for data in data_protos for key in (data.batch.keys() if data.batch is not None else [])})
    for key in tensor_keys:
        template = next(data.batch[key] for data in data_protos if data.batch is not None and key in data.batch.keys())
        per_step_values = []
        for data in data_protos:
            if data.batch is not None and key in data.batch.keys():
                per_step_values.append(data.batch[key])
            else:
                per_step_values.append(torch.zeros_like(template))
        merged[f"{prefix}.{key}"] = torch.stack(per_step_values, dim=1)

    non_tensor_keys = sorted({key for data in data_protos for key in data.non_tensor_batch.keys()})
    for key in non_tensor_keys:
        template = next(data.non_tensor_batch[key] for data in data_protos if key in data.non_tensor_batch)
        per_step_values = []
        for data in data_protos:
            if key in data.non_tensor_batch:
                per_step_values.append(data.non_tensor_batch[key])
            else:
                per_step_values.append(np.zeros_like(template))
        merged[f"{prefix}.{key}"] = np.stack(per_step_values, axis=1)

    return merged


def flatten_trajectories(data: DataProto) -> DataProto:
    batch_size, num_steps = data.batch["t0.action.action"].shape[:2]
    new_batch_fields = {}
    new_non_tensor_fields = {}

    for key, tensor in data.batch.items():
        if len(tensor.shape) >= 2 and tensor.shape[0] == batch_size and tensor.shape[1] == num_steps:
            new_batch_fields[key] = tensor.reshape(batch_size * num_steps, *tensor.shape[2:])
        else:
            new_batch_fields[key] = tensor.repeat_interleave(num_steps)

    for key, array in data.non_tensor_batch.items():
        if array.ndim >= 2 and array.shape[0] == batch_size and array.shape[1] == num_steps:
            new_non_tensor_fields[key] = array.reshape(batch_size * num_steps, *array.shape[2:])
        else:
            new_non_tensor_fields[key] = np.repeat(array, num_steps, axis=0)

    return DataProto.from_dict(
        tensors=new_batch_fields,
        non_tensors=new_non_tensor_fields,
        meta_info=data.meta_info,
    )


def add_transition_prefixes(data: DataProto, transition_boundary_mask: torch.Tensor | None = None) -> DataProto:
    """Convert rollout slots into ``t0``/``t1`` transition fields.

    ``transition_boundary_mask`` marks steps whose next observation belongs to a
    new episode after auto reset. For those steps, ``t1.obs`` is replaced by the
    same step's ``t0.obs`` so the previous trajectory does not consume the reset
    observation from the next trajectory.

    Example with multiple trajectories in one batch row:

        obs:     o0  o1  o2  o3  o4  o5
        done:    F   T   F   F   T   F

        shift t1:
        t1.obs:  o1  o2  o3  o4  o5  o5

        fixed:
        t1.obs:  o1  o1  o3  o4  o4  o5
                     ^ done      ^ done
    """
    batch = data.batch
    non_tensor_batch = data.non_tensor_batch

    def next_steps(x):
        last_step = x[:, -1:, ...]
        if isinstance(x, torch.Tensor):
            return torch.cat([x[:, 1:, ...], last_step], dim=1)
        return np.concatenate([x[:, 1:, ...], last_step], axis=1)

    def replace_boundary_steps(next_value, current_value):
        if transition_boundary_mask is None:
            return next_value
        mask = transition_boundary_mask
        while mask.ndim < next_value.ndim:
            mask = mask.unsqueeze(-1)
        if isinstance(next_value, torch.Tensor):
            return torch.where(mask.to(device=next_value.device), current_value, next_value)
        return np.where(mask.cpu().numpy(), current_value, next_value)

    obs_prefix = f"{OBS_KEY}."
    action_prefix = f"{ACTION_KEY}."
    keys = [key for key in batch.keys() if key.startswith(obs_prefix) or key.startswith(action_prefix)]
    non_tensor_keys = [
        key for key in non_tensor_batch.keys() if key.startswith(obs_prefix) or key.startswith(action_prefix)
    ]

    for key in keys:
        batch[f"t0.{key}"] = batch[key]
        batch[f"t1.{key}"] = next_steps(batch[key])
        if key.startswith(obs_prefix):
            batch[f"t1.{key}"] = replace_boundary_steps(batch[f"t1.{key}"], batch[f"t0.{key}"])
        del batch[key]

    for key in non_tensor_keys:
        non_tensor_batch[f"t0.{key}"] = non_tensor_batch[key]
        non_tensor_batch[f"t1.{key}"] = next_steps(non_tensor_batch[key])
        if key.startswith(obs_prefix):
            non_tensor_batch[f"t1.{key}"] = replace_boundary_steps(
                non_tensor_batch[f"t1.{key}"], non_tensor_batch[f"t0.{key}"]
            )
        del non_tensor_batch[key]

    return data
