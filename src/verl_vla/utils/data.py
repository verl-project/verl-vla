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

from verl_vla.workers.actor_critic.base import ACTION_KEY, OBS_KEY


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
    batch_size, num_steps = data.batch["action.action"].shape[:2]
    new_batch_fields = {}
    new_non_tensor_fields = {}

    for key, tensor in data.batch.items():
        if len(tensor.shape) >= 2 and tensor.shape[0] == batch_size and tensor.shape[1] == num_steps:
            new_batch_fields[key] = tensor.reshape(batch_size * num_steps, *tensor.shape[2:])
        elif len(tensor.shape) == 1 and tensor.shape[0] == batch_size:
            new_batch_fields[key] = tensor.repeat_interleave(num_steps)
        else:
            new_batch_fields[key] = tensor

    for key, array in data.non_tensor_batch.items():
        if array.ndim >= 2 and array.shape[0] == batch_size and array.shape[1] == num_steps:
            new_non_tensor_fields[key] = array.reshape(batch_size * num_steps, *array.shape[2:])
        elif array.ndim == 1 and array.shape[0] == batch_size:
            new_non_tensor_fields[key] = np.repeat(array, num_steps, axis=0)
        else:
            new_non_tensor_fields[key] = array

    return DataProto.from_dict(
        tensors=new_batch_fields,
        non_tensors=new_non_tensor_fields,
        meta_info=data.meta_info,
    )


def add_transition_prefixes(data: DataProto) -> DataProto:
    batch = data.batch
    non_tensor_batch = data.non_tensor_batch
    step_key = "action.full_action" if "action.full_action" in batch else "action.action"
    if step_key not in batch:
        return data

    num_steps = batch[step_key].shape[1]
    if num_steps <= 1:
        return data

    def slice_steps(x, start: int, end: int | None):
        return x[:, slice(start, end), ...]

    obs_prefix = f"{OBS_KEY}."
    action_prefix = f"{ACTION_KEY}."
    keys = [key for key in batch.keys() if key.startswith(obs_prefix) or key.startswith(action_prefix)]
    non_tensor_keys = [
        key for key in non_tensor_batch.keys() if key.startswith(obs_prefix) or key.startswith(action_prefix)
    ]

    for key in keys:
        batch[f"t0.{key}"] = slice_steps(batch[key], 0, -1)
        batch[f"t1.{key}"] = slice_steps(batch[key], 1, None)

    for key in non_tensor_keys:
        non_tensor_batch[f"t0.{key}"] = slice_steps(non_tensor_batch[key], 0, -1)
        non_tensor_batch[f"t1.{key}"] = slice_steps(non_tensor_batch[key], 1, None)

    batch_size = batch[step_key].shape[0]
    for key, tensor in list(batch.items()):
        if tensor.ndim >= 2 and tensor.shape[0] == batch_size and tensor.shape[1] == num_steps:
            batch[key] = slice_steps(tensor, 0, -1)

    for key, array in list(non_tensor_batch.items()):
        if array.ndim >= 2 and array.shape[0] == batch_size and array.shape[1] == num_steps:
            non_tensor_batch[key] = slice_steps(array, 0, -1)

    return data
