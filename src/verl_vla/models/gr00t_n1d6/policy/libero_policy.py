# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LIBERO adapter for the external GR00T N1.6 policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from verl import DataProto

from .base import Gr00tPolicyInput, Gr00tPolicyOutput

LIBERO_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
LIBERO_STATE_SLICES = {
    "x": (0, 1),
    "y": (1, 2),
    "z": (2, 3),
    "roll": (3, 4),
    "pitch": (4, 5),
    "yaw": (5, 6),
    "gripper": (6, 8),
}
LIBERO_ACTION_SLICES = {key: (index, index + 1) for index, key in enumerate(LIBERO_KEYS)}


def _stats_for_slice(flat_stats: dict[str, Any], start: int, end: int) -> dict[str, list[float]]:
    required = ("min", "max", "mean", "std", "q01", "q99")
    missing = [name for name in required if name not in flat_stats]
    if missing:
        raise ValueError(f"Normalization statistics are missing {missing}.")
    return {name: [float(value) for value in flat_stats[name][start:end]] for name in required}


def load_libero_statistics(path: str | Path) -> dict[str, Any]:
    """Load official nested stats or convert flat LIBERO state/action stats."""
    stats_path = Path(path).expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(f"NORM_STATS_PATH does not exist: {stats_path}")
    with stats_path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if "libero_panda" in raw:
        return {"libero_panda": raw["libero_panda"]}

    modality_slices = {
        "state": LIBERO_STATE_SLICES,
        "action": LIBERO_ACTION_SLICES,
    }
    for modality, slices in modality_slices.items():
        if modality not in raw:
            raise ValueError(f"Normalization statistics do not contain '{modality}': {stats_path}")
        lengths = {name: len(raw[modality].get(name, [])) for name in ("min", "max", "mean", "std", "q01", "q99")}
        expected_dim = max(end for _, end in slices.values())
        if any(length != expected_dim for length in lengths.values()):
            raise ValueError(f"Expected {expected_dim} {modality} statistics in {stats_path}, got {lengths}.")

    return {
        "libero_panda": {
            modality: {
                key: _stats_for_slice(raw[modality], start, end)
                for key, (start, end) in modality_slices[modality].items()
            }
            for modality in modality_slices
        }
    }


def image_to_uint8_hwc(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert one CHW/HWC image in [0, 1] or [0, 255] to uint8 HWC."""
    image = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected one image with three dimensions, got {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image in CHW or HWC layout, got {image.shape}.")
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def prepare_libero_gripper_action(action: np.ndarray) -> np.ndarray:
    """Convert GR00T's [0, 1] gripper value to LIBERO's inverted {-1, 1} command."""
    prepared = np.asarray(action, dtype=np.float32).copy()
    prepared[..., -1] = -np.sign(2.0 * prepared[..., -1] - 1.0)
    return prepared


def load_gr00t_processor(model_path: str, norm_stats_path: str | None, *, training: bool):
    """Load the official processor without making GR00T a verl-vla dependency."""
    from verl_vla.models.register_vla_models import register_gr00t_n1d6_model

    register_gr00t_n1d6_model(required=True)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

    processor = Gr00tN1d6Processor.from_pretrained(
        model_path,
        modality_configs={"libero_panda": MODALITY_CONFIGS["libero_panda"]},
        use_relative_action=False,
    )
    if norm_stats_path:
        processor.set_statistics(load_libero_statistics(norm_stats_path), override=True)
    processor.train() if training else processor.eval()
    return processor


def _numpy_float32(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _make_vla_step(
    image: torch.Tensor | np.ndarray,
    wrist_image: torch.Tensor | np.ndarray,
    state: torch.Tensor | np.ndarray,
    task: Any,
    action: torch.Tensor | np.ndarray | None = None,
):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import VLAStepData

    state_array = _numpy_float32(state).reshape(-1)
    expected_state_dim = max(end for _, end in LIBERO_STATE_SLICES.values())
    if state_array.shape[0] != expected_state_dim:
        raise ValueError(f"Expected {expected_state_dim} LIBERO state values, got {state_array.shape}.")
    states = {key: state_array[start:end].reshape(1, end - start) for key, (start, end) in LIBERO_STATE_SLICES.items()}

    actions: dict[str, np.ndarray] = {}
    if action is not None:
        action_array = _numpy_float32(action)
        if action_array.ndim == 1:
            action_array = action_array[None, :]
        expected_action_dim = max(end for _, end in LIBERO_ACTION_SLICES.values())
        if action_array.ndim != 2 or action_array.shape[-1] != expected_action_dim:
            raise ValueError(f"Expected LIBERO actions shaped [T, 7], got {action_array.shape}.")
        actions = {key: action_array[:, start:end] for key, (start, end) in LIBERO_ACTION_SLICES.items()}

    return VLAStepData(
        images={
            "image": [image_to_uint8_hwc(image)],
            "wrist_image": [image_to_uint8_hwc(wrist_image)],
        },
        states=states,
        actions=actions,
        text=str(task),
        embodiment=EmbodimentTag.LIBERO_PANDA,
    )


def _process_vla_step(
    processor,
    step,
    action_valid_mask: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    from gr00t.data.types import MessageType

    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    if action_valid_mask is not None and "action_mask" in processed:
        valid = torch.as_tensor(action_valid_mask, dtype=torch.bool).reshape(-1)
        mask = torch.as_tensor(processed["action_mask"]).clone()
        length = min(mask.shape[0], valid.shape[0])
        invalid_indices = torch.nonzero(~valid[:length], as_tuple=False).flatten()
        mask[invalid_indices] = 0
        processed["action_mask"] = mask
    return {
        key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in processed.items()
    }


class LiberoGr00tInput(Gr00tPolicyInput):
    """Raw shared VLA input represented as official LIBERO ``VLAStepData`` objects."""

    def __init__(self, steps: list[Any], raw_states: list[dict[str, np.ndarray]]):
        self.steps = steps
        self.raw_states = raw_states

    @classmethod
    def from_data_proto(
        cls,
        obs: DataProto,
        actions: torch.Tensor | None = None,
    ) -> LiberoGr00tInput:
        images = obs.batch["observation.images.image"]
        wrist_images = obs.batch["observation.images.wrist_image"]
        states = obs.batch["observation.state"]
        tasks = obs.non_tensor_batch["task"]

        steps = []
        raw_states = []
        for index in range(len(obs)):
            step = _make_vla_step(
                images[index],
                wrist_images[index],
                states[index],
                tasks[index],
                None if actions is None else actions[index],
            )
            steps.append(step)
            raw_states.append(step.states)
        return cls(steps=steps, raw_states=raw_states)

    def collate(
        self,
        processor: Any,
        *,
        action_valid_mask: torch.Tensor | None = None,
    ) -> Any:
        processed = [
            _process_vla_step(
                processor,
                step,
                None if action_valid_mask is None else action_valid_mask[index],
            )
            for index, step in enumerate(self.steps)
        ]
        return processor.collator(processed)


class LiberoGr00tOutput(Gr00tPolicyOutput):
    """Decode GR00T actions and apply the official LIBERO gripper semantics."""

    @classmethod
    def from_model_output(
        cls,
        model_output: dict[str, Any],
        *,
        processor: Any,
        policy_input: LiberoGr00tInput,
        action_chunk_size: int,
        device: torch.device | str,
    ) -> LiberoGr00tOutput:
        from gr00t.data.embodiment_tags import EmbodimentTag

        normalized_action = model_output["action_pred"].float().cpu().numpy()
        batched_states = {
            key: np.stack([state[key] for state in policy_input.raw_states], axis=0) for key in LIBERO_KEYS
        }
        decoded = processor.decode_action(
            normalized_action,
            EmbodimentTag.LIBERO_PANDA,
            batched_states,
        )
        full_action = np.concatenate([decoded[key] for key in LIBERO_KEYS], axis=-1)
        full_action = prepare_libero_gripper_action(full_action)
        action = torch.from_numpy(full_action[:, :action_chunk_size]).to(device=device)
        return cls(action)


__all__ = [
    "LIBERO_ACTION_SLICES",
    "LIBERO_KEYS",
    "LIBERO_STATE_SLICES",
    "LiberoGr00tInput",
    "LiberoGr00tOutput",
    "image_to_uint8_hwc",
    "load_gr00t_processor",
    "load_libero_statistics",
    "prepare_libero_gripper_action",
]
