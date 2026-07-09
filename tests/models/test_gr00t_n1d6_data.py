# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_vla.models.gr00t_n1d6.policy.libero_policy import (
    LIBERO_KEYS,
    LiberoGr00tInput,
    LiberoGr00tOutput,
    image_to_uint8_hwc,
    load_libero_statistics,
    prepare_libero_gripper_action,
)

PACKAGE_DIR = Path(__file__).parents[2] / "src/verl_vla/models/gr00t_n1d6"


def test_source_commit_is_pinned():
    package_init = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "e29d8fc50b0e4745120ae3fb72447986fe638aa6" in package_init


def test_load_flat_libero_statistics(tmp_path):
    stats = {
        modality: {
            name: [float(index + offset) for index in range(8 if modality == "state" else 7)]
            for offset, name in enumerate(("min", "max", "mean", "std", "q01", "q99"))
        }
        for modality in ("state", "action")
    }
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")

    nested = load_libero_statistics(path)["libero_panda"]

    assert tuple(nested["state"]) == LIBERO_KEYS
    assert nested["state"]["x"]["min"] == [0.0]
    assert nested["state"]["gripper"]["min"] == [6.0, 7.0]
    assert nested["action"]["gripper"]["q99"] == [11.0]


def test_image_to_uint8_hwc_scales_chw_float_image():
    image = np.ones((3, 4, 5), dtype=np.float32) * 0.5

    converted = image_to_uint8_hwc(image)

    assert converted.shape == (4, 5, 3)
    assert converted.dtype == np.uint8
    assert np.all(converted == 127)


def test_flat_statistics_require_min_and_max(tmp_path):
    stats = {
        modality: {name: [0.0] * (8 if modality == "state" else 7) for name in ("mean", "std", "q01", "q99")}
        for modality in ("state", "action")
    }
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 8 state statistics"):
        load_libero_statistics(path)


def test_prepare_libero_gripper_action_matches_official_semantics():
    action = np.zeros((1, 3, 7), dtype=np.float32)
    action[..., -1] = np.array([0.1, 0.5, 0.9], dtype=np.float32)

    prepared = prepare_libero_gripper_action(action)

    np.testing.assert_array_equal(prepared[0, :, -1], np.array([1.0, 0.0, -1.0]))
    np.testing.assert_array_equal(
        action[0, :, -1],
        np.array([0.1, 0.5, 0.9], dtype=np.float32),
    )


def _raw_libero_batch() -> tuple[DataProto, torch.Tensor, torch.Tensor]:
    actions = torch.zeros((1, 16, 7), dtype=torch.float32)
    action_valid_mask = torch.ones((1, 16), dtype=torch.float32)
    action_valid_mask[:, -1] = 0
    obs = DataProto.from_dict(
        tensors={
            "observation.images.image": torch.zeros((1, 3, 8, 8)),
            "observation.images.wrist_image": torch.zeros((1, 3, 8, 8)),
            "observation.state": torch.arange(8, dtype=torch.float32).reshape(1, 8),
            "action": actions,
            "action_is_pad": ~action_valid_mask.bool(),
        },
        non_tensors={"task": np.asarray(["pick up the bowl"], dtype=object)},
    )
    return obs, actions, action_valid_mask


def test_libero_input_consumes_raw_dataproto_and_applies_valid_mask():
    pytest.importorskip("gr00t")
    obs, actions, action_valid_mask = _raw_libero_batch()

    policy_input = LiberoGr00tInput.from_data_proto(obs, actions=actions)

    assert tuple(policy_input.raw_states[0]) == LIBERO_KEYS
    assert policy_input.raw_states[0]["gripper"].shape == (1, 2)

    class FakeProcessor:
        def __call__(self, _messages):
            return {"action_mask": torch.ones((50, 128), dtype=torch.float32)}

        def collator(self, samples):
            return samples

    processed = policy_input.collate(
        FakeProcessor(),
        action_valid_mask=action_valid_mask,
    )
    assert processed[0]["action_mask"][14].all()
    assert not processed[0]["action_mask"][15].any()


def test_libero_output_decodes_and_prepares_gripper():
    pytest.importorskip("gr00t")
    obs, _, _ = _raw_libero_batch()
    policy_input = LiberoGr00tInput.from_data_proto(obs)

    class FakeProcessor:
        def decode_action(self, _normalized_action, _embodiment, _states):
            decoded = {key: np.zeros((1, 2, 1), dtype=np.float32) for key in LIBERO_KEYS}
            decoded["gripper"][0, :, 0] = np.array([0.2, 0.8], dtype=np.float32)
            return decoded

    output = LiberoGr00tOutput.from_model_output(
        {"action_pred": torch.zeros((1, 2, 128))},
        processor=FakeProcessor(),
        policy_input=policy_input,
        action_chunk_size=2,
        device="cpu",
    )

    assert output.action.shape == (1, 2, 7)
    torch.testing.assert_close(output.action[0, :, -1], torch.tensor([1.0, -1.0]))
