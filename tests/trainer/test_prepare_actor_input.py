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

"""Unit tests for preparing raw SAC rollouts as actor input.

The tests use synthetic collated rollout DataProto objects and run on CPU
without Ray, models, or simulators. Observations encode ``(lane, time)`` so
transition continuity can be asserted exactly across rollout windows.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_vla.trainer.sac.episode_buffer import EpisodeBuffer
from verl_vla.trainer.sac.sac_ray_trainer import RobRaySACTrainer

NUM_SUBSTEPS = 2


def make_rollout(
    num_lanes: int,
    num_steps: int,
    *,
    t0: int = 0,
    terminated: set[tuple[int, int]] = frozenset(),
    truncated: set[tuple[int, int]] = frozenset(),
    success: set[tuple[int, int]] = frozenset(),
) -> DataProto:
    """Build a synthetic collated [B, S] rollout whose obs encode (lane, time)."""
    times = torch.arange(t0, t0 + num_steps, dtype=torch.float32).expand(num_lanes, num_steps)
    lanes = torch.arange(num_lanes, dtype=torch.float32).unsqueeze(1).expand(num_lanes, num_steps)

    state = torch.stack([lanes, times], dim=-1)  # [B, S, 2]
    image = (lanes * 1000 + times).unsqueeze(-1).unsqueeze(-1).expand(num_lanes, num_steps, 2, 2).clone()
    action = torch.stack([lanes, times, torch.full_like(times, 0.5)], dim=-1)  # [B, S, 3]

    terminated_steps = torch.zeros(num_lanes, num_steps, NUM_SUBSTEPS, dtype=torch.bool)
    truncated_steps = torch.zeros_like(terminated_steps)
    success_steps = torch.zeros_like(terminated_steps)
    reward_steps = torch.zeros(num_lanes, num_steps, NUM_SUBSTEPS, dtype=torch.float32)
    reward_steps[:, :, 0] = lanes * 100 + times

    for lane, step in terminated:
        terminated_steps[lane, step, -1] = True
    for lane, step in truncated:
        truncated_steps[lane, step, -1] = True
    for lane, step in success:
        success_steps[lane, step, 0] = True

    task_id = (np.arange(num_lanes, dtype=np.int64) + 7)[:, None].repeat(num_steps, axis=1)
    task = np.empty((num_lanes, num_steps), dtype=object)
    for lane in range(num_lanes):
        task[lane, :] = f"task-{lane}"

    return DataProto.from_dict(
        tensors={
            "obs.state": state,
            "obs.image": image,
            "action.action": action,
            "next.terminated": terminated_steps,
            "next.truncated": truncated_steps,
            "next.success": success_steps,
            "next.reward": reward_steps,
        },
        non_tensors={"obs.task_id": task_id, "obs.task": task},
    )


def sort_by_lane_time(data: DataProto) -> DataProto:
    key = data.batch["t0.obs.state"][:, 0] * 1_000_000 + data.batch["t0.obs.state"][:, 1]
    idx = torch.argsort(key)
    idx_np = idx.numpy()
    return DataProto.from_dict(
        tensors={k: v[idx] for k, v in data.batch.items()},
        non_tensors={k: v[idx_np] for k, v in data.non_tensor_batch.items()},
    )


def lane_time(data: DataProto, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    state = data.batch[f"{prefix}.obs.state"]
    return state[:, 0], state[:, 1]


def make_trainer(*, step_penalty: float = 0.0, world_size: int = 1, auto_reset: bool = True) -> RobRaySACTrainer:
    trainer = RobRaySACTrainer.__new__(RobRaySACTrainer)
    trainer.trainer_config = SimpleNamespace(step_penalty=step_penalty)
    trainer.global_steps = 0
    trainer.cluster = SimpleNamespace(actor_worker_group=SimpleNamespace(world_size=world_size))
    trainer._episode_buffer = EpisodeBuffer(auto_reset=auto_reset)
    return trainer


def test_prepare_actor_input_emits_complete_episodes_with_episode_success():
    kwargs = dict(
        terminated={(0, 1), (1, 2)},
        truncated={(0, 3)},
        success={(0, 1)},
    )

    trainer = make_trainer(step_penalty=0.25)
    out = trainer._prepare_actor_input(make_rollout(3, 4, **kwargs))
    assert out is not None

    # lane0: two full episodes (steps 0-1, 2-3); lane1: steps 0-2; lane2: no done -> stays open.
    assert len(out) == 7

    out = sort_by_lane_time(out)
    torch.testing.assert_close(
        out.batch["t0.obs.state"],
        torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        out.batch["t1.obs.state"],
        torch.tensor([[0, 1], [0, 1], [0, 3], [0, 3], [1, 1], [1, 2], [1, 2]], dtype=torch.float32),
    )
    torch.testing.assert_close(out.batch["info.valids"], torch.ones(7))
    torch.testing.assert_close(out.batch["info.success_mask"], torch.tensor([1, 1, 0, 0, 0, 0, 0]).float())
    torch.testing.assert_close(out.batch["info.terminateds"], torch.tensor([0, 1, 0, 0, 0, 0, 1]).float())
    torch.testing.assert_close(
        out.batch["info.rewards"], torch.tensor([-0.25, 0.75, 1.75, 2.75, 99.75, 100.75, 101.75])
    )
    assert out.non_tensor_batch["t0.obs.task_id"].dtype == np.int64
    np.testing.assert_array_equal(out.non_tensor_batch["t0.obs.task_id"], np.array([7, 7, 7, 7, 8, 8, 8]))
    np.testing.assert_array_equal(
        out.non_tensor_batch["t0.obs.task"], np.array(["task-0"] * 4 + ["task-1"] * 3, dtype=object)
    )


def test_prepare_actor_input_discards_single_slot_done_segments():
    trainer = make_trainer()
    out = trainer._prepare_actor_input(make_rollout(1, 6, terminated={(0, 2), (0, 3), (0, 4), (0, 5)}))

    assert out is not None and len(out) == 3
    _, t0_times = lane_time(out, "t0")
    torch.testing.assert_close(t0_times, torch.tensor([0.0, 1.0, 2.0]))


def test_prepare_actor_input_recovers_cross_rollout_episode():
    """An episode straddling two windows keeps its early transitions (the legacy path drops them)."""
    trainer = make_trainer()

    assert trainer._prepare_actor_input(make_rollout(1, 3, t0=0)) is None

    out = trainer._prepare_actor_input(make_rollout(1, 3, t0=3, terminated={(0, 1)}, success={(0, 1)}))
    assert out is not None
    assert len(out) == 5  # times 0..4; window-2 step 2 stays open

    out = sort_by_lane_time(out)
    _, t0_times = lane_time(out, "t0")
    _, t1_times = lane_time(out, "t1")
    torch.testing.assert_close(t0_times, torch.arange(5, dtype=torch.float32))
    # Continuity across the rollout boundary: t1 is the true next obs, including 2 -> 3.
    torch.testing.assert_close(t1_times[:4], torch.arange(1, 5, dtype=torch.float32))
    assert t1_times[4].item() == t0_times[4].item()  # terminal self-copy

    torch.testing.assert_close(out.batch["info.terminateds"], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(out.batch["info.success_mask"], torch.ones(5))
    torch.testing.assert_close(out.batch["info.rewards"], torch.arange(5, dtype=torch.float32))


def test_prepare_actor_input_keeps_open_episode_until_done():
    trainer = make_trainer()

    assert trainer._prepare_actor_input(make_rollout(1, 256, t0=0)) is None
    out = trainer._prepare_actor_input(make_rollout(1, 1, t0=256, terminated={(0, 0)}))

    assert out is not None and len(out) == 257
    _, t0_times = lane_time(out, "t0")
    torch.testing.assert_close(t0_times, torch.arange(257, dtype=torch.float32))


def test_prepare_actor_input_pads_transitions_to_actor_world_size():
    trainer = make_trainer(world_size=8)
    out = trainer._prepare_actor_input(make_rollout(1, 3, terminated={(0, 2)}))

    assert out is not None and len(out) == 8
    assert out.batch["info.valids"].sum().item() == 3


def test_non_auto_reset_rejects_lane_without_done():
    """Without auto reset the next rollout starts from a real reset, so a lane
    that ends the window mid-episode would splice with an unrelated episode."""
    trainer = make_trainer(auto_reset=False)

    with pytest.raises(ValueError, match="without a done in lanes \\[1\\]"):
        trainer._prepare_actor_input(make_rollout(2, 3, terminated={(0, 2)}))


def test_non_auto_reset_accepts_done_in_every_lane():
    trainer = make_trainer(auto_reset=False)

    # lane0 finishes early and pads with repeated terminal dones; lane1 finishes at the window end.
    out = trainer._prepare_actor_input(make_rollout(2, 4, terminated={(0, 1), (0, 2), (0, 3), (1, 3)}))

    assert out is not None and len(out) == 6  # lane0 steps 0-1 + lane1 steps 0-3; padding discarded


if __name__ == "__main__":
    test_prepare_actor_input_emits_complete_episodes_with_episode_success()
    test_prepare_actor_input_discards_single_slot_done_segments()
    test_prepare_actor_input_recovers_cross_rollout_episode()
    test_prepare_actor_input_keeps_open_episode_until_done()
    test_prepare_actor_input_pads_transitions_to_actor_world_size()
    test_non_auto_reset_rejects_lane_without_done()
    test_non_auto_reset_accepts_done_in_every_lane()
    print("all actor input preparation tests passed")
