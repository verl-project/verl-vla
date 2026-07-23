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

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import verl_vla.recorder.impl.video as video_module
from verl_vla.envs.base import BaseEnv
from verl_vla.recorder.base import BaseRecorder
from verl_vla.recorder.impl.video import VideoRecorder
from verl_vla.recorder.recorder import MultiRecorder


class _SpyRecorder(BaseRecorder):
    def __init__(self) -> None:
        self.modes: list[str] = []
        self.cleared: list[int] = []

    def record_once(self, **kwargs) -> None:
        pass

    def save_episode(self, env_id: int = 0) -> None:
        pass

    def clear_episode(self, env_id: int = 0) -> None:
        self.cleared.append(env_id)

    def set_mode(self, mode: str) -> None:
        self.modes.append(mode)

    def finalize(self) -> None:
        pass


class _ModeFakeEnv(BaseEnv):
    env_type = "fake"

    def __init__(self, recorder: BaseRecorder | None, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.auto_reset_enabled = True
        self._latest_obs = {"observation": [], "task": [], "task_id": []}
        self.teleops = []
        self.recorder = recorder
        self._recorder_episode_done = np.zeros(num_envs, dtype=bool)
        self._confirm_before_record_enabled = False

    def env_reset(self, *, env_ids, reset_eval: bool = False):
        del reset_eval
        return self._latest_obs

    def env_step(self, action, *, env_ids):
        raise NotImplementedError


def test_set_recorder_mode_forwards_and_clears_buffers() -> None:
    recorder = _SpyRecorder()
    env = _ModeFakeEnv(recorder)
    env._recorder_episode_done[:] = True

    env.set_recorder_mode("train")
    assert recorder.modes == []
    assert recorder.cleared == []

    env.set_recorder_mode("eval")
    assert recorder.modes == ["eval"]
    assert recorder.cleared == [0, 1]
    assert not env._recorder_episode_done.any()

    env.set_recorder_mode("eval")
    assert recorder.modes == ["eval"]


def test_reset_applies_mode_even_on_auto_reset_early_return() -> None:
    recorder = _SpyRecorder()
    env = _ModeFakeEnv(recorder)

    env.reset(options={"env_idx": [0, 1], "mode": "eval"})
    assert recorder.modes == ["eval"]

    # Auto-reset early return still switches the recorder back to train.
    env.reset(options={"env_idx": [0, 1], "mode": "train"})
    assert recorder.modes == ["eval", "train"]


def test_multi_recorder_forwards_set_mode() -> None:
    children = [_SpyRecorder(), _SpyRecorder()]
    MultiRecorder(list(children)).set_mode("eval")
    assert all(child.modes == ["eval"] for child in children)


def test_video_recorder_saves_per_mode_dirs_and_counts(monkeypatch, tmp_path: Path) -> None:
    class _Strategy:
        fps = 10
        robot_type = "fake"

        def make_frame(self, **kwargs) -> dict[str, Any]:
            return {"observation.images.cam": np.zeros((8, 8, 3), dtype=np.uint8)}

    saved: list[tuple[str, str]] = []
    monkeypatch.setattr(video_module, "get_lerobot_strategy", lambda *args, **kwargs: _Strategy())
    monkeypatch.setattr(
        video_module,
        "save_rollout_video",
        lambda frames, output_dir, video_name, fps: saved.append((output_dir, video_name)),
    )

    recorder = VideoRecorder(
        cfg=SimpleNamespace(root=str(tmp_path), fps=10, font_size=12),
        env_type="fake",
        rank=0,
        stage_id=0,
        num_envs=1,
    )

    def _record_and_save() -> None:
        recorder.record_once(observation={}, action=np.zeros(2), task="t")
        recorder.save_episode(0)

    _record_and_save()
    recorder.set_mode("eval")
    _record_and_save()
    recorder.set_mode("train")
    _record_and_save()

    assert saved == [
        (str(tmp_path / "train" / "rank_0" / "stage_0" / "env_0"), "0"),
        (str(tmp_path / "eval" / "rank_0" / "stage_0" / "env_0"), "0"),
        (str(tmp_path / "train" / "rank_0" / "stage_0" / "env_0"), "1"),
    ]
