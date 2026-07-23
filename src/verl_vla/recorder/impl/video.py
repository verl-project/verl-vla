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

"""Video recorder implementation for rollout visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing_extensions import override

from verl_vla.recorder.base import BaseRecorder
from verl_vla.recorder.config import VideoRecorderConfig
from verl_vla.recorder.strategies import get_lerobot_strategy
from verl_vla.utils.envs.action import save_rollout_video, tile_images


class VideoRecorder(BaseRecorder):
    """Record rollout frames as mp4 videos."""

    def __init__(
        self,
        *,
        cfg: VideoRecorderConfig,
        env_type: str,
        rank: int = 0,
        stage_id: int = 0,
        num_envs: int = 1,
        strategy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.env_type = env_type
        self.rank = rank
        self.stage_id = stage_id
        self.num_envs = num_envs
        self.root = Path(cfg.root)
        self.mode = "train"
        self.strategy = get_lerobot_strategy(env_type, **(strategy_kwargs or {}))
        self._frames: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._video_counts: dict[str, np.ndarray] = {self.mode: np.zeros(num_envs, dtype=np.int64)}

    @override
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._video_counts.setdefault(mode, np.zeros(self.num_envs, dtype=np.int64))

    @override
    def record_once(
        self,
        *,
        env_id: int = 0,
        observation: dict[str, Any],
        extra: dict[str, Any] | None = None,
        action: Any,
        task: str,
        next_reward: Any = 0.0,
        next_terminated: Any = False,
        next_truncated: Any = False,
        next_success: Any = False,
        is_intervention: Any = False,
        critic_value: Any = None,
    ) -> None:
        frame = self.strategy.make_frame(
            observation={**observation, **(extra or {})},
            action=action,
            task=task,
            next_reward=next_reward,
            next_terminated=next_terminated,
            next_truncated=next_truncated,
            next_success=next_success,
            is_intervention=is_intervention,
        )
        if critic_value is not None:
            frame["critic_value"] = critic_value
        images = [value for key, value in frame.items() if key.startswith("observation.images.")]
        if len(images) == 0:
            return

        image_panel = np.asarray(tile_images([np.asarray(image) for image in images], nrows=1))
        text_lines = [
            f"env_type: {self.env_type}",
            f"rank: {self.rank}",
            f"stage_id: {self.stage_id}",
            f"env_id: {env_id}",
            f"strategy.fps: {self.strategy.fps}",
            f"strategy.robot_type: {self.strategy.robot_type}",
        ]
        text_lines.extend(self._frame_lines(frame))
        self._frames[env_id].append(self._append_text_panel(image_panel, text_lines))

    @override
    def save_episode(self, env_id: int = 0) -> None:
        frames = self._frames[env_id]
        if not frames:
            return
        output_dir = self.root / self.mode / f"rank_{self.rank}" / f"stage_{self.stage_id}" / f"env_{env_id}"
        video_name = f"{self._video_counts[self.mode][env_id]}"
        save_rollout_video(frames, output_dir=str(output_dir), video_name=video_name, fps=self.cfg.fps)
        self._video_counts[self.mode][env_id] += 1
        frames.clear()

    @override
    def clear_episode(self, env_id: int = 0) -> None:
        self._frames[env_id].clear()

    @override
    def finalize(self) -> None:
        for env_id in range(self.num_envs):
            self.clear_episode(env_id)

    @staticmethod
    def _frame_lines(frame: dict[str, Any]) -> list[str]:
        lines = []
        for key, value in frame.items():
            if key.startswith("observation.images."):
                shape = getattr(value, "shape", None)
                lines.append(f"{key}: image shape={shape}")
            else:
                lines.append(f"{key}: {VideoRecorder._format_value(value)}")
        return lines

    @staticmethod
    def _format_value(value: Any) -> str:
        array = np.asarray(value)
        if array.ndim == 0 or array.size <= 12:
            return np.array2string(array, precision=4, threshold=12)
        return f"{np.array2string(array.reshape(-1)[:12], precision=4, threshold=12)} ... shape={array.shape}"

    def _append_text_panel(self, image: np.ndarray, lines: list[str]) -> np.ndarray:
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        font = ImageFont.load_default(size=self.cfg.font_size)
        line_height = max(self.cfg.font_size + 6, int(self.cfg.font_size * 1.35))
        panel_height = max(line_height * 4, line_height * (len(lines) + 1))
        total_height = image.shape[0] + panel_height
        total_height = int(np.ceil(total_height / 16) * 16)
        panel_height = total_height - image.shape[0]
        canvas = Image.new("RGB", (image.shape[1], image.shape[0] + panel_height), color=(18, 20, 24))
        canvas.paste(Image.fromarray(image), (0, 0))

        draw = ImageDraw.Draw(canvas)
        y = image.shape[0] + max(8, self.cfg.font_size // 2)
        for line in lines:
            for wrapped in VideoRecorder._wrap_line(line, font, image.shape[1] - 20):
                if y + line_height > image.shape[0] + panel_height:
                    return np.asarray(canvas)
                draw.text((10, y), wrapped, fill=(235, 238, 245), font=font)
                y += line_height
        return np.asarray(canvas)

    @staticmethod
    def _wrap_line(line: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        words = line.split()
        if not words:
            return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.getlength(candidate) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines
