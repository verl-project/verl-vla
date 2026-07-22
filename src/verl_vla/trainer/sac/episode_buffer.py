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

"""Per-lane buffering of raw SAC rollout steps until an episode completes."""

import numpy as np
import torch
from verl import DataProto

from verl_vla.utils.data import reduce_substep_dims


class EpisodeBuffer:
    """Return complete raw episodes while retaining unfinished steps across rollouts.

    The rollout contract excludes one-slot episodes. Such segments are repeated
    terminal padding from non-auto-reset environments and are discarded.
    Non-auto-reset rollouts must contain a ``done`` in every lane before the
    next rollout performs a real reset.
    """

    def __init__(self, *, auto_reset: bool):
        """``auto_reset`` mirrors the env config: it decides whether steps may be retained across rollouts."""
        self._auto_reset = bool(auto_reset)
        self._lanes: list[list[DataProto]] | None = None

    def ingest(self, rollout: DataProto) -> list[DataProto]:
        """Append a collated ``[B, S]`` rollout to the per-lane buffers and return the episodes it completes."""
        terminated = reduce_substep_dims(rollout.batch["next.terminated"].bool(), reduction="any")
        truncated = reduce_substep_dims(rollout.batch["next.truncated"].bool(), reduction="any")
        done = terminated | truncated
        num_lanes, num_steps = done.shape

        if self._lanes is None:
            self._lanes = [[] for _ in range(num_lanes)]

        episodes = []
        for lane in range(num_lanes):
            buffer = self._lanes[lane]
            for step in range(num_steps):
                buffer.append(self._select_step(rollout, lane, step))
                if done[lane, step]:
                    if len(buffer) > 1:
                        episodes.append(self._concat_steps(buffer))
                    buffer.clear()

        if not self._auto_reset:
            unfinished = [lane for lane, buffer in enumerate(self._lanes) if buffer]
            if unfinished:
                raise ValueError(
                    f"Rollout window ended without a done in lanes {unfinished} but auto_reset is off: "
                    "Ensure every lane terminates or truncates within one rollout "
                    "window (episode horizon <= rollout window length), or enable "
                    "cluster.env.env_worker.auto_reset."
                )

        return episodes

    def clear(self) -> None:
        """Discard incomplete episodes when environment continuity breaks."""
        if self._lanes is not None:
            for buffer in self._lanes:
                buffer.clear()

    @staticmethod
    def _select_step(data: DataProto, lane: int, step: int) -> DataProto:
        """Copy one ``[1, 1]`` step out of a rollout so buffered steps do not retain the full rollout tensors."""
        return DataProto.from_dict(
            tensors={key: value[lane : lane + 1, step : step + 1].clone() for key, value in data.batch.items()},
            non_tensors={
                key: value[lane : lane + 1, step : step + 1].copy() for key, value in data.non_tensor_batch.items()
            },
        )

    @staticmethod
    def _concat_steps(steps: list[DataProto]) -> DataProto:
        """Join buffered ``[1, 1]`` steps along the time axis into one ``[1, T]`` episode."""
        first = steps[0]
        return DataProto.from_dict(
            tensors={key: torch.cat([step.batch[key] for step in steps], dim=1) for key in first.batch.keys()},
            non_tensors={
                key: np.concatenate([step.non_tensor_batch[key] for step in steps], axis=1)
                for key in first.non_tensor_batch
            },
        )
