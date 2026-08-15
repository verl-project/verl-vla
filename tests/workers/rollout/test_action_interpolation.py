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

import torch

from verl_vla.workers.rollout.action_chunk_processor import (
    ActionChunkStitchingProcessor,
    ActionInterpolationProcessor,
    CompositeActionChunkProcessor,
)


def test_action_chunk_processor_linearly_upsamples_action_chunk() -> None:
    processor = ActionInterpolationProcessor(factor=2)
    actions = torch.tensor([[[0.0, 2.0], [3.0, 5.0], [6.0, 8.0]]])

    interpolated = processor.process(actions, stage_id=0, episode_start=None)

    assert interpolated.shape == (1, 6, 2)
    torch.testing.assert_close(interpolated[:, 0], actions[:, 0])
    torch.testing.assert_close(interpolated[:, -1], actions[:, -1])
    torch.testing.assert_close(interpolated[0, :, 0], torch.linspace(0.0, 6.0, 6))


def test_action_chunk_processor_blends_previous_plan_tail_into_new_prefix() -> None:
    processor = ActionChunkStitchingProcessor(execution_steps=4, blend_steps=2)
    previous = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)

    first = processor.process(
        previous,
        stage_id=0,
        episode_start=torch.tensor([True]),
    )
    second = processor.process(
        torch.full_like(previous, 10.0),
        stage_id=0,
        episode_start=torch.tensor([False]),
    )
    after_reset = processor.process(
        torch.full_like(previous, 20.0),
        stage_id=0,
        episode_start=torch.tensor([True]),
    )

    torch.testing.assert_close(first, previous[:, :4])
    torch.testing.assert_close(second[0, :, 0], torch.tensor([4.0, 10.0, 10.0, 10.0]))
    torch.testing.assert_close(after_reset, torch.full((1, 4, 1), 20.0))


def test_composite_action_chunk_processor_applies_children_in_order() -> None:
    processor = CompositeActionChunkProcessor(
        [
            ActionInterpolationProcessor(factor=2),
            ActionChunkStitchingProcessor(execution_steps=4, blend_steps=2),
        ]
    )
    actions = torch.tensor([[[0.0], [3.0], [6.0]]])

    processed = processor.process(actions, stage_id=0, episode_start=torch.tensor([True]))

    assert processed.shape == (1, 4, 1)
    torch.testing.assert_close(processed[0, :, 0], torch.linspace(0.0, 6.0, 6)[:4])
