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

from verl_vla.utils import rollout_collection


class FakeCluster:
    def __init__(self) -> None:
        self.calls = 0

    def rollout(self):
        self.calls += 1
        return None, {"collected_dataset": {"root": "new_dataset", "repo_id": "local/new_dataset"}}, {}


def test_collect_lerobot_rollout_dataset_counts_initial_episodes(monkeypatch) -> None:
    cluster = FakeCluster()
    counts = iter([3, 8])
    truncations = []

    monkeypatch.setattr(rollout_collection, "count_lerobot_episodes", lambda _root: next(counts))

    def fake_truncate(root, count):
        truncations.append((root, count))

    monkeypatch.setattr(rollout_collection, "truncate_lerobot_episodes", fake_truncate)

    result = rollout_collection.collect_lerobot_rollout_dataset(
        cluster,
        target_episodes=10,
        initial_completed_episodes=5,
        log_prefix="test rollout",
    )

    assert cluster.calls == 2
    assert result["collected_dataset"]["root"] == "new_dataset"
    assert truncations == [("new_dataset", 5)]


def test_collect_lerobot_rollout_dataset_skips_when_initial_count_reaches_target() -> None:
    cluster = FakeCluster()

    result = rollout_collection.collect_lerobot_rollout_dataset(
        cluster,
        target_episodes=5,
        initial_completed_episodes=5,
        log_prefix="test rollout",
    )

    assert cluster.calls == 0
    assert result == {}
