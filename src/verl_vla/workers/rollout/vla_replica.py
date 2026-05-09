# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from omegaconf import DictConfig
from verl.workers.rollout.replica import (
    RolloutConfig,
    RolloutReplica,
    RolloutReplicaRegistry,
)


class VLARolloutReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        rollout_workers: list = None,
    ) -> None:
        super().__init__(
            replica_rank=replica_rank,
            config=config,
            model_config=model_config,
            gpus_per_node=gpus_per_node,
            is_reward_model=is_reward_model,
        )
        self.workers = rollout_workers or []

    async def launch_servers(self):
        pass

    async def wake_up(self):
        pass

    async def sleep(self):
        pass

    async def abort_all_requests(self):
        pass

    async def resume_generation(self):
        pass

    async def clear_kv_cache(self):
        pass

    async def start_profile(self, **kwargs):
        pass

    async def stop_profile(self):
        pass

    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        if self.workers:
            return [worker.execute_checkpoint_engine.remote(method, *args, **kwargs) for worker in self.workers]
        return []


def _load_vla():
    return VLARolloutReplica


RolloutReplicaRegistry.register("vla", _load_vla)
