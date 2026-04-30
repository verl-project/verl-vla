from typing import Any
from omegaconf import DictConfig
from verl.workers.rollout.replica import (
    RolloutReplica,
    RolloutReplicaRegistry,
    RolloutConfig,
    HFModelConfig,
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
        """将权重同步相关调用转发给所有 rollout worker"""
        if self.workers:
            # 返回一个 ray.ObjectRef 列表，方便 manager 并行等待
            return [worker.execute_checkpoint_engine.remote(method, *args, **kwargs) for worker in self.workers]
        return []


def _load_vla():
    return VLARolloutReplica


RolloutReplicaRegistry.register("vla", _load_vla)