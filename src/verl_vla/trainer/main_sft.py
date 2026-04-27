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

import logging
from dataclasses import dataclass
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role

from verl_vla.trainer.sft.sft_ray_trainer import RobRaySFTTrainer
from verl_vla.workers.engine import VLAActorRolloutRefWorker

logger = logging.getLogger(__name__)


@dataclass
class VLASFTResourcePoolManager(ResourcePoolManager):
    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def _check_resource_available(self):
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


@hydra.main(config_path="config", config_name="rob_sft_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        logger.info(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.Actor: ray.remote(VLAActorRolloutRefWorker),
    }

    train_rollout_gpu_num = config.trainer.n_gpus_per_node
    train_rollout_nodes_num = config.trainer.nnodes

    resource_pool_spec = {
        "train_rollout_pool": [train_rollout_gpu_num] * train_rollout_nodes_num,
    }
    mapping = {
        Role.Actor: "train_rollout_pool",
    }
    resource_pool_manager = VLASFTResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )

    trainer = RobRaySFTTrainer(
        config=config,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
