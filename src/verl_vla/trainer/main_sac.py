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
from dataclasses import dataclass, field
from pprint import pprint

import datasets
import hydra
import ray
from omegaconf import OmegaConf
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl_vla.workers.engine import VLAActorRolloutRefWorker
from verl_vla.workers.env.env_worker import EnvWorker

from .sac.sac_ray_trainer import RobRaySACTrainer

logger = logging.getLogger(__name__)


@dataclass
class VLAResourcePoolManager(ResourcePoolManager):
    cpu_pool_names: set[str] = field(default_factory=set)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            use_gpu = resource_pool_name not in self.cpu_pool_names
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=use_gpu,
                max_colocate_count=3,
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
            n_gpus
            for pool_name, process_on_nodes in self.resource_pool_spec.items()
            if pool_name not in self.cpu_pool_names
            for n_gpus in process_on_nodes
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


@hydra.main(config_path="config", config_name="rob_sac_trainer", version_base=None)
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
    # print initial config
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(VLAActorRolloutRefWorker),
        Role.Env: ray.remote(EnvWorker),
    }

    # setup resource pool manager
    train_rollout_gpu_num = config.trainer.n_rollout_gpus_per_node
    train_rollout_nodes_num = config.trainer.nnodes
    env_worker_num = config.trainer.get("n_env_workers_per_node", config.trainer.get("n_env_gpus_per_node", 0))
    env_device = str(config.env.train.get("device", "cuda")).lower()
    env_use_gpu = env_device != "cpu"
    env_nodes_num = config.env.disagg_sim.nnodes if config.env.disagg_sim.enable else config.trainer.nnodes
    env_pool_name = "env_cpu_pool" if not env_use_gpu else "env_gpu_pool"

    resource_pool_spec = {
        "train_rollout_pool": [train_rollout_gpu_num] * train_rollout_nodes_num,
        env_pool_name: [env_worker_num] * env_nodes_num,
    }
    mapping = {
        Role.ActorRollout: "train_rollout_pool",
        Role.Env: env_pool_name,
    }
    resource_pool_manager = VLAResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
        cpu_pool_names={env_pool_name} if not env_use_gpu else set(),
    )

    # create datasets
    train_dataset = datasets.load_dataset("parquet", data_files=config.data.train_files)["train"]
    val_dataset = datasets.load_dataset("parquet", data_files=config.data.val_files)["train"]

    # instantiate trainer and start training
    trainer = RobRaySACTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
