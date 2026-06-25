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

import math

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import Role

from verl_vla.env_loop.env_loop import EnvLoop
from verl_vla.trainer.main_sac import VLAResourcePoolManager
from verl_vla.workers.engine import VLAActorRolloutRefWorker
from verl_vla.workers.env.env_worker import EnvWorker


def _ensure_ray(config):
    if ray.is_initialized():
        return
    default_runtime_env = get_ppo_ray_runtime_env()
    ray_kwargs = OmegaConf.select(config, "ray_kwargs", default={})
    ray_init_kwargs = ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    ray.init(**OmegaConf.to_container(ray_init_kwargs))


def _count_completed_policy_eval_episodes(
    output: DataProto,
    remaining: int,
    *,
    allow_multiple_per_row: bool,
) -> tuple[int, int]:
    dones = output.batch["next.done"].bool()
    rewards = output.batch["next.reward"].float()
    if dones.ndim == 1:
        dones = dones[:, None]
        rewards = rewards[:, None]

    completed = 0
    success = 0
    for batch_idx in range(dones.shape[0]):
        done_row = dones[batch_idx].reshape(-1)
        reward_row = rewards[batch_idx].reshape(-1)
        done_steps = torch.nonzero(done_row, as_tuple=False).flatten().tolist()
        if not allow_multiple_per_row:
            done_steps = done_steps[:1]
        for done_idx in done_steps:
            if completed >= remaining:
                break
            completed += 1
            success += int(float(reward_row[done_idx]) > 0.0)
        if completed >= remaining:
            break
    return completed, success


def eval_recap_policy(config, policy_path: str) -> dict[str, float]:
    _ensure_ray(config)
    return ray.get(run_policy_eval.remote(config, policy_path))


@ray.remote
def run_policy_eval(config, policy_path: str) -> dict[str, float]:
    eval_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    OmegaConf.set_struct(eval_config, False)
    OmegaConf.update(eval_config, "actor_rollout_ref.model.path", policy_path)
    OmegaConf.update(eval_config, "actor_rollout_ref.model.tokenizer_path", policy_path)
    OmegaConf.update(eval_config, "env.train.modes", ["eval"], merge=True)
    OmegaConf.resolve(eval_config)

    from verl.single_controller.ray import RayWorkerGroup

    env_device = str(eval_config.env.train.get("device", "cuda")).lower()
    env_pool_name = "env_cpu_pool" if env_device == "cpu" else "env_gpu_pool"
    resource_pool_manager = VLAResourcePoolManager(
        resource_pool_spec={
            "train_rollout_pool": [eval_config.trainer.n_gpus_per_node] * eval_config.trainer.nnodes,
            env_pool_name: [eval_config.trainer.n_env_workers_per_node] * eval_config.trainer.nnodes,
        },
        mapping={
            Role.ActorRollout: "train_rollout_pool",
            Role.Env: env_pool_name,
        },
        cpu_pool_names={env_pool_name} if env_device == "cpu" else set(),
    )
    resource_pool_manager.create_resource_pool()

    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}
    resource_pool_to_cls[resource_pool_manager.get_resource_pool(Role.ActorRollout)]["actor_rollout"] = (
        RayClassWithInitArgs(
            cls=ray.remote(VLAActorRolloutRefWorker),
            config=eval_config.actor_rollout_ref,
            role="actor_rollout",
        )
    )
    resource_pool_to_cls[resource_pool_manager.get_resource_pool(Role.Env)]["env"] = RayClassWithInitArgs(
        cls=ray.remote(EnvWorker),
        config=eval_config.env,
    )

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            device_name=eval_config.trainer.device,
        )
        all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()
    env_wg = all_wg["env"]
    env_loop = EnvLoop(config=eval_config, rollout_wg=actor_rollout_wg, env_wg=env_wg)

    state_ids_result = env_wg.get_all_state_ids()
    benchmark_size = len(state_ids_result[0]) if isinstance(state_ids_result, list) else len(state_ids_result)
    max_episodes = OmegaConf.select(eval_config, "recap.policy_eval.max_episodes", default=None)
    max_episodes = benchmark_size if max_episodes is None else int(max_episodes)

    total_envs = (
        env_loop.env_wg.world_size
        * int(eval_config.env.train.num_envs)
        * int(eval_config.env.rollout.pipeline_stage_num)
    )
    if bool(eval_config.env.train.get("single_env_rollout", False)):
        total_envs = env_loop.env_wg.world_size
    dummy_state_ids = np.zeros(total_envs, dtype=np.int64)
    dummy_task_ids = np.zeros(total_envs, dtype=np.int64)

    if bool(eval_config.env.train.get("async_reset", False)):
        episodes_per_env = math.ceil(max_episodes / max(1, total_envs))
        chunks_per_episode = math.ceil(
            int(eval_config.env.train.max_episode_steps) / int(eval_config.env.actor.model.num_action_chunks)
        )
        env_loop.max_interactions = episodes_per_env * chunks_per_episode
        env_loop.configured_max_interactions = env_loop.max_interactions

    completed = 0
    success = 0
    eval_step = 0
    while completed < max_episodes:
        reset_eval = eval_step == 0
        prompts = DataProto.from_dict(
            non_tensors={
                "state_ids": dummy_state_ids,
                "task_ids": dummy_task_ids,
            },
            meta_info={
                "mode": "eval",
                "reset_eval": reset_eval,
                "validate": True,
                "global_steps": eval_step,
            },
        )
        reset_future = env_wg.reset_envs_to_state_ids(
            DataProto.from_dict(
                non_tensors={
                    "state_ids": dummy_state_ids,
                    "task_ids": dummy_task_ids,
                },
                meta_info={
                    "mode": "eval",
                    "reset_eval": reset_eval,
                },
            )
        )
        rollout_output = env_loop.generate_sequences(prompts, reset_future)
        rollout_completed, rollout_success = _count_completed_policy_eval_episodes(
            rollout_output,
            remaining=max_episodes - completed,
            allow_multiple_per_row=bool(eval_config.env.train.get("async_reset", False)),
        )
        completed += rollout_completed
        success += rollout_success
        eval_step += 1
        if rollout_completed == 0:
            break

    return {
        "episodes": float(completed),
        "success": float(success),
        "success_rate": float(success / completed) if completed > 0 else 0.0,
        "benchmark_size": float(benchmark_size),
    }
