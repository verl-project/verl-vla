# Copyright 2025 The RLinf Authors.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    collect_lazy_compute_data_proto,
    dispatch_lazy_compute_data_proto,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_name,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig

from verl_vla.utils.recorder import merge_lerobot_datasets
from verl_vla.workers.env.config import EnvWorkerConfig

from .env_manager import EnvManager


def dispatch_reset_env(worker_group, *args, **kwargs):
    mode = kwargs.pop("mode", "train")
    reset_eval = kwargs.pop("reset_eval", False)
    reset_args = DataProto.from_dict(
        meta_info={
            "mode": mode,
            "reset_eval": reset_eval,
        }
    )
    return dispatch_lazy_compute_data_proto("env", worker_group, reset_args, **kwargs)


def collect_reset_env(worker_group, *args, **kwargs):
    return collect_lazy_compute_data_proto("env", worker_group, *args, **kwargs)


def put_tensor_cpu(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_cpu(value)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cpu().contiguous()
    return data_dict


def create_env_batch(obs, rews, dones, infos, meta=None):
    ret_dict = {"obs": obs, "rews": rews, "dones": dones, "infos": infos}
    if meta is not None:
        ret_dict.update(meta=meta)

    ret_dict = put_tensor_cpu(ret_dict)
    return ret_dict


def create_env_batch_dataproto(obs, rewards, dones, infos, meta=None):
    step_result = {
        "observation": obs["observation"],
        "task": obs["task"],
        "task_id": obs.get("task_id"),
        "next.reward": rewards,
        "next.done": dones,
        "info": infos,
    }
    if meta is not None:
        step_result["meta"] = meta

    step_result = put_tensor_cpu(step_result)
    obs_tensor_batch = {}
    observations = step_result["observation"]
    if observations:
        for key in observations[0]:
            obs_tensor_batch[f"obs.{key}"] = torch.as_tensor(
                np.stack([observation[key] for observation in observations])
            )
    tensor_batch = {
        **obs_tensor_batch,
        "next.reward": step_result["next.reward"],
        "next.done": step_result["next.done"],
    }
    non_tensor_batch = {"obs.task": step_result["task"]}
    if step_result["task_id"] is not None:
        non_tensor_batch["obs.task_id"] = np.asarray(step_result["task_id"], dtype=np.int64)
    output = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

    return output


class EnvWorker(Worker, DistProfilerExtension):
    def __init__(self, config: DictConfig, role=None):
        Worker.__init__(self)
        self.cfg = config
        self.role = role
        self.env_worker_cfg: EnvWorkerConfig = config.env_worker
        self.simulator_type = self.env_worker_cfg.simulator.simulator_type
        self.simulator_cfg = OmegaConf.structured(self.env_worker_cfg)
        self.train_video_cnt = 0
        self.eval_video_cnt = 0

        self.simulator_list = []
        self.last_obs_list = []
        self.last_dones_list = []
        self.eval_simulator_list = []

        self.stage_num = self.cfg.env_loop.pipeline_stage_num
        self.stage_modes = ["train"] * self.stage_num
        device_name = self.env_worker_cfg.device or get_device_name()
        if device_name == "cpu":
            # CPU env workers do not need torch distributed collectives; only Ray dispatch metadata is required.
            self._register_dispatch_collect_info("env", dp_rank=self.rank, is_collect=True)
        else:
            initialize_global_process_group_ray(timeout_second=None)
            env_device_mesh = init_device_mesh(
                device_name, mesh_shape=(self.world_size, 1), mesh_dim_names=["dp", "tp"]
            )
            self._register_dispatch_collect_info("env", dp_rank=env_device_mesh["dp"].get_local_rank(), is_collect=True)

        # Initialize profiler
        omega_profiler_config = self.env_worker_cfg.profiler or {}
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    def _make_eval_env_cfg(self):
        eval_cfg = OmegaConf.create(OmegaConf.to_container(self.simulator_cfg, resolve=False))
        OmegaConf.set_struct(eval_cfg, False)
        if "teleop" in eval_cfg:
            eval_cfg.teleop.enable = False
        return eval_cfg

    def _simulators(self, mode: str):
        if mode == "eval":
            if not self.eval_simulator_list:
                raise RuntimeError("Eval simulator is not initialized. Add 'eval' to env.env_worker.modes.")
            return self.eval_simulator_list
        return self.simulator_list

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="green", role="env_init")
    def init_worker(self):
        if self.simulator_type == "libero":
            from verl_vla.envs.libero_env.libero_env import LiberoEnv

            modes = list(self.env_worker_cfg.modes)
            if not set(modes).issubset({"train", "eval"}):
                raise ValueError(f"Unsupported LIBERO env modes: {modes}")
            eval_cfg = self._make_eval_env_cfg() if "eval" in modes else None
            for stage_id in range(self.stage_num):
                if "train" in modes:
                    self.simulator_list.append(
                        EnvManager(
                            self.simulator_cfg,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=LiberoEnv,
                            stage_id=stage_id,
                        )
                    )
                if eval_cfg is not None:
                    self.eval_simulator_list.append(
                        EnvManager(
                            eval_cfg,
                            rank=self._rank,
                            world_size=self._world_size,
                            env_cls=LiberoEnv,
                            stage_id=stage_id,
                            only_eval=True,
                        )
                    )

        elif self.simulator_type == "isaac":
            from verl_vla.envs.isaac_env.isaac_env import IsaacEnv

            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.simulator_cfg,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=IsaacEnv,
                        stage_id=stage_id,
                    )
                )
        elif self.simulator_type == "lerobot":
            from verl_vla.envs.lerobot_env.lerobot_env import LeRobotEnv

            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.simulator_cfg,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=LeRobotEnv,
                        stage_id=stage_id,
                    )
                )
        elif self.simulator_type == "arena":
            from verl_vla.envs.arena_env.arena_env import IsaacLabArenaEnv

            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.simulator_cfg,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=IsaacLabArenaEnv,
                        stage_id=stage_id,
                    )
                )
        else:
            raise NotImplementedError(f"Simulator type {self.simulator_type} not implemented")

        for simulator in self.simulator_list:
            simulator.start_simulator()
        for simulator in self.eval_simulator_list:
            simulator.start_simulator()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="env"), blocking=False)
    @DistProfiler.annotate(color="red", role="env_interact_step")
    def env_interact_step(self, data: DataProto) -> dict:
        """
        This function is used to interact with the environment.
        """
        if data.batch is not None and "action" in data.batch.keys():
            chunk_actions: torch.Tensor = data.batch["action"]
            chunk_values = data.batch.get("critic_value")
        else:
            chunk_actions = torch.as_tensor(data.non_tensor_batch["action"])
            chunk_values = (
                torch.as_tensor(data.non_tensor_batch["critic_value"])
                if "critic_value" in data.non_tensor_batch
                else None
            )
        stage_id: int = data.meta_info["stage_id"]

        # Pi0.5 Libero is not required
        # TODO: prepare actions according to simulator type
        # chunk_actions = prepare_actions(
        #     simulator_type=self.simulator_type,
        #     raw_chunk_actions=chunk_actions,
        #     num_action_chunks=self.cfg.actor.model.num_action_chunks,
        #     action_dim=self.cfg.actor.model.action_dim,
        # )

        mode = self.stage_modes[stage_id]
        simulators = self._simulators(mode)
        extracted_obs, chunk_rewards, chunk_dones, _chunk_truncations, infos = simulators[stage_id].step(
            chunk_actions, chunk_values=chunk_values
        )

        env_batch = create_env_batch_dataproto(
            obs=extracted_obs,
            rewards=chunk_rewards,
            dones=chunk_dones,
            infos=infos,
        )
        return env_batch

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_eval_benchmark_size(self):
        """Get the number of episodes in the eval benchmark."""
        simulator = self.eval_simulator_list[0] if self.eval_simulator_list else self.simulator_list[0]
        return int(simulator.env_benchmark_size())

    @register(
        dispatch_mode={
            "dispatch_fn": dispatch_reset_env,
            "collect_fn": collect_reset_env,
        },
        blocking=False,
    )
    @DistProfiler.annotate(color="blue", role="env_reset_env")
    def reset_env(self, _data: DataProto):
        mode = _data.meta_info.get("mode", "train")
        reset_eval = bool(_data.meta_info.get("reset_eval", False))
        simulators = self._simulators(mode)

        result_list = []
        for stage_id in range(self.stage_num):
            self.stage_modes[stage_id] = mode
            options = {
                "env_idx": list(range(self.env_worker_cfg.num_envs)),
            }
            if reset_eval:
                options["reset_eval"] = True
            result = simulators[stage_id].reset(options=options)
            result_list.append(result)
        output_tensor_dict = {}
        output_non_tensor_dict = {}

        observations = [observation for obs, _info in result_list for observation in obs["observation"]]
        if observations:
            for key in observations[0]:
                output_tensor_dict[key] = torch.as_tensor(np.stack([observation[key] for observation in observations]))
        output_non_tensor_dict["task"] = [task for obs, _info in result_list for task in obs["task"]]
        task_ids = [task_id for obs, _info in result_list for task_id in obs.get("task_id", [])]
        if task_ids:
            output_non_tensor_dict["task_id"] = np.asarray(task_ids, dtype=np.int64)

        output = DataProto.from_dict(tensors=output_tensor_dict, non_tensors=output_non_tensor_dict)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="gray", role="env_finish_rollout")
    def finish_rollout(self):
        for simulator in self.simulator_list:
            simulator.finish_rollout()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="gray", role="env_pop_lerobot_dataset")
    def pop_lerobot_dataset(self):
        recorder_cfg = self.env_worker_cfg.recorder
        if not recorder_cfg.enable or not recorder_cfg.lerobot.enable:
            return None

        datasets = []
        for simulator in self.simulator_list:
            dataset = simulator.pop_completed_dataset()
            if dataset is not None:
                datasets.append(dataset)
        if not datasets:
            return None

        root = Path(recorder_cfg.lerobot.root)
        repo_id = f"{recorder_cfg.lerobot.repo_id}_rank_{self._rank}"
        return merge_lerobot_datasets(
            roots=[dataset["root"] for dataset in datasets],
            output_root=root / repo_id,
            repo_id=repo_id,
            repo_ids=[dataset["repo_id"] for dataset in datasets],
            append=True,
        )
