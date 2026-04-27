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

from __future__ import annotations

import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics


class RobRaySFTTrainer(RayPPOTrainer):
    def __init__(self, config, role_worker_mapping, resource_pool_manager, ray_worker_group_cls):
        self.config = config
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = self.config.trainer.device
        self.use_critic = False
        self.use_reference_policy = False
        self.sft_dataloader = self._create_sft_dataloader()
        self.train_dataloader = self.sft_dataloader
        self.actor_rollout_wg = None

    def _create_sft_dataloader(self) -> StatefulDataLoader:
        sft_config = OmegaConf.select(self.config, "data.sft")
        if not sft_config or not sft_config.get("enable", False):
            raise ValueError("`data.sft.enable` must be True for the VLA SFT pipeline.")

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id=sft_config.repo_id,
            revision=sft_config.get("revision"),
            video_backend=sft_config.get("video_backend"),
        )

        if sft_config.get("shuffle", True):
            generator = torch.Generator()
            seed = sft_config.get("seed", self.config.data.get("seed", self.config.trainer.get("seed")))
            if seed is not None:
                generator.manual_seed(int(seed))
            sampler = RandomSampler(data_source=dataset, generator=generator)
        else:
            sampler = SequentialSampler(data_source=dataset)

        return StatefulDataLoader(
            dataset=dataset,
            batch_size=sft_config.batch_size,
            num_workers=sft_config.num_workers,
            drop_last=sft_config.get("drop_last", False),
            sampler=sampler,
        )

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        resource_pool_to_cls[resource_pool]["actor"] = actor_cls

        all_wg = {}
        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg["actor"]
        self.actor_rollout_wg.init_model()

    def fit(self):
        if self.actor_rollout_wg is None:
            raise RuntimeError("Workers are not initialized. Call `init_workers()` before `fit()`.")

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        total_epochs = int(self.config.trainer.total_epochs)
        steps_per_epoch = len(self.sft_dataloader)
        self.total_training_steps = total_epochs * steps_per_epoch
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.max_steps_duration = 0.0

        for epoch in range(total_epochs):
            for batch in self.sft_dataloader:
                step_start_time = time.perf_counter()
                actor_output = self.actor_rollout_wg.update_actor(self._batch_to_dataproto(batch))
                metrics = reduce_metrics(actor_output.meta_info["metrics"])
                self.global_steps += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{metrics.get('loss', 0.0):.4f}",
                    grad_norm=f"{metrics.get('grad_norm', 0.0):.4f}",
                )
                step_duration = time.perf_counter() - step_start_time
                self.max_steps_duration = max(self.max_steps_duration, step_duration)
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "timing_s/step": step_duration,
                    }
                )
                logger.log(data=metrics, step=self.global_steps)

                is_last_step = self.global_steps >= self.total_training_steps
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    self._save_checkpoint()

                if is_last_step:
                    progress_bar.close()
                    return

        progress_bar.close()

    def _batch_to_dataproto(self, batch: dict) -> DataProto:
        tensor_batch = {}
        non_tensor_batch = {}
        batch_size = None
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                tensor_batch[key] = value
                if batch_size is None:
                    batch_size = value.shape[0]
            else:
                non_tensor_batch[key] = np.array(value, dtype=object)
                if batch_size is None and hasattr(value, "__len__"):
                    batch_size = len(value)
        if batch_size is None:
            batch_size = 1
        meta_info = {
            "global_token_num": [0] * batch_size,
        }
        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)
