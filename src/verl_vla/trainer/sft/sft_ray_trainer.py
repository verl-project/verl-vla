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

import logging
import os

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.fs import local_mkdir_safe
from verl.utils.metric import reduce_metrics

from verl_vla.utils.dataloader import LeRobotDataLoaderConfig

from .config import SFTTrainerConfig

logger = logging.getLogger(__name__)


class RobRaySFTTrainer(RayPPOTrainer):
    def __init__(self, config, role_worker_mapping, resource_pool_manager, ray_worker_group_cls):
        self.config = config
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.trainer_config: SFTTrainerConfig = instantiate(self.config.trainer)
        self.device_name = self.trainer_config.device
        self.use_critic = False
        self.use_reference_policy = False
        self.data_config: LeRobotDataLoaderConfig = instantiate(self.config.data)
        self.sft_dataloader = self._create_sft_dataloader(self.data_config)
        self.train_dataloader = self.sft_dataloader
        self.actor_rollout_wg = None

    def _create_sft_dataloader(self, data_config: LeRobotDataLoaderConfig) -> StatefulDataLoader:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        action_delta_steps = int(data_config.action_delta_steps)
        delta_timestamps = None
        if action_delta_steps > 0:
            probe_dataset = LeRobotDataset(
                repo_id=data_config.repo_id,
                root=data_config.root,
                revision=data_config.revision,
                video_backend=data_config.video_backend,
            )
            delta_timestamps = {"action": [t / probe_dataset.fps for t in range(action_delta_steps)]}

        dataset = LeRobotDataset(
            repo_id=data_config.repo_id,
            root=data_config.root,
            revision=data_config.revision,
            video_backend=data_config.video_backend,
            delta_timestamps=delta_timestamps,
        )

        batch_size = int(data_config.batch_size)
        train_world_size = int(self.trainer_config.n_gpus_per_node) * int(self.trainer_config.nnodes)
        if train_world_size > 1 and batch_size % train_world_size != 0:
            raise ValueError(
                "data.batch_size must be divisible by trainer world size for DataProto dispatch: "
                f"batch_size={batch_size}, world_size={train_world_size}."
            )
        drop_last = bool(data_config.drop_last)
        if train_world_size > 1 and not drop_last:
            logger.warning(
                "Forcing data.drop_last=True because distributed DataProto dispatch requires every batch "
                "to be divisible by trainer world size=%s.",
                train_world_size,
            )
            drop_last = True
        dataset_size = len(dataset)
        if drop_last and dataset_size < batch_size:
            raise ValueError(
                "SFT dataset is smaller than one batch with drop_last=True; no batches can be produced. "
                f"dataset_size={dataset_size}, batch_size={batch_size}, repo_id={data_config.repo_id}, "
                f"root={data_config.root}. Reduce data.batch_size or set drop_last=False for single-GPU runs."
            )
        logger.info(
            "Created SFT dataloader: dataset_size=%s, batch_size=%s, drop_last=%s, "
            "steps_per_epoch=%s, repo_id=%s, root=%s",
            dataset_size,
            batch_size,
            drop_last,
            dataset_size // batch_size if drop_last else (dataset_size + batch_size - 1) // batch_size,
            data_config.repo_id,
            data_config.root,
        )

        if data_config.shuffle:
            generator = torch.Generator()
            if data_config.seed is not None:
                generator.manual_seed(int(data_config.seed))
            sampler = RandomSampler(data_source=dataset, generator=generator)
        else:
            sampler = SequentialSampler(data_source=dataset)

        num_workers = int(data_config.num_workers)
        return StatefulDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            sampler=sampler,
            pin_memory=bool(data_config.pin_memory),
            persistent_workers=bool(data_config.persistent_workers) if num_workers > 0 else False,
            prefetch_factor=int(data_config.prefetch_factor) if num_workers > 0 else None,
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

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(
            self.trainer_config.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.trainer_config.default_hdfs_dir is None
            else os.path.join(self.trainer_config.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.trainer_config.remove_previous_ckpt_in_save
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = self.trainer_config.max_actor_ckpt_to_keep if not remove_previous_ckpt_in_save else 1
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        checkpoint_config = self.config.actor_rollout_ref.actor.checkpoint
        async_save = bool(getattr(checkpoint_config, "async_save", False) or checkpoint_config.get("async_save", False))
        if async_save:
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return

        local_mkdir_safe(self.trainer_config.default_local_dir)
        local_latest_checkpointed_iteration = os.path.join(
            self.trainer_config.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.trainer_config.resume_mode == "disable":
            return 0

        if self.trainer_config.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")

        checkpoint_folder = self.trainer_config.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.trainer_config.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        elif self.trainer_config.resume_mode == "resume_path":
            assert isinstance(self.trainer_config.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.trainer_config.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.trainer_config.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.trainer_config.del_local_ckpt_after_load
        )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def fit(self):
        if self.actor_rollout_wg is None:
            raise RuntimeError("Workers are not initialized. Call `init_workers()` before `fit()`.")

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.trainer_config.project_name,
            experiment_name=self.trainer_config.experiment_name,
            default_backend=self.trainer_config.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        total_epochs = int(self.trainer_config.total_epochs)
        steps_per_epoch = len(self.sft_dataloader)
        self.total_training_steps = total_epochs * steps_per_epoch
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.max_steps_duration = 0.0

        current_epoch = self.global_steps // steps_per_epoch
        for epoch in range(current_epoch, total_epochs):
            train_iter = iter(self.sft_dataloader)
            start_step_in_epoch = self.global_steps % steps_per_epoch
            actor_output_future = None
            for step_in_epoch in range(start_step_in_epoch, steps_per_epoch):
                timing_raw = {}

                with marked_timer("step", timing_raw):
                    # ensure actor_output_future is ready
                    if actor_output_future is None:
                        actor_output_future = self._submit_sft_update(train_iter, timing_raw)

                    current_step = self.global_steps + 1
                    is_last_step = current_step >= self.total_training_steps
                    has_next_batch = step_in_epoch + 1 < steps_per_epoch
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.trainer_config.esi_redundant_time,
                    )
                    save_last = self.trainer_config.save_last
                    should_save = (
                        self.trainer_config.save_freq > 0
                        and (
                            is_last_step or current_step % self.trainer_config.save_freq == 0 or esi_close_to_expiration
                        )
                    ) or (save_last and is_last_step)

                    # submit the next update
                    next_actor_output_future = None
                    if has_next_batch and not should_save:
                        next_actor_output_future = self._submit_sft_update(train_iter, timing_raw)

                    # wait for the current update
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = actor_output_future.get()

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    self.global_steps += 1

                    if should_save:
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    actor_output_future = next_actor_output_future

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics = {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
                metrics.update(actor_output_metrics)
                metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})

                progress_bar.update(1)
                progress_bar.set_postfix(
                    sft_loss=f"{metrics.get('sft/loss', 0.0):.4f}",
                    grad_pre=f"{metrics.get('sft/grad_norm', 0.0):.4f}",
                )
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    progress_bar.close()
                    return

        progress_bar.close()

    def _submit_sft_update(self, train_iter, timing_raw):
        with marked_timer("data_loading", timing_raw):
            batch = next(train_iter)
            batch_proto = self._batch_to_dataproto(batch)

        with marked_timer("update_actor", timing_raw, color="red"):
            return self.actor_rollout_wg.update_actor_async(batch_proto)

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
