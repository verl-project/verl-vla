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

import os
from typing import Any

import torch
from hydra.utils import instantiate
from tqdm import tqdm
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.fs import local_mkdir_safe
from verl.utils.metric import reduce_metrics

from verl_vla.train_cluster import TrainCluster
from verl_vla.utils.data import dataloader_batch_to_dataproto
from verl_vla.utils.dataloader import LeRobotDataLoaderConfig, build_lerobot_sft_dataloader
from verl_vla.utils.dataloader.state import load_dataloader_state
from verl_vla.utils.early_stopping import TrendEarlyStopper

from .config import SFTTrainerConfig


class RobRaySFTTrainer(RayPPOTrainer):
    def __init__(
        self,
        data_config,
        trainer_config,
        cluster: TrainCluster,
        tracking_config: dict[str, Any],
    ):
        self.cluster = cluster
        self.tracking_config = tracking_config
        self.trainer_config: SFTTrainerConfig = instantiate(trainer_config)
        self.use_critic = False
        self.use_reference_policy = False
        self.data_config: LeRobotDataLoaderConfig = instantiate(data_config)
        self.sft_dataloader = build_lerobot_sft_dataloader(
            self.data_config,
            train_world_size=self.cluster.train_world_size,
        )
        self.train_dataloader = self.sft_dataloader

    def _save_checkpoint(self):
        def save_dataloader_state(local_global_step_folder: str) -> None:
            print(f"local_global_step_folder: {local_global_step_folder}")
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
            torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        self.cluster.save_checkpoint(self.global_steps, save_extra_state=save_dataloader_state)

    def _load_checkpoint(self):
        checkpoint_state = self.cluster.load_checkpoint()
        if checkpoint_state is None:
            return

        self.global_steps, checkpoint_dir = checkpoint_state
        if self.trainer_config.resume_dataloader_state:
            load_dataloader_state(self.train_dataloader, checkpoint_dir)

    def fit(self):
        from verl.utils.tracking import Tracking

        tracking_logger = Tracking(
            project_name=self.trainer_config.project_name,
            experiment_name=self.trainer_config.experiment_name,
            default_backend=self.trainer_config.logger,
            config=self.tracking_config,
        )

        self.global_steps = 0
        self._load_checkpoint()
        total_epochs = int(self.trainer_config.total_epochs)
        steps_per_epoch = len(self.sft_dataloader)
        self.epoch = self.global_steps // steps_per_epoch if steps_per_epoch > 0 else 0
        self.total_training_steps = total_epochs * steps_per_epoch
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.max_steps_duration = 0.0
        early_stopper = self._build_early_stopper()

        train_iter = iter(self.sft_dataloader)
        actor_output_future = None
        actor_output_epoch = self.epoch
        while self.global_steps < self.total_training_steps:
            timing_raw = {}

            with marked_timer("step", timing_raw):
                if actor_output_future is None:
                    actor_output_future, train_iter, actor_output_epoch = self._submit_sft_update(
                        train_iter,
                        timing_raw,
                    )

                is_last_step, should_save = self._get_step_control_flags()
                next_actor_output_future = None
                next_actor_output_epoch = self.epoch
                if not is_last_step and not should_save:
                    next_actor_output_future, train_iter, next_actor_output_epoch = self._submit_sft_update(
                        train_iter,
                        timing_raw,
                    )

                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = actor_output_future.get()

                completed_epoch = actor_output_epoch
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                actor_output_future = next_actor_output_future
                actor_output_epoch = next_actor_output_epoch
                self.global_steps += 1

                if should_save:
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            metrics = {
                "training/global_step": self.global_steps,
                "training/epoch": completed_epoch,
            }
            metrics.update(actor_output_metrics)

            early_stop = self._update_early_stopper(early_stopper, metrics)
            if early_stop:
                metrics.update(early_stop)
                if not should_save:
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})

            progress_bar.update(1)
            postfix = {
                "sft_loss": f"{metrics.get('sft/loss', 0.0):.4f}",
                "grad_pre": f"{metrics.get('sft/grad_norm', 0.0):.4f}",
            }
            if early_stopper is not None:
                threshold_progress = metrics.get("sft/early_stop/threshold_progress", 0.0)
                postfix["early_stop"] = f"{threshold_progress * 100.0:.1f}%"
            progress_bar.set_postfix(**postfix)
            tracking_logger.log(data=metrics, step=self.global_steps)

            if early_stop:
                progress_bar.set_postfix(
                    sft_loss=f"{metrics.get('sft/loss', 0.0):.4f}",
                    early_stop="true",
                )
                progress_bar.close()
                return

        progress_bar.close()

    def _build_early_stopper(self) -> TrendEarlyStopper | None:
        early_stopping = self.trainer_config.early_stopping
        if not early_stopping.enable:
            return None
        return TrendEarlyStopper(early_stopping)

    def _update_early_stopper(
        self,
        early_stopper: TrendEarlyStopper | None,
        metrics: dict[str, float],
    ) -> dict[str, float] | None:
        if early_stopper is None:
            return None

        early_stopping = self.trainer_config.early_stopping
        metric_name = early_stopping.metric
        metric_value = metrics.get(metric_name)
        if metric_value is None:
            return None

        early_stop_metrics = early_stopper.update(float(metric_value))
        metrics.update({f"sft/early_stop/{key}": value for key, value in early_stop_metrics.items()})

        if not early_stopper.should_stop:
            return None

        print(f"Early stopping triggered at step {self.global_steps}.")
        return {
            "sft/early_stop/triggered": 1.0,
        }

    def _get_step_control_flags(self) -> tuple[bool, bool]:
        current_step = self.global_steps + 1
        is_last_step = current_step >= self.total_training_steps
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.trainer_config.esi_redundant_time,
        )
        should_save = (
            self.trainer_config.save_freq > 0
            and (is_last_step or current_step % self.trainer_config.save_freq == 0 or esi_close_to_expiration)
        ) or (self.trainer_config.save_last and is_last_step)
        return is_last_step, should_save

    def _submit_sft_update(self, train_iter, timing_raw):
        with marked_timer("data_loading", timing_raw):
            batch_epoch = self.epoch
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                batch_epoch = self.epoch
                train_iter = iter(self.sft_dataloader)
                batch = next(train_iter)
            batch_proto = dataloader_batch_to_dataproto(batch)

        with marked_timer("update_actor", timing_raw, color="red"):
            return self.cluster.train(batch_proto, async_update=True), train_iter, batch_epoch
