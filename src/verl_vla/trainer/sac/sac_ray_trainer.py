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

import math
from pprint import pprint
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from verl_vla.train_cluster import TrainCluster
from verl_vla.utils.data import (
    _build_sac_transition_masks,
    add_transition_prefixes,
    flatten_trajectories,
    reduce_substep_dims,
)
from verl_vla.utils.rlpd import (
    iter_rlpd_replay_prefill_batches,
    pad_dataproto_to_divisor_with_valid_mask,
)

from .config import SACTrainerConfig
from .episode_buffer import EpisodeBuffer


def prepare_sac_actor_input(
    episode: DataProto,
    *,
    trainer_config: SACTrainerConfig,
    global_steps: int,
) -> DataProto:
    """Convert a complete raw episode into SAC transitions."""
    terminated_steps = reduce_substep_dims(episode.batch["next.terminated"].bool(), reduction="any")
    truncated_steps = reduce_substep_dims(episode.batch["next.truncated"].bool(), reduction="any")
    done_steps = terminated_steps | truncated_steps
    success_steps = reduce_substep_dims(episode.batch["next.success"].bool(), reduction="any")
    reward_steps = reduce_substep_dims(episode.batch["next.reward"].float(), reduction="sum")
    del episode.batch["next.terminated"]
    del episode.batch["next.truncated"]
    del episode.batch["next.success"]
    del episode.batch["next.reward"]

    valid_mask, success_mask = _build_sac_transition_masks(done_steps, success_steps)
    episode.batch["info.terminateds"] = terminated_steps.float()
    episode.batch["info.valids"] = valid_mask.float()
    episode.batch["info.rewards"] = (reward_steps - float(trainer_config.step_penalty)) * valid_mask.float()
    episode.batch["info.success_mask"] = success_mask.float()
    episode.meta_info["global_steps"] = global_steps

    return flatten_trajectories(add_transition_prefixes(episode, transition_boundary_mask=done_steps))


class RobRaySACTrainer:
    def __init__(
        self,
        trainer_config,
        cluster: TrainCluster,
        tracking_config: dict[str, Any],
    ):
        self.cluster = cluster
        self.tracking_config = tracking_config
        self.trainer_config: SACTrainerConfig = instantiate(trainer_config)
        self.config = OmegaConf.create(tracking_config)

        auto_reset = bool(OmegaConf.select(self.config, "cluster.env.env_worker.auto_reset", default=False))
        self._episode_buffer = EpisodeBuffer(auto_reset=auto_reset)

    def _prepare_actor_input(self, rollout_output: DataProto) -> DataProto | None:
        """Collect complete episodes and turn them into SAC transitions."""
        episodes = self._episode_buffer.ingest(rollout_output)
        if not episodes:
            return None

        parts = [
            prepare_sac_actor_input(
                episode,
                trainer_config=self.trainer_config,
                global_steps=self.global_steps,
            )
            for episode in episodes
        ]
        actor_input = parts[0] if len(parts) == 1 else DataProto.concat(parts)
        actor_input.meta_info.update(rollout_output.meta_info)
        actor_input.meta_info["global_steps"] = self.global_steps
        actor_input.meta_info.setdefault("global_token_num", [0])
        return pad_dataproto_to_divisor_with_valid_mask(
            actor_input,
            int(self.cluster.actor_worker_group.world_size),
            valid_key="info.valids",
        )

    def _prefill_replay_pool_from_rlpd(self) -> None:
        rlpd_config = self.trainer_config.rlpd
        if not rlpd_config.enable:
            return

        for prefill_batch in iter_rlpd_replay_prefill_batches(rlpd_config, global_steps=self.global_steps):
            self._submit_rlpd_prefill_batch(prefill_batch)

    def _submit_rlpd_prefill_batch(self, prefill_batch: DataProto) -> None:
        prefill_batch = pad_dataproto_to_divisor_with_valid_mask(
            prefill_batch,
            int(self.cluster.actor_worker_group.world_size),
            valid_key="info.valids",
        )
        prefill_batch.meta_info["global_steps"] = self.global_steps
        prefill_batch.meta_info["global_token_num"] = [0]
        prefill_batch.meta_info["add_to_offline_replay_only"] = True
        self.cluster.actor_worker_group.add_offline_replay_data(prefill_batch)

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.trainer_config.project_name,
            experiment_name=self.trainer_config.experiment_name,
            default_backend=self.trainer_config.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        checkpoint_state = self.cluster.load_checkpoint()
        if checkpoint_state is not None:
            self.global_steps, _checkpoint_dir = checkpoint_state
        self._prefill_replay_pool_from_rlpd()

        # perform evaluation before training
        # currently, we only support evaluation using the reward_function.
        if self.trainer_config.val_before_train:
            val_metrics = self.cluster.eval()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial evaluation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.trainer_config.val_only:
                return

        rollout_times = int(self.trainer_config.rollout_times)
        rollout_interval = int(self.trainer_config.rollout_interval)
        actor_config = self.config.cluster.actor_rollout_ref.actor
        profiler_config = OmegaConf.select(self.config, "global_profiler", default=None)
        critic_only_steps_after_rollout = int(actor_config.critic.only_steps_after_rollout)

        self.total_training_steps = int(self.trainer_config.total_training_steps)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in profiler_config.steps
            if profiler_config is not None and profiler_config.steps is not None
            else False
        )
        next_step_profile = False

        total_rollout_windows = math.ceil(self.total_training_steps / rollout_interval)
        for rollout_window in range(total_rollout_windows):
            print(f"Starting rollout window {rollout_window}")
            for training_step in range(rollout_interval):
                metrics = {}
                timing_raw = {}

                # === start profiling ===
                with marked_timer("start_profile", timing_raw):
                    do_profile = (
                        not prev_step_profile and curr_step_profile
                        if profiler_config is not None and profiler_config.profile_continuous_steps
                        else curr_step_profile
                    )
                    if do_profile:
                        self.cluster.start_profiling(step=self.global_steps)

                with marked_timer("step", timing_raw):
                    # === rollout ===
                    # Determine whether to perform rollout:
                    # enable at start and early warmup, disable during critic warmup phase
                    warm_rollout_steps = int(self.trainer_config.warm_rollout_steps)
                    need_rollout = (training_step < rollout_times) or self.global_steps < warm_rollout_steps
                    if warm_rollout_steps <= self.global_steps < actor_config.critic.warmup_steps:
                        need_rollout = False
                    actor_input = None
                    if need_rollout:
                        with marked_timer("rollout", timing_raw):
                            with marked_timer("generate", timing_raw, color="red"):
                                rollout_output, _collected_datasets, rollout_metrics = self.cluster.rollout(
                                    async_rollout=self.trainer_config.async_rollout,
                                )

                            # compute rewards and other metrics, and prepare for actor update
                            metrics.update(rollout_metrics)
                            actor_input = self._prepare_actor_input(rollout_output)
                    # === update policy ===
                    critic_only_update = training_step < rollout_times + critic_only_steps_after_rollout
                    with marked_timer("update_actor", timing_raw, color="red"):
                        if actor_input is not None:
                            actor_input.meta_info["critic_only_update"] = critic_only_update
                            actor_output = self.cluster.train(actor_input, async_update=False)
                        else:
                            actor_output = self.cluster.train(
                                DataProto(
                                    meta_info={
                                        "empty_batch": True,
                                        "global_steps": self.global_steps,
                                        "critic_only_update": critic_only_update,
                                    }
                                ),
                                async_update=False,
                            )
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # === eval ===
                is_last_step = self.global_steps >= self.total_training_steps
                if (
                    self.trainer_config.test_freq > 0
                    and (is_last_step or self.global_steps % self.trainer_config.test_freq == 0)
                    and self.global_steps >= actor_config.critic.warmup_steps
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self.cluster.eval()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    self._episode_buffer.clear()

                # === save checkpoint ===
                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.trainer_config.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.trainer_config.save_freq > 0 and (
                    is_last_step or self.global_steps % self.trainer_config.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self.cluster.save_checkpoint(self.global_steps)

                # === stop profiling ===
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in profiler_config.steps
                        if profiler_config is not None and profiler_config.steps is not None
                        else False
                    )
                    do_profile = (
                        curr_step_profile and not next_step_profile
                        if profiler_config is not None and profiler_config.profile_continuous_steps
                        else curr_step_profile
                    )
                    if do_profile:
                        self.cluster.stop_profiling()
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # === training metrics ===
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/rollout_window": rollout_window,
                    }
                )
                metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})
                if actor_input is not None:
                    metrics.update(
                        {key: value for key, value in actor_input.meta_info.items() if key.startswith("data/")}
                    )
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if hasattr(actor_config, "profiler") and actor_config.profiler.tool == "torch_memory":
                    self.cluster.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final evaluation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
