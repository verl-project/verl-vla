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

import logging
from typing import Optional, cast

import hydra
import ray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.trainer.recap.compute_return import CollectedDatasets
from verl_vla.trainer.train_cluster import TrainCluster
from verl_vla.utils.lerobot import count_lerobot_episodes, truncate_lerobot_episodes
from verl_vla.utils.ray_utils import ensure_ray_initialized, get_controller_remote_options

logger = logging.getLogger(__name__)


def get_collect_config(config):
    return OmegaConf.select(config, "recap.collect_data", default=config)


def collect_recap_env_data(config, policy_path: Optional[str] = None) -> CollectedDatasets:
    """Run RECAP env data collection and return existing/new datasets."""
    ensure_ray_initialized(config)
    collect_config = get_collect_config(config)
    remote_options = get_controller_remote_options(collect_config)
    return cast(CollectedDatasets, ray.get(run_env_loop.options(**remote_options).remote(config, policy_path)))


@ray.remote
def run_env_loop(config, policy_path: Optional[str] = None):
    OmegaConf.set_struct(config, False)
    if policy_path is not None:
        OmegaConf.update(config, "recap.collect_data.cluster.actor_rollout_ref.model.path", policy_path)
    OmegaConf.resolve(config)

    collect_config = get_collect_config(config)
    cluster = TrainCluster(instantiate(collect_config.cluster, _recursive_=False))
    cluster.start()
    try:
        collected_datasets = {}
        target_episodes = int(collect_config.max_episodes)
        if target_episodes <= 0:
            raise ValueError(f"recap.collect_data.max_episodes must be positive, got {target_episodes}.")

        completed_episodes = 0
        rollout_idx = 0
        while completed_episodes < target_episodes:
            rollout_output, collected_datasets = cluster.rollout()
            collected_dataset = collected_datasets.get("collected_dataset")
            metrics = rollout_output.meta_info.get("metrics", {})
            if collected_dataset is None:
                logger.info(
                    "Finished recap env loop rollout %s without completed trajectories: "
                    "collected_episodes=%s/%s, dataset_keys=%s, metrics=%s",
                    rollout_idx,
                    completed_episodes,
                    target_episodes,
                    list(collected_datasets.keys()),
                    metrics,
                )
                rollout_idx += 1
                continue

            previous_completed_episodes = completed_episodes
            completed_episodes = count_lerobot_episodes(collected_dataset["root"])
            logger.info(
                "Finished recap env loop rollout %s: collected_episodes=%s/%s, metrics=%s",
                rollout_idx,
                completed_episodes,
                target_episodes,
                metrics,
            )
            if completed_episodes <= previous_completed_episodes:
                logger.info(
                    "Recap env loop rollout %s did not add completed trajectories; continuing.",
                    rollout_idx,
                )
                rollout_idx += 1
                continue
            if completed_episodes > target_episodes:
                dataset_root = collected_datasets["collected_dataset"]["root"]
                truncate_lerobot_episodes(dataset_root, target_episodes)
                completed_episodes = target_episodes
            rollout_idx += 1

        return collected_datasets
    finally:
        cluster.shutdown()


@hydra.main(config_path="../config", config_name="rob_recap_trainer", version_base=None)
def main(config):
    collected_datasets = collect_recap_env_data(config)
    logger.info("RECAP collect finished: %s", collected_datasets)
    print(collected_datasets)


if __name__ == "__main__":
    main()
