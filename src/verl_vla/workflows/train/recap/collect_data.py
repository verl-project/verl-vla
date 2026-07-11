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

import ray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.recorder.collection import collect_lerobot_rollout_dataset
from verl_vla.train_cluster import TrainCluster
from verl_vla.trainer.recap.compute_return import CollectedDatasets
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
        target_episodes = int(collect_config.max_episodes)
        return collect_lerobot_rollout_dataset(
            cluster,
            target_episodes=target_episodes,
            log_prefix="recap env loop rollout",
            max_episodes_name="recap.collect_data.max_episodes",
            log=logger,
        )
    finally:
        cluster.shutdown()
