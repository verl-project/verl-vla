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
from pathlib import Path
from pprint import pprint

from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.recorder import prepare_lerobot_output_root
from verl_vla.train_cluster import TrainCluster
from verl_vla.utils.ray_utils import ensure_ray_initialized

logger = logging.getLogger(__name__)


def run_record(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)

    num_episodes = int(config.num_episodes)
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}.")

    resume = bool(config.resume)
    recorder_cfg = config.cluster.env.env_worker.recorder
    target_repo_id = str(recorder_cfg.lerobot.repo_id)
    dataset_root = Path(recorder_cfg.lerobot.root) / target_repo_id
    initial_episodes = prepare_lerobot_output_root(dataset_root, resume=resume)
    if initial_episodes >= num_episodes:
        print(f"Record dataset already has {initial_episodes}/{num_episodes} episodes: {dataset_root}")
        return

    ensure_ray_initialized(config)
    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        dataset_root = None
        for episode_idx in range(initial_episodes, num_episodes):
            print(f"Recording episode {episode_idx + 1}/{num_episodes}")
            dataset_root = cluster.record()
        assert dataset_root is not None
        if resume:
            print(f"Record dataset appended to: {dataset_root}")
        else:
            print(f"Record dataset saved to: {dataset_root}")
    finally:
        cluster.shutdown()
