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
from pprint import pprint

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.trainer.train_cluster import TrainCluster
from verl_vla.utils.ray_utils import ensure_ray_initialized

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main_record", version_base=None)
def main(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)

    ensure_ray_initialized(config)
    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        num_episodes = int(config.num_episodes)
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}.")
        dataset_root = None
        for episode_idx in range(num_episodes):
            print(f"Recording episode {episode_idx + 1}/{num_episodes}")
            dataset_root = cluster.record()
        assert dataset_root is not None
        print(f"Record dataset saved to: {dataset_root}")
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()
