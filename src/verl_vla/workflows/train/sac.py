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


from pprint import pprint

import ray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.train_cluster import TrainCluster
from verl_vla.trainer.sac.sac_ray_trainer import RobRaySACTrainer
from verl_vla.utils.ray_utils import ensure_ray_initialized, get_controller_remote_options


def run_sac(config):
    ensure_ray_initialized(config)
    remote_options = get_controller_remote_options(config)
    return ray.get(_run_sac_remote.options(**remote_options).remote(config))


@ray.remote
def _run_sac_remote(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)

    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        trainer = RobRaySACTrainer(
            trainer_config=config.trainer,
            cluster=cluster,
            tracking_config=OmegaConf.to_container(config, resolve=True),
        )
        trainer.fit()
    finally:
        cluster.shutdown()
