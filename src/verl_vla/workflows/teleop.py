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

from hydra.utils import instantiate

from verl_vla.train_cluster import TrainCluster
from verl_vla.utils.ray_utils import ensure_ray_initialized


def run_teleop(config):
    ensure_ray_initialized(config)
    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    print("Teleop started. Press Ctrl+C to stop.")
    while True:
        cluster.record(collect_dataset=False)
        print("Teleop episode finished; resetting environment.")
