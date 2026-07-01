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

import ray
from omegaconf import OmegaConf
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

logger = logging.getLogger(__name__)


def ensure_ray_initialized(config) -> None:
    if ray.is_initialized():
        return

    default_runtime_env = get_ppo_ray_runtime_env()
    ray_kwargs = OmegaConf.select(config, "ray_kwargs", default={})
    ray_init_kwargs = ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    logger.info("ray init kwargs: %s", ray_init_kwargs)
    ray.init(**OmegaConf.to_container(ray_init_kwargs, resolve=True))


def get_controller_remote_options(config) -> dict:
    controller_label = OmegaConf.select(config, "cluster.resource.controller_label", default=None)
    return {} if controller_label is None else {"resources": {controller_label: 0.001}}
