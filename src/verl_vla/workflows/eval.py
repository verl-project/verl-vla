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

import json
from pathlib import Path
from pprint import pprint

import ray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.train_cluster import TrainCluster
from verl_vla.utils.ray_utils import ensure_ray_initialized, get_controller_remote_options


def save_eval_metrics(result_path: str | None, metrics: dict[str, float]) -> Path | None:
    """Write evaluation metrics to JSON when an output path is configured."""

    if result_path in (None, ""):
        return None

    output_path = Path(result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
        file.write("\n")
    return output_path


def run_eval(
    config,
    *,
    save_result: bool = True,
    print_metrics: bool = True,
) -> dict[str, float]:
    """Evaluate a policy with a rollout-only TrainCluster."""

    ensure_ray_initialized(config)
    remote_options = get_controller_remote_options(config)
    metrics = ray.get(_run_eval_remote.options(**remote_options).remote(config))
    output_dir = OmegaConf.select(config, "output_dir", default=None) if save_result else None
    result_path = None if output_dir in (None, "") else Path(str(output_dir)) / "metrics.json"
    saved_path = save_eval_metrics(None if result_path is None else str(result_path), metrics)
    if print_metrics:
        pprint(metrics)
    if saved_path is not None:
        print(f"Evaluation metrics saved to {saved_path}")
    return metrics


@ray.remote
def _run_eval_remote(config) -> dict[str, float]:
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)

    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        max_episodes = OmegaConf.select(config, "max_episodes", default=None)
        return cluster.eval(max_episodes=None if max_episodes is None else int(max_episodes))
    finally:
        cluster.shutdown()
