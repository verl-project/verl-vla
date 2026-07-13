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

from pathlib import Path

from omegaconf import OmegaConf

from verl_vla.workflows.eval import run_eval, save_eval_metrics


def _build_policy_eval_config(config, policy_path: str, *, disable_acp: bool = False):
    eval_config_node = OmegaConf.select(config, "recap.policy_eval")
    if eval_config_node is None:
        raise ValueError("`recap.policy_eval` is required when RECAP policy evaluation is enabled.")

    eval_config = OmegaConf.create(OmegaConf.to_container(eval_config_node, resolve=False))
    OmegaConf.set_struct(eval_config, False)
    ray_kwargs = OmegaConf.select(config, "ray_kwargs", default={})
    OmegaConf.update(
        eval_config,
        "ray_kwargs",
        OmegaConf.to_container(ray_kwargs, resolve=False),
        force_add=True,
    )
    OmegaConf.update(eval_config, "cluster.actor_rollout_ref.model.path", str(policy_path))
    if disable_acp:
        OmegaConf.update(eval_config, "cluster.actor_rollout_ref.rollout.acp.enable", False)
    OmegaConf.resolve(eval_config)
    return eval_config


def _next_policy_eval_result_path(result_dir: Path) -> Path:
    result_idx = 1
    while True:
        result_path = result_dir / f"eval_{result_idx:04d}.json"
        if not result_path.exists():
            return result_path
        result_idx += 1


def _save_policy_eval_metrics(eval_config, metrics: dict[str, float]) -> Path | None:
    result_dir = OmegaConf.select(eval_config, "result_dir", default=None)
    if result_dir in (None, ""):
        return None

    result_root = Path(str(result_dir))
    result_path = _next_policy_eval_result_path(result_root)
    return save_eval_metrics(str(result_path), metrics)


def eval_recap_policy(
    config,
    policy_path: str,
    *,
    disable_acp: bool = False,
) -> dict[str, float]:
    eval_config = _build_policy_eval_config(config, policy_path, disable_acp=disable_acp)
    metrics = run_eval(eval_config, save_result=False, print_metrics=False)
    _save_policy_eval_metrics(eval_config, metrics)
    return metrics
