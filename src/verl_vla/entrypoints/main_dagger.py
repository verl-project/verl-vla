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
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.trainer.train_cluster import TrainCluster
from verl_vla.utils.ray_utils import ensure_ray_initialized
from verl_vla.utils.recorder import (
    is_lerobot_dataset,
    merge_lerobot_datasets,
    move_lerobot_dataset_to_output,
    prepare_lerobot_output_root,
)
from verl_vla.utils.rollout_collection import collect_lerobot_rollout_dataset

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main_dagger", version_base=None)
def main(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)

    target_episodes = int(config.max_episodes)
    if target_episodes <= 0:
        raise ValueError(f"max_episodes must be positive, got {target_episodes}.")

    resume = bool(config.resume)
    recorder_cfg = config.cluster.env.env_worker.recorder
    dataset_root = Path(recorder_cfg.lerobot.root) / str(recorder_cfg.lerobot.repo_id)
    initial_episodes = prepare_lerobot_output_root(dataset_root, resume=resume)
    print(
        f"DAgger recording progress: {initial_episodes}/{target_episodes} episodes at {dataset_root} (resume={resume})"
    )
    if initial_episodes >= target_episodes:
        print(f"DAgger dataset already has {initial_episodes}/{target_episodes} episodes: {dataset_root}")
        return

    ensure_ray_initialized(config)
    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    collected_datasets: dict[str, dict[str, Any]] = {}
    try:
        collected_datasets = collect_lerobot_rollout_dataset(
            cluster,
            target_episodes=target_episodes,
            initial_completed_episodes=initial_episodes,
            log_prefix="DAgger rollout",
            max_episodes_name="max_episodes",
            log=logger,
        )
    except KeyboardInterrupt:
        print("DAgger interrupted; finalizing any completed collected dataset before shutdown.")
    finally:
        _finalize_dagger_collected_dataset(
            collected_datasets,
            recorder_cfg=recorder_cfg,
            dataset_root=dataset_root,
            resume=resume,
        )
        cluster.shutdown()


def _finalize_dagger_collected_dataset(
    collected_datasets: dict[str, dict[str, Any]],
    *,
    recorder_cfg,
    dataset_root: Path,
    resume: bool,
) -> Path | None:
    target_repo_id = str(recorder_cfg.lerobot.repo_id)
    collected_dataset = collected_datasets.get("collected_dataset")
    if collected_dataset is None:
        collected_repo_id = f"{target_repo_id}_collected"
        collected_root = Path(recorder_cfg.lerobot.root) / collected_repo_id
        if not is_lerobot_dataset(collected_root):
            print(f"No complete DAgger collected dataset to finalize at: {collected_root}")
            return None
        collected_dataset = {"root": collected_root, "repo_id": collected_repo_id}

    collected_root = Path(collected_dataset["root"])
    if not is_lerobot_dataset(collected_root):
        print(f"No complete DAgger collected dataset to finalize at: {collected_root}")
        return None

    print(f"DAgger collected dataset before finalizing: {collected_root}")
    if resume:
        output_dataset = merge_lerobot_datasets(
            roots=[collected_root],
            output_root=dataset_root,
            repo_id=target_repo_id,
            repo_ids=[str(collected_dataset["repo_id"])],
            append=True,
            video_files_size_in_mb=recorder_cfg.lerobot.video_files_size_in_mb,
        )
        output_root = Path(output_dataset["root"])
        print(f"DAgger dataset appended to: {output_root}")
        return output_root

    output_root = move_lerobot_dataset_to_output(collected_root, dataset_root)
    print(f"DAgger dataset saved to: {output_root}")
    return output_root


if __name__ == "__main__":
    main()
