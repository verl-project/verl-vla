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

from verl_vla.workflows.train.sft import run_sft


def _select_value_model_dataset(collected_datasets):
    if collected_datasets is None:
        raise ValueError("RECAP value-model training requires a collected or configured LeRobot dataset.")
    dataset = collected_datasets.get("collected_dataset") or collected_datasets.get("existing_dataset")
    if dataset is None:
        raise ValueError("RECAP value-model training is enabled but no LeRobot dataset was collected or found.")
    return dataset


def _find_latest_value_model_hf_checkpoint(value_model_config) -> Path:
    checkpoint_root = Path(str(value_model_config.cluster.checkpoint.default_local_dir))
    if not checkpoint_root.is_absolute():
        checkpoint_root = Path.cwd() / checkpoint_root

    latest_file = checkpoint_root / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        global_step = latest_file.read_text().strip()
        candidates = [checkpoint_root / f"global_step_{global_step}"]
    else:
        candidates = sorted(
            [path for path in checkpoint_root.glob("global_step_*") if path.is_dir()],
            key=lambda path: int(path.name.split("global_step_")[-1]),
        )

    for step_dir in reversed(candidates):
        hf_path = step_dir / "actor" / "huggingface"
        if hf_path.exists():
            return hf_path
    raise FileNotFoundError(f"No value-model HF checkpoint found under {checkpoint_root}.")


def _build_value_model_sft_config(config, collected_datasets):
    sft_config_node = OmegaConf.select(config, "recap.train_value_model")
    if sft_config_node is None:
        raise ValueError("`recap.train_value_model` is required when RECAP value-model training is enabled.")

    sft_config = OmegaConf.create(OmegaConf.to_container(sft_config_node, resolve=True))
    OmegaConf.set_struct(sft_config, False)

    dataset = _select_value_model_dataset(collected_datasets)
    OmegaConf.update(sft_config, "data.repo_id", str(dataset["repo_id"]))
    OmegaConf.update(sft_config, "data.root", str(dataset["root"]))
    return_field = str(config.recap.compute_return.fields["return"])
    OmegaConf.update(sft_config, "cluster.actor_rollout_ref.data_keys.action_mask", None)
    OmegaConf.update(sft_config, "cluster.actor_rollout_ref.data_keys.target_value", return_field)

    OmegaConf.resolve(sft_config)
    return sft_config


def train_recap_value_model(config, collected_datasets) -> str:
    sft_config = _build_value_model_sft_config(config, collected_datasets)
    run_sft(sft_config)
    return str(_find_latest_value_model_hf_checkpoint(sft_config))
