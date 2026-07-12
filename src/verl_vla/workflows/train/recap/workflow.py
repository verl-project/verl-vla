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

from omegaconf import OmegaConf

from verl_vla.workflows.train.recap.collect_data import collect_recap_env_data
from verl_vla.workflows.train.recap.compute_return import (
    CollectedDatasets,
    ensure_recap_fields,
    merge_recap_collected_dataset_into_sft_dataset,
)
from verl_vla.workflows.train.recap.config import MainReCapConfig
from verl_vla.workflows.train.recap.policy_eval import eval_recap_policy
from verl_vla.workflows.train.recap.train_policy import train_recap_policy
from verl_vla.workflows.train.recap.train_value_model import train_recap_value_model
from verl_vla.workflows.train.recap.value_infer import infer_recap_values


def _configure_iteration_sft_stage(config, stage_name: str, iteration: int) -> None:
    stage_cfg = OmegaConf.select(config, f"recap.{stage_name}")
    if stage_cfg is None:
        return

    iteration_suffix = f"iter_{iteration:04d}"
    trainer_path = f"recap.{stage_name}.trainer"

    base_experiment_name = str(
        OmegaConf.select(stage_cfg, "_base_experiment_name", default=stage_cfg.trainer.experiment_name)
    )

    OmegaConf.update(config, f"recap.{stage_name}._base_experiment_name", base_experiment_name, force_add=True)
    OmegaConf.update(config, f"{trainer_path}.experiment_name", f"{base_experiment_name}_{iteration_suffix}")


def run_recap(config):
    recap_config = MainReCapConfig.from_omega_conf(config.recap)

    policy_path_from_previous_iteration = None
    for iteration_idx in range(recap_config.resume_iteration - 1, recap_config.num_iterations):
        iteration = iteration_idx + 1
        print(f"Starting ReCap iteration {iteration}/{recap_config.num_iterations}")
        _configure_iteration_sft_stage(config, "train_value_model", iteration)
        _configure_iteration_sft_stage(config, "train_policy", iteration)

        # Step 1: evaluate the configured or previous-iteration RECAP policy on the environment benchmark.
        if recap_config.should_run_stage(iteration, 1) and recap_config.stage_enabled("policy_eval", default=False):
            policy_path = policy_path_from_previous_iteration or OmegaConf.select(
                config,
                "recap.policy_eval.model_path",
            )
            disable_eval_acp = iteration_idx == 0 and bool(
                OmegaConf.select(config, "recap.policy_eval.disable_acp_on_first_iteration", default=True)
            )
            metrics = eval_recap_policy(config, str(policy_path), disable_acp=disable_eval_acp)
            print(f"ReCap policy eval finished: {metrics}")

        # Step 2: collect rollout data into a LeRobot dataset.
        if recap_config.should_run_stage(iteration, 2) and recap_config.stage_enabled("collect_data", default=True):
            collect_policy_path = (
                None if policy_path_from_previous_iteration is None else str(policy_path_from_previous_iteration)
            )
            collected_datasets: CollectedDatasets | None = collect_recap_env_data(
                config,
                collect_policy_path,
            )
        else:
            collected_datasets = None

        # Step 3: add RECAP return fields to the collected or configured dataset.
        if recap_config.should_run_stage(iteration, 3) and recap_config.stage_enabled("compute_return", default=True):
            if collected_datasets is None:
                dataset_cfg = config.recap.compute_return.dataset
                collected_datasets = {
                    "collected_dataset": {
                        "root": str(dataset_cfg.root),
                        "repo_id": str(dataset_cfg.repo_id),
                    }
                }
            collected_datasets = ensure_recap_fields(config, collected_datasets)
            collected_datasets = merge_recap_collected_dataset_into_sft_dataset(config, collected_datasets)
        else:
            collected_datasets = None

        # Step 4: train the RECAP value model with the SFT trainer.
        if recap_config.should_run_stage(iteration, 4) and recap_config.stage_enabled(
            "train_value_model", default=True
        ):
            if collected_datasets is None:
                dataset_cfg = config.recap.train_value_model.data
                collected_datasets = {
                    "collected_dataset": {
                        "root": str(dataset_cfg.root),
                        "repo_id": str(dataset_cfg.repo_id),
                    }
                }
            value_model_path = train_recap_value_model(config, collected_datasets)
            print(f"ReCap value-model training finished: {value_model_path}")
        else:
            value_model_path = None

        # Step 5: infer RECAP values and write them back to the LeRobot dataset.
        if recap_config.should_run_stage(iteration, 5) and recap_config.stage_enabled("value_infer", default=True):
            if collected_datasets is None:
                dataset_cfg = config.recap.value_infer.dataset
                collected_datasets = {
                    "collected_dataset": {
                        "root": str(dataset_cfg.root),
                        "repo_id": str(dataset_cfg.repo_id),
                    }
                }
            dataset = collected_datasets.get("collected_dataset") or collected_datasets.get("existing_dataset")
            if dataset is None:
                raise ValueError("RECAP value inference is enabled but no LeRobot dataset was collected or configured.")
            model_path = value_model_path or str(config.recap.value_infer.model_path)
            value_infer_metrics = infer_recap_values(config, dataset, model_path)
            print(f"ReCap value inference finished: {value_infer_metrics}")

        # Step 6: train the final RECAP policy with the SFT trainer.
        if recap_config.should_run_stage(iteration, 6) and recap_config.stage_enabled("train_policy", default=True):
            if collected_datasets is None:
                dataset_cfg = config.recap.train_policy.dataset
                collected_datasets = {
                    "collected_dataset": {
                        "root": str(dataset_cfg.root),
                        "repo_id": str(dataset_cfg.repo_id),
                    }
                }
            policy_path = train_recap_policy(config, collected_datasets)
            print(f"ReCap policy training finished: {policy_path}")
            if policy_path is not None:
                policy_path_from_previous_iteration = policy_path
