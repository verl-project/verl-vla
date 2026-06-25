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

import hydra
from omegaconf import OmegaConf

from verl_vla.trainer.recap.collect import collect_recap_env_data
from verl_vla.trainer.recap.policy import train_recap_policy
from verl_vla.trainer.recap.policy_eval import eval_recap_policy
from verl_vla.trainer.recap.returns import CollectedDatasets, DatasetInfo, ensure_recap_fields
from verl_vla.trainer.recap.value_infer import infer_recap_values
from verl_vla.trainer.recap.value_model import train_recap_value_model


@hydra.main(config_path="config", config_name="rob_recap_trainer", version_base=None)
def main(config):
    # Step 1: collect rollout data into a LeRobot dataset.
    if OmegaConf.select(config, "recap.collect.enable", default=True):
        collected_datasets: CollectedDatasets | None = collect_recap_env_data(config)
    else:
        collected_datasets = None

    # Step 2: add RECAP return fields to the collected or configured dataset.
    if OmegaConf.select(config, "recap.returns.enable", default=True):
        if collected_datasets is None:
            dataset_cfg = config.recap.returns.dataset
            collected_datasets = {
                "collected_dataset": {
                    "root": str(dataset_cfg.root),
                    "repo_id": str(dataset_cfg.repo_id),
                }
            }
        collected_datasets = ensure_recap_fields(config, collected_datasets)
    else:
        collected_datasets = None

    # Step 3: train the RECAP value model with the SFT trainer.
    if OmegaConf.select(config, "recap.value_model.enable", default=True):
        if collected_datasets is None:
            value_model_cfg = config.recap.value_model
            collected_datasets = {
                "collected_dataset": {
                    "root": str(value_model_cfg.root),
                    "repo_id": str(value_model_cfg.repo_id),
                }
            }
        value_model_path = train_recap_value_model(config, collected_datasets)
    else:
        value_model_path = None

    # Step 4: infer RECAP values and write them back to the LeRobot dataset.
    if OmegaConf.select(config, "recap.value_infer.enable", default=True):
        if value_model_path is None:
            value_model_path = OmegaConf.select(config, "recap.value_infer.model_path")
        if value_model_path is None:
            raise ValueError(
                "recap.value_infer.enable=True requires recap.value_model.enable=True or recap.value_infer.model_path."
            )
        if collected_datasets is None:
            value_infer_cfg = config.recap.value_infer
            collected_datasets = {
                "collected_dataset": {
                    "root": str(value_infer_cfg.root),
                    "repo_id": str(value_infer_cfg.repo_id),
                }
            }
        dataset: DatasetInfo = collected_datasets["collected_dataset"]
        metrics = infer_recap_values(config, dataset, value_model_path)
        print(f"ReCap value inference finished with metrics: {metrics}")

    # Step 5: train the final RECAP policy with the SFT trainer.
    if OmegaConf.select(config, "recap.policy.enable", default=True):
        if collected_datasets is None:
            policy_cfg = config.recap.policy
            collected_datasets = {
                "collected_dataset": {
                    "root": str(policy_cfg.root),
                    "repo_id": str(policy_cfg.repo_id),
                }
            }
        policy_path = train_recap_policy(config, collected_datasets)
        print("ReCap policy training finished.")
    else:
        policy_path = None

    # Step 6: evaluate the RECAP policy on the environment benchmark.
    if OmegaConf.select(config, "recap.policy_eval.enable", default=False):
        if policy_path is None:
            policy_path = OmegaConf.select(config, "recap.policy_eval.model_path")
        if policy_path is None:
            raise ValueError(
                "recap.policy_eval.enable=True requires recap.policy.enable=True or recap.policy_eval.model_path."
            )
        metrics = eval_recap_policy(config, policy_path)
        print(f"ReCap policy eval finished with metrics: {metrics}")


if __name__ == "__main__":
    main()
