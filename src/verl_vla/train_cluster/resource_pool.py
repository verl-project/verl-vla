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

"""Ray resource-pool management used by TrainCluster."""

from __future__ import annotations

from dataclasses import dataclass, field

import ray
from ray.util.placement_group import placement_group
from verl.single_controller.ray.base import RayResourcePool, sort_placement_group_by_node_ip
from verl.trainer.ppo.ray_trainer import ResourcePoolManager

__all__ = ["VLAResourcePoolManager"]


class VLARayResourcePool(RayResourcePool):
    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        if self.pgs is not None:
            return self.pgs

        pool_shape = "_".join(str(process_count) for process_count in self._store)
        pg_name_prefix = name or f"{self.name_prefix}verl_group_{pool_shape}:"
        ray_device_name = {"cuda": "GPU", "npu": "NPU"}.get(device_name, device_name)

        resource_bundle = {"CPU": self.max_colocate_count}
        if self.use_gpu:
            resource_bundle[ray_device_name] = 1
        if self.accelerator_type is not None:
            resource_bundle[self.accelerator_type] = 1e-4

        bundle_groups = [[resource_bundle.copy() for _ in range(process_count)] for process_count in self._store]
        lifetime = "detached" if self.detached else None
        placement_groups = [
            placement_group(
                bundles=bundles,
                strategy=strategy,
                name=f"{pg_name_prefix}{idx}",
                lifetime=lifetime,
            )
            for idx, bundles in enumerate(bundle_groups)
        ]

        ray.get([pg.ready() for pg in placement_groups])
        self.pgs = sort_placement_group_by_node_ip(placement_groups)
        return self.pgs


@dataclass
class VLAResourcePoolManager(ResourcePoolManager):
    cpu_pool_names: set[str] = field(default_factory=set)
    resource_labels: dict[str, str] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = VLARayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=resource_pool_name not in self.cpu_pool_names,
                max_colocate_count=3,
                name_prefix=resource_pool_name,
                accelerator_type=self.resource_labels.get(resource_pool_name),
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def _check_resource_available(self):
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            n_gpus
            for pool_name, process_on_nodes in self.resource_pool_spec.items()
            if pool_name not in self.cpu_pool_names
            for n_gpus in process_on_nodes
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )
