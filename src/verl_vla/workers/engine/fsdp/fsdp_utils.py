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
from torch.distributed.fsdp import fully_shard
from verl.utils.fsdp_utils import CPUOffloadPolicy, maybe_patch_fsdp_module


def _select_fsdp2_wrap_targets(model, transformer_layer_cls_to_wrap):
    # Deliberately exclude the generic nn.Embedding/lm_head targets selected by
    # Verl. Native VLA policies may access those parameters without calling the
    # owning module, so their FSDP forward hooks would never unshard the weight.
    return [module for module in model.modules() if module.__class__.__name__ in transformer_layer_cls_to_wrap]


def apply_fsdp2(model, fsdp_kwargs, config):
    """Apply FSDP2 with wrapping targets that are safe for native VLA policies.

    Verl's generic implementation independently wraps every ``nn.Embedding``.
    That assumes parameters are read through the embedding module's forward,
    where FSDP can run its unshard hook. Native policies such as LeRobot ACT
    instead read learned embeddings through ``.weight``. Separately sharding
    those modules leaves DTensors in code that otherwise receives materialized
    tensors and causes mixed Tensor/DTensor operations during training.

    Keep Verl's FSDP2 flow, but independently wrap only the configured
    transformer layers. The root FSDP group owns embeddings and all other
    non-transformer parameters, so registered policy forward methods unshard
    them together.
    """

    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    if isinstance(transformer_layer_cls_to_wrap, str):
        transformer_layer_cls_to_wrap = [transformer_layer_cls_to_wrap]

    assert len(transformer_layer_cls_to_wrap) > 0 and transformer_layer_cls_to_wrap[0] is not None

    modules = _select_fsdp2_wrap_targets(model, transformer_layer_cls_to_wrap)
    for module in modules:
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)
