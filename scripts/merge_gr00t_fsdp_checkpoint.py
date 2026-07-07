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

"""Merge a verl FSDP2 GR00T SFT checkpoint into a loadable Hugging Face dir.

The SFT trainer saves an actor checkpoint as per-rank FSDP2 shards
(``model_world_size_N_rank_R.pt``) plus a ``huggingface/`` dir holding the
``gr00t_torch`` ``config.json`` and the Qwen3-VL processor. FSDP2 shards every
parameter along a flattened dim-0, so this script reconstructs each full tensor
by concatenating all ranks' local shards, trimming to the reference numel, and
reshaping to the reference shape taken from the base checkpoint's safetensors.

The wrapper prefix ``model.`` (``Gr00tForActionPrediction.model``) is stripped so
the result is keyed at the ``Gr00tN1d7Model`` level (``backbone.*`` /
``action_head.*``), which is what ``Gr00tForConditionalGeneration.from_pretrained``
expects. Training-only buffers such as ``flow_sde_step`` are dropped.

Example:
    python scripts/merge_gr00t_fsdp_checkpoint.py \
        --actor-dir  /data/output/gr00t_libero_sft/global_step_25/actor \
        --base-dir   /data/models/gr00t_n1d7_libero_base \
        --output-dir /data/models/gr00t_libero_sft_step25_hf
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _reference_shapes(base_dir: str) -> dict[str, list[int]]:
    """Full (unsharded) tensor shapes from the base checkpoint's safetensors."""
    shapes: dict[str, list[int]] = {}
    shards = glob.glob(os.path.join(base_dir, "*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No *.safetensors found in base dir {base_dir}")
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                shapes[key] = list(f.get_slice(key).get_shape())
    return shapes


def _local(t: torch.Tensor) -> torch.Tensor:
    """Return a plain CPU tensor from a (possibly DTensor) FSDP2 shard."""
    if hasattr(t, "to_local"):
        t = t.to_local()
    return t.detach().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor-dir", required=True, help="global_step_N/actor dir with model_world_size_*_rank_*.pt")
    ap.add_argument("--base-dir", required=True, help="Base gr00t_torch checkpoint (for reference tensor shapes)")
    ap.add_argument("--output-dir", required=True, help="Output HF checkpoint dir")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ref_shapes = _reference_shapes(args.base_dir)

    rank_files = sorted(glob.glob(os.path.join(args.actor_dir, "model_world_size_*_rank_*.pt")))
    if not rank_files:
        raise FileNotFoundError(f"No FSDP rank shards in {args.actor_dir}")
    print(f"Loading {len(rank_files)} FSDP rank shards")
    ranks = [torch.load(f, map_location="cpu", weights_only=False) for f in rank_files]

    out_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    merged: dict[str, torch.Tensor] = {}
    skipped = []
    for key in ranks[0]:
        if key == "flow_sde_step" or not key.startswith("model."):
            skipped.append(key)
            continue
        base_key = key[len("model.") :]
        shape = ref_shapes.get(base_key)
        if shape is None:
            skipped.append(key)
            continue
        numel = 1
        for s in shape:
            numel *= s
        flat = torch.cat([_local(r[key]).flatten() for r in ranks])[:numel]
        tensor = flat.reshape(shape)
        if tensor.is_floating_point():
            tensor = tensor.to(out_dtype)
        merged[base_key] = tensor.contiguous().clone()

    print(f"Merged {len(merged)} tensors (skipped {len(skipped)}: e.g. {skipped[:3]})")

    hf_dir = os.path.join(args.actor_dir, "huggingface")
    if os.path.isdir(hf_dir):
        for name in os.listdir(hf_dir):
            shutil.copy2(os.path.join(hf_dir, name), os.path.join(args.output_dir, name))
        print("Copied config + processor from", hf_dir)

    save_file(merged, os.path.join(args.output_dir, "model.safetensors"), metadata={"format": "pt"})
    stale_index = os.path.join(args.output_dir, "model.safetensors.index.json")
    if os.path.exists(stale_index):
        os.remove(stale_index)
    print("Wrote", os.path.join(args.output_dir, "model.safetensors"))
    print("Load with Gr00tForConditionalGeneration.from_pretrained(%s)" % args.output_dir)


if __name__ == "__main__":
    main()
