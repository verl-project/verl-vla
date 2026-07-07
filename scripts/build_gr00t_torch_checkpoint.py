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

"""Assemble a `gr00t_torch`-format checkpoint directory that
``Gr00tForConditionalGeneration.from_pretrained`` can load.

The verl-vla ``Gr00tN1d7Model`` is a verbatim copy of upstream ``Gr00tN1d7``, so
upstream ``nvidia/GR00T-N1.7-*`` weights load with an identity key mapping. This
script therefore does NOT rename weights; it just:

  1. (optionally) downloads the upstream GR00T checkpoint + the gated
     Cosmos-Reason2-2B backbone/processor from HuggingFace,
  2. copies the safetensors shard(s) + index verbatim,
  3. copies the Qwen3-VL processor files (so ``vlm_processor_path=None`` resolves
     to this dir),
  4. writes a ``config.json`` in the ``Gr00tTorchConfig`` schema with the
     required non-empty ``state_norm_stats`` / ``action_norm_stats`` embedded.

Norm stats come from ``scripts/compute_norm_stats.py`` run on your Libero LeRobot
dataset (produces ``{"state": {...q01,q99...}, "action": {...q01,q99...}}``).

Example (inside the container, HF_TOKEN exported for gated Cosmos):
    python scripts/build_gr00t_torch_checkpoint.py \
        --gr00t-repo nvidia/GR00T-N1.7-LIBERO \
        --cosmos-repo nvidia/Cosmos-Reason2-2B \
        --norm-stats /data/datasets/libero_lerobot/norm_stats.json \
        --output /data/models/gr00t_n1d7_libero_base
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_gr00t_ckpt")

# Processor files to bundle from Cosmos-Reason2-2B (or the upstream GR00T ckpt).
PROCESSOR_FILES = [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "added_tokens.json",
]

WEIGHT_GLOBS = ["*.safetensors", "*.safetensors.index.json"]

# ONLY the verl-vla integration fields. Architecture-shaping fields
# (select_layer, diffusion_model_cfg, vl_self_attention_cfg, max_*_dim,
# hidden_size, action_horizon, max_num_embodiments, use_alternate_vl_dit,
# attend_text_every_n_blocks, ...) MUST come verbatim from the upstream config
# so tensor shapes match the checkpoint — they are intentionally NOT listed here.
GR00T_TORCH_CONFIG_DEFAULTS = {
    "model_type": "gr00t_torch",
    "architectures": ["Gr00tForConditionalGeneration"],
    "model_name": "nvidia/Cosmos-Reason2-2B",
    # verl-vla integration
    "policy_type": "libero",
    "embodiment_id": 2,  # libero_sim projector slot
    "action_chunk_size": 10,
    "sft_action_horizon": 16,
    "vlm_processor_path": None,
    # pure SFT: no critic / stochastic sampler baked into the base ckpt config
    "sac_enable": False,
    "flow_sde_enable": False,
}


def maybe_download(repo: str, local_dir: str | None) -> str:
    """Return a local dir for `repo`, downloading via HF if it isn't a path."""
    if repo and os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download  # lazy import

    target = local_dir or os.path.join(os.getcwd(), repo.replace("/", "__"))
    logger.info("Downloading %s -> %s", repo, target)
    return snapshot_download(repo_id=repo, local_dir=target)


def copy_matching(src_dir: str, dst_dir: str, names: list[str] | None = None,
                  globs: list[str] | None = None) -> list[str]:
    copied = []
    if names:
        for name in names:
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst_dir, name))
                copied.append(name)
    if globs:
        for pattern in globs:
            for src in glob.glob(os.path.join(src_dir, pattern)):
                base = os.path.basename(src)
                shutil.copy2(src, os.path.join(dst_dir, base))
                copied.append(base)
    return copied


def load_norm_stats(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        stats = json.load(f)
    # compute_norm_stats.py emits {"state": {...}, "action": {...}, "meta": {...}}
    state = stats["state"] if "state" in stats else stats.get("state_norm_stats")
    action = stats["action"] if "action" in stats else stats.get("action_norm_stats")
    for name, s in (("state", state), ("action", action)):
        if not s or "q01" not in s or "q99" not in s:
            raise ValueError(f"{name} norm stats missing q01/q99: {s!r}")
    logger.info("state q01/q99 dim=%d, action q01/q99 dim=%d", len(state["q01"]), len(action["q01"]))
    return state, action


def build_config(src_config_path: str | None, state_stats: dict, action_stats: dict) -> dict:
    # Start from the FULL upstream config so every architecture-shaping field
    # (select_layer, diffusion_model_cfg.num_layers, use_vl_self_attention,
    # vl_self_attention_cfg, ...) matches the checkpoint's tensor shapes exactly.
    # The public N1.7-3B uses select_layer=16 / 32 DiT layers / vl_self_attn=4,
    # which differ from Gr00tTorchConfig's defaults — copying verbatim avoids a
    # shape mismatch at load.
    cfg = {}
    if src_config_path and os.path.isfile(src_config_path):
        with open(src_config_path) as f:
            cfg = dict(json.load(f))
    # Overlay the verl-vla gr00t_torch integration fields (defaults + identity).
    cfg.update(GR00T_TORCH_CONFIG_DEFAULTS)
    cfg["state_norm_stats"] = state_stats
    cfg["action_norm_stats"] = action_stats
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gr00t-repo", required=True, help="HF repo id or local dir of upstream GR00T N1.7 ckpt")
    ap.add_argument("--gr00t-local-dir", default=None, help="Where to download the GR00T ckpt")
    ap.add_argument("--gr00t-subdir", default="", help="Sub-dir inside the ckpt (e.g. a libero suite)")
    ap.add_argument("--cosmos-repo", default="nvidia/Cosmos-Reason2-2B", help="Backbone/processor repo (gated)")
    ap.add_argument("--cosmos-local-dir", default=None)
    ap.add_argument("--norm-stats", required=True, help="JSON from scripts/compute_norm_stats.py")
    ap.add_argument("--output", required=True, help="Output gr00t_torch checkpoint dir")
    ap.add_argument("--drop-backbone", action="store_true",
                    help="Drop backbone.* shards (backbone reloads from Cosmos at build).")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    gr00t_dir = maybe_download(args.gr00t_repo, args.gr00t_local_dir)
    if args.gr00t_subdir:
        gr00t_dir = os.path.join(gr00t_dir, args.gr00t_subdir)
    cosmos_dir = maybe_download(args.cosmos_repo, args.cosmos_local_dir)

    logger.info("Copying weight shards from %s", gr00t_dir)
    weights = copy_matching(gr00t_dir, str(out), globs=WEIGHT_GLOBS)
    if not weights:
        raise SystemExit(f"No safetensors found in {gr00t_dir}")
    logger.info("Copied weights: %s", weights)

    logger.info("Bundling Qwen3-VL processor files")
    proc = copy_matching(cosmos_dir, str(out), names=PROCESSOR_FILES)
    if not proc:  # fall back to processor bundled in the GR00T ckpt
        proc = copy_matching(gr00t_dir, str(out), names=PROCESSOR_FILES)
    logger.info("Copied processor files: %s", proc)

    state_stats, action_stats = load_norm_stats(args.norm_stats)
    cfg = build_config(os.path.join(gr00t_dir, "config.json"), state_stats, action_stats)
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Wrote %s", out / "config.json")

    logger.info("DONE. Load with Gr00tForConditionalGeneration.from_pretrained(%s)", out)
    logger.info("If --drop-backbone was NOT set, Cosmos-Reason2-2B (gated) is still "
                "loaded at build time; ensure HF access.")


if __name__ == "__main__":
    main()
