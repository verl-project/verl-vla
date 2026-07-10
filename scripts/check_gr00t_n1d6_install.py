# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Check that the opt-in GR00T package came from verl-vla's pinned commit."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from verl_vla.models.gr00t_n1d6 import GR00T_N1D6_COMMIT
from verl_vla.models.register_vla_models import register_gr00t_n1d6_model

_EAGLE_ASSETS = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)


def direct_url() -> dict:
    try:
        dist = distribution("gr00t")
    except PackageNotFoundError as exc:
        register_gr00t_n1d6_model(required=True)
        raise AssertionError("unreachable") from exc
    direct_url_file = next((file for file in (dist.files or []) if file.name == "direct_url.json"), None)
    if direct_url_file is None:
        raise RuntimeError(
            "The installed gr00t distribution has no direct_url.json; install it from the pinned Git URL."
        )
    with dist.locate_file(direct_url_file).open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    metadata = direct_url()
    commit = metadata.get("vcs_info", {}).get("commit_id")
    if commit != GR00T_N1D6_COMMIT:
        raise RuntimeError(
            f"Unsupported GR00T source commit {commit!r}; expected {GR00T_N1D6_COMMIT}. "
            "Reinstall with the command documented in examples/gr00t_sft/README.md."
        )
    register_gr00t_n1d6_model(required=True)

    # The upstream VCS wheel does not currently include Eagle's non-Python
    # package data.  Fail during image construction instead of allowing
    # Transformers to silently infer OPTConfig from the /opt/... path.
    import gr00t
    from transformers import AutoConfig

    eagle_dir = Path(gr00t.__file__).parent / "model" / "modules" / "nvidia" / "Eagle-Block2A-2B-v2"
    missing_assets = [name for name in _EAGLE_ASSETS if not (eagle_dir / name).is_file()]
    if missing_assets:
        raise RuntimeError(
            f"The installed GR00T package is missing Eagle assets: {missing_assets}. "
            "Use docker/Dockerfile.gr00t, which restores them from the pinned source commit."
        )

    eagle_config = AutoConfig.from_pretrained(eagle_dir, trust_remote_code=True)
    if eagle_config.model_type != "eagle_3_vl":
        raise RuntimeError(f"Expected Eagle config model_type 'eagle_3_vl', got {eagle_config.model_type!r}.")

    print(f"GR00T N1.6 source package and Eagle assets verified at {commit}")


if __name__ == "__main__":
    main()
