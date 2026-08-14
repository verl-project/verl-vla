# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native LeRobot processor construction for ACT policies."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.utils import load_stats
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def _validate_normalization_stats(config, stats: dict) -> None:
    required_fields = {
        NormalizationMode.MEAN_STD: ("mean", "std"),
        NormalizationMode.MIN_MAX: ("min", "max"),
        NormalizationMode.QUANTILES: ("q01", "q99"),
        NormalizationMode.QUANTILE10: ("q10", "q90"),
    }
    for name, feature in {**config.input_features, **config.output_features}.items():
        if name not in config.image_features and name in stats and "mean" in stats[name]:
            actual_dim = int(np.asarray(stats[name]["mean"]).size)
            expected_dim = int(np.prod(feature.shape))
            if actual_dim != expected_dim:
                raise ValueError(
                    f"ACT normalization stats for {name!r} have dimension {actual_dim}, "
                    f"but the checkpoint expects {expected_dim}"
                )
        mode = config.normalization_mapping.get(feature.type, NormalizationMode.IDENTITY)
        if mode == NormalizationMode.IDENTITY:
            continue
        missing = [field for field in required_fields[mode] if field not in stats.get(name, {})]
        if missing:
            raise ValueError(f"ACT normalization stats for {name!r} are missing {missing}")


def load_act_processors(
    config,
    model_path: str | Path,
    *,
    dataset_root: str | Path | None,
):
    """Load checkpoint processors or initialize native processors from dataset statistics.

    LeRobot stores normalization outside the policy config. A trained checkpoint must
    therefore reload its saved processor artifacts, while a config-only ACT initializer
    needs the training dataset statistics once to construct those artifacts. Visual
    features intentionally use LeRobot's standard ImageNet statistics, matching its
    native ACT training factory; state and action features retain dataset statistics.
    """

    model_path = Path(model_path)
    preprocessor_path = model_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    postprocessor_path = model_path / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
    processor_files = (preprocessor_path.is_file(), postprocessor_path.is_file())
    if any(processor_files) and not all(processor_files):
        raise FileNotFoundError(
            f"Incomplete native ACT processor artifacts in {model_path}: both "
            f"{preprocessor_path.name} and {postprocessor_path.name} are required"
        )
    if all(processor_files):
        return make_pre_post_processors(config, pretrained_path=str(model_path))

    if dataset_root is None:
        raise FileNotFoundError(
            f"Native ACT processor artifacts are missing from {model_path}. "
            "Set model.adapter.processor_dataset_root when initializing ACT from config."
        )

    dataset_root = Path(dataset_root)
    stats = load_stats(dataset_root)
    if stats is None:
        raise FileNotFoundError(f"LeRobot dataset statistics are missing from {dataset_root / 'meta/stats.json'}")

    stats = deepcopy(stats)
    for image_key in config.image_features:
        stats[image_key] = deepcopy(IMAGENET_STATS)
    _validate_normalization_stats(config, stats)
    return make_pre_post_processors(config, dataset_stats=stats)


__all__ = ["load_act_processors"]
