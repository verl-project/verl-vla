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

"""LeRobot dataset metadata and Parquet helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def list_lerobot_data_files(dataset_root: str | Path) -> list[Path]:
    data_files = sorted((Path(dataset_root) / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No LeRobot data files found under {Path(dataset_root) / 'data'}.")
    return data_files


def load_lerobot_feature_names(dataset_root: str | Path) -> set[str]:
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        return set()
    with open(info_path) as f:
        info = json.load(f)
    return set(info.get("features", {}).keys())


def count_lerobot_episodes(dataset_root: str | Path) -> int:
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        return 0
    with open(info_path) as f:
        info = json.load(f)
    return int(info.get("total_episodes", 0))


def truncate_lerobot_episodes(dataset_root: str | Path, max_episodes: int) -> None:
    root = Path(dataset_root)
    if max_episodes <= 0:
        raise ValueError(f"max_episodes must be positive, got {max_episodes}.")

    total_episodes = count_lerobot_episodes(root)
    if total_episodes <= max_episodes:
        return

    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    tmp_root = root.parent / f".{root.name}.truncate_tmp"
    shutil.rmtree(tmp_root, ignore_errors=True)
    dataset = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
    delete_episodes(
        dataset,
        episode_indices=list(range(max_episodes, total_episodes)),
        output_dir=tmp_root,
        repo_id=f"local/{tmp_root.name}",
    )
    shutil.rmtree(root)
    shutil.move(str(tmp_root), str(root))


def collect_lerobot_columns(data_files: list[str | Path]) -> set[str]:
    columns: set[str] = set()
    for parquet_path in data_files:
        columns.update(pq.ParquetFile(parquet_path).schema_arrow.names)
    return columns


def iter_lerobot_frame_records(
    dataset_root: str | Path,
    *,
    columns: list[str],
    optional_columns: dict[str, Any] | None = None,
):
    optional_columns = optional_columns or {}
    for data_file in list_lerobot_data_files(dataset_root):
        names = set(pq.ParquetFile(data_file).schema_arrow.names)
        missing_columns = [column for column in columns if column not in names]
        if missing_columns:
            raise KeyError(f"Missing LeRobot columns in {data_file}: {missing_columns}")

        read_columns = [*columns, *[column for column in optional_columns if column in names and column not in columns]]
        table = pq.read_table(data_file, columns=read_columns)
        column_values = {column: table[column].to_numpy() for column in columns}
        row_count = len(next(iter(column_values.values()), []))
        for column, fallback in optional_columns.items():
            if column in names:
                column_values[column] = table[column].to_numpy()
            else:
                column_values[column] = np.full(row_count, fallback)

        for row_idx in range(row_count):
            yield {column: values[row_idx] for column, values in column_values.items()}


def write_lerobot_frame_columns(
    dataset_root: str | Path,
    *,
    columns_by_index: dict[str, dict[int, Any]],
    dtypes: dict[str, np.dtype],
) -> None:
    for data_file in list_lerobot_data_files(dataset_root):
        table = pq.read_table(data_file, columns=["index"])
        indices = table["index"].to_numpy().astype(np.int64, copy=False)
        columns = {
            field: np.asarray([values[int(index)] for index in indices], dtype=dtypes[field])
            for field, values in columns_by_index.items()
        }
        write_parquet_columns(parquet_path=data_file, columns=columns)


def write_parquet_columns(parquet_path: Path, columns: dict[str, np.ndarray]) -> None:
    table = pq.read_table(parquet_path)
    for field, values in columns.items():
        table = set_parquet_column(table=table, field=field, values=values)
    pq.write_table(table, parquet_path, compression="snappy")


def set_parquet_column(table: pa.Table, field: str, values: np.ndarray) -> pa.Table:
    array = pa.array(values)
    field_index = table.schema.get_field_index(field)
    if field_index >= 0:
        return table.set_column(field_index, field, array)
    return table.append_column(field, array)


def update_lerobot_feature_metadata(dataset_root: str | Path, feature_infos: dict[str, dict[str, Any]]) -> None:
    try:
        from lerobot.datasets.utils import load_info, write_info
    except ImportError:
        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"LeRobot info.json not found: {info_path}") from None
        with open(info_path) as f:
            info = json.load(f)
        features = info.setdefault("features", {})
        for feature_name, feature_info in feature_infos.items():
            features[feature_name] = feature_info
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        return

    info = load_info(Path(dataset_root))
    for feature_name, feature_info in feature_infos.items():
        info["features"][feature_name] = {
            "dtype": feature_info["dtype"],
            "shape": tuple(feature_info["shape"]),
            "names": feature_info.get("names"),
        }
    write_info(info, Path(dataset_root))
