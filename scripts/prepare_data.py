"""Prepare canonical processed data and split artifacts for Paper 2.

This script converts a verified raw MovieLens dataset into versioned,
reproducible artifacts consumed by downstream experiments.

Pipeline
--------
1. Load the experiment configuration.
2. Load and integrity-check the raw dataset.
3. Apply configured user/item interaction filters.
4. Create deterministic contiguous user/item indices.
5. Create the canonical train/validation/test split.
6. Validate the split independently.
7. Save processed data, mappings, and split files.
8. Record SHA-256 hashes and metadata in versioned manifests.

The raw dataset is never modified.

Example
-------
python scripts/prepare_data.py --config config/dev.yaml --version v1
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data import load_dataset, load_yaml
from src.split import split_from_config
from src.validation import validate_split


CORE_COLUMNS = [
    "userId",
    "movieId",
    "rating",
    "timestamp",
]

PROCESSED_COLUMNS = [
    "userId",
    "movieId",
    "rating",
    "timestamp",
    "user_idx",
    "item_idx",
]


def _resolve_project_path(
    path: str | Path,
) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _sha256_file(
    path: Path,
) -> str:
    """Compute SHA-256 for one file."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def _semantic_yaml_hash(
    data: dict[str, Any],
) -> str:
    """Hash YAML content independent of file line endings."""
    canonical = yaml.safe_dump(
        data,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(
        canonical
    ).hexdigest()


def _write_yaml(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Write deterministic UTF-8 YAML."""
    content = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
    )

    path.write_bytes(
        content.encode("utf-8")
    )


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write deterministic CSV without an index column."""
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.10g",
    )


def _apply_interaction_filters(
    ratings: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    """Apply iterative user/item minimum-frequency filtering."""
    if min_user_interactions < 1:
        raise ValueError(
            "min_user_interactions must be at least 1"
        )

    if min_item_interactions < 1:
        raise ValueError(
            "min_item_interactions must be at least 1"
        )

    filtered = ratings[
        CORE_COLUMNS
    ].copy()

    while True:
        rows_before = len(
            filtered
        )

        user_counts = (
            filtered["userId"]
            .value_counts()
        )

        valid_users = set(
            user_counts.loc[
                user_counts
                >= min_user_interactions
            ].index
        )

        filtered = filtered.loc[
            filtered["userId"].isin(
                valid_users
            )
        ].copy()

        item_counts = (
            filtered["movieId"]
            .value_counts()
        )

        valid_items = set(
            item_counts.loc[
                item_counts
                >= min_item_interactions
            ].index
        )

        filtered = filtered.loc[
            filtered["movieId"].isin(
                valid_items
            )
        ].copy()

        if len(filtered) == 0:
            raise ValueError(
                "Interaction filtering removed "
                "the entire dataset"
            )

        if len(filtered) == rows_before:
            break

    return (
        filtered.sort_values(
            by=[
                "userId",
                "movieId",
                "timestamp",
                "rating",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _add_canonical_indices(
    ratings: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create deterministic contiguous zero-based user/item indices."""
    user_ids = sorted(
        ratings["userId"]
        .astype(int)
        .unique()
        .tolist()
    )

    item_ids = sorted(
        ratings["movieId"]
        .astype(int)
        .unique()
        .tolist()
    )

    user_mapping = pd.DataFrame(
        {
            "userId": user_ids,
            "user_idx": range(
                len(user_ids)
            ),
        }
    )

    item_mapping = pd.DataFrame(
        {
            "movieId": item_ids,
            "item_idx": range(
                len(item_ids)
            ),
        }
    )

    user_map = dict(
        zip(
            user_mapping["userId"],
            user_mapping["user_idx"],
        )
    )

    item_map = dict(
        zip(
            item_mapping["movieId"],
            item_mapping["item_idx"],
        )
    )

    indexed = ratings.copy()

    indexed["user_idx"] = (
        indexed["userId"]
        .map(user_map)
        .astype("int64")
    )

    indexed["item_idx"] = (
        indexed["movieId"]
        .map(item_map)
        .astype("int64")
    )

    indexed = indexed[
        PROCESSED_COLUMNS
    ]

    return (
        indexed,
        user_mapping,
        item_mapping,
    )


def _attach_indices(
    frame: pd.DataFrame,
    user_mapping: pd.DataFrame,
    item_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Attach canonical user/item indices to a split frame."""
    result = frame.merge(
        user_mapping,
        on="userId",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        item_mapping,
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    if result[
        ["user_idx", "item_idx"]
    ].isna().any().any():
        raise RuntimeError(
            "Failed to attach canonical indices"
        )

    result["user_idx"] = (
        result["user_idx"]
        .astype("int64")
    )

    result["item_idx"] = (
        result["item_idx"]
        .astype("int64")
    )

    return (
        result[
            PROCESSED_COLUMNS
        ]
        .sort_values(
            by=[
                "userId",
                "movieId",
                "timestamp",
                "rating",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _combined_split_hash(
    train_hash: str,
    validation_hash: str,
    test_hash: str,
) -> str:
    """Create one stable fingerprint for the three canonical split files."""
    payload = (
        f"train:{train_hash}\n"
        f"validation:{validation_hash}\n"
        f"test:{test_hash}\n"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def _prepare_output_directory(
    path: Path,
    force: bool,
) -> None:
    """Create a clean versioned output directory."""
    if path.exists():
        contains_files = any(
            path.iterdir()
        )

        if contains_files and not force:
            raise FileExistsError(
                f"Output directory already contains files: {path}\n"
                "Use --force only if you intentionally want "
                "to regenerate this version."
            )

        if force:
            shutil.rmtree(
                path
            )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def prepare_data(
    config_path: str | Path,
    version: str,
    force: bool = False,
) -> None:
    """Run the complete deterministic data-preparation pipeline."""
    if not isinstance(
        version,
        str,
    ):
        raise TypeError(
            "version must be a string"
        )

    if not version.strip():
        raise ValueError(
            "version must not be empty"
        )

    config = load_yaml(
        config_path
    )

    dataset = load_dataset(
        config_path,
        verify_integrity=True,
    )

    dataset_id = dataset.dataset_id

    preprocessing = config[
        "preprocessing"
    ]

    if not bool(
        preprocessing[
            "reindex_users"
        ]
    ):
        raise ValueError(
            "Canonical pipeline requires "
            "reindex_users=true"
        )

    if not bool(
        preprocessing[
            "reindex_items"
        ]
    ):
        raise ValueError(
            "Canonical pipeline requires "
            "reindex_items=true"
        )

    raw_rows = len(
        dataset.ratings
    )

    filtered = _apply_interaction_filters(
        ratings=dataset.ratings,
        min_user_interactions=int(
            preprocessing[
                "min_user_interactions"
            ]
        ),
        min_item_interactions=int(
            preprocessing[
                "min_item_interactions"
            ]
        ),
    )

    (
        processed,
        user_mapping,
        item_mapping,
    ) = _add_canonical_indices(
        filtered
    )

    split = split_from_config(
        processed,
        config,
    )

    validation_report = validate_split(
        original=processed,
        split=split,
        config=config,
    )

    train = _attach_indices(
        split.train,
        user_mapping,
        item_mapping,
    )

    validation = _attach_indices(
        split.validation,
        user_mapping,
        item_mapping,
    )

    test = _attach_indices(
        split.test,
        user_mapping,
        item_mapping,
    )

    processed_root = (
        _resolve_project_path(
            config["dataset"][
                "processed_root"
            ]
        )
        / dataset_id
        / version
    )

    split_root = (
        _resolve_project_path(
            config["dataset"][
                "split_root"
            ]
        )
        / dataset_id
        / version
    )

    _prepare_output_directory(
        processed_root,
        force=force,
    )

    _prepare_output_directory(
        split_root,
        force=force,
    )

    processed_path = (
        processed_root
        / "ratings.csv"
    )

    user_mapping_path = (
        processed_root
        / "user_mapping.csv"
    )

    item_mapping_path = (
        processed_root
        / "item_mapping.csv"
    )

    train_path = (
        split_root
        / "train.csv"
    )

    validation_path = (
        split_root
        / "validation.csv"
    )

    test_path = (
        split_root
        / "test.csv"
    )

    _write_csv(
        processed,
        processed_path,
    )

    _write_csv(
        user_mapping,
        user_mapping_path,
    )

    _write_csv(
        item_mapping,
        item_mapping_path,
    )

    _write_csv(
        train,
        train_path,
    )

    _write_csv(
        validation,
        validation_path,
    )

    _write_csv(
        test,
        test_path,
    )

    processed_hash = (
        _sha256_file(
            processed_path
        )
    )

    user_mapping_hash = (
        _sha256_file(
            user_mapping_path
        )
    )

    item_mapping_hash = (
        _sha256_file(
            item_mapping_path
        )
    )

    train_hash = _sha256_file(
        train_path
    )

    validation_hash = (
        _sha256_file(
            validation_path
        )
    )

    test_hash = _sha256_file(
        test_path
    )

    split_hash = (
        _combined_split_hash(
            train_hash=train_hash,
            validation_hash=validation_hash,
            test_hash=test_hash,
        )
    )

    config_hash = (
        _semantic_yaml_hash(
            config
        )
    )

    source_manifest = (
        dataset.manifest
    )

    source_manifest_hash = (
        _semantic_yaml_hash(
            source_manifest
        )
    )

    preparation_manifest = {
        "artifact": {
            "type": "processed_dataset",
            "version": version,
            "dataset_id": dataset_id,
        },
        "source": {
            "raw_rows": raw_rows,
            "raw_users": int(
                dataset.ratings[
                    "userId"
                ].nunique()
            ),
            "raw_items": int(
                dataset.ratings[
                    "movieId"
                ].nunique()
            ),
            "raw_archive_sha256": (
                source_manifest[
                    "integrity"
                ]["sha256"]
            ),
            "dataset_manifest_sha256": (
                source_manifest_hash
            ),
            "config_sha256": (
                config_hash
            ),
        },
        "preprocessing": {
            "min_user_interactions": int(
                preprocessing[
                    "min_user_interactions"
                ]
            ),
            "min_item_interactions": int(
                preprocessing[
                    "min_item_interactions"
                ]
            ),
            "reindex_users": True,
            "reindex_items": True,
            "rows_after_filtering": int(
                len(processed)
            ),
            "users_after_filtering": int(
                processed[
                    "userId"
                ].nunique()
            ),
            "items_after_filtering": int(
                processed[
                    "movieId"
                ].nunique()
            ),
        },
        "files": {
            "ratings": {
                "path": str(
                    processed_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": (
                    processed_hash
                ),
            },
            "user_mapping": {
                "path": str(
                    user_mapping_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": (
                    user_mapping_hash
                ),
            },
            "item_mapping": {
                "path": str(
                    item_mapping_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": (
                    item_mapping_hash
                ),
            },
        },
    }

    preparation_manifest_path = (
        processed_root
        / "preparation_manifest.yaml"
    )

    _write_yaml(
        preparation_manifest_path,
        preparation_manifest,
    )

    split_manifest = {
        "artifact": {
            "type": "canonical_split",
            "version": version,
            "dataset_id": dataset_id,
        },
        "source": {
            "processed_dataset_sha256": (
                processed_hash
            ),
            "config_sha256": (
                config_hash
            ),
        },
        "protocol": {
            "strategy": config[
                "split"
            ]["strategy"],
            "train_fraction": float(
                config["split"][
                    "train_fraction"
                ]
            ),
            "validation_fraction": float(
                config["split"][
                    "validation_fraction"
                ]
            ),
            "test_fraction": float(
                config["split"][
                    "test_fraction"
                ]
            ),
            "seed": int(
                split.seed
            ),
            "constrained_attempt": int(
                split.attempt
            ),
            "enforce_train_item_coverage": bool(
                split.enforce_train_item_coverage
            ),
        },
        "statistics": {
            "original_rows": int(
                validation_report.original_rows
            ),
            "train_rows": int(
                validation_report.train_rows
            ),
            "validation_rows": int(
                validation_report.validation_rows
            ),
            "test_rows": int(
                validation_report.test_rows
            ),
            "users": int(
                validation_report.users
            ),
            "items": int(
                validation_report.original_items
            ),
            "train_items": int(
                validation_report.train_items
            ),
            "cold_validation_items": int(
                validation_report.cold_validation_items
            ),
            "cold_test_items": int(
                validation_report.cold_test_items
            ),
        },
        "files": {
            "train": {
                "path": str(
                    train_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": train_hash,
            },
            "validation": {
                "path": str(
                    validation_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": validation_hash,
            },
            "test": {
                "path": str(
                    test_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "sha256": test_hash,
            },
        },
        "split_sha256": split_hash,
    }

    split_manifest_path = (
        split_root
        / "split_manifest.yaml"
    )

    _write_yaml(
        split_manifest_path,
        split_manifest,
    )

    print(
        "=== DATA PREPARATION COMPLETE ==="
    )

    print(
        "Dataset:",
        dataset_id,
    )

    print(
        "Version:",
        version,
    )

    print(
        "Raw rows:",
        raw_rows,
    )

    print(
        "Rows after filtering:",
        len(processed),
    )

    print(
        "Users:",
        processed[
            "userId"
        ].nunique(),
    )

    print(
        "Items:",
        processed[
            "movieId"
        ].nunique(),
    )

    print()

    print(
        "Train rows:",
        len(train),
    )

    print(
        "Validation rows:",
        len(validation),
    )

    print(
        "Test rows:",
        len(test),
    )

    print()

    print(
        "Cold validation items:",
        validation_report.cold_validation_items,
    )

    print(
        "Cold test items:",
        validation_report.cold_test_items,
    )

    print()

    print(
        "Processed dataset SHA256:",
        processed_hash,
    )

    print(
        "Canonical split SHA256:",
        split_hash,
    )

    print()

    print(
        "Processed output:",
        processed_root.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Split output:",
        split_root.relative_to(
            PROJECT_ROOT
        ),
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic processed data "
            "and canonical split artifacts."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Experiment configuration path, "
            "for example config/dev.yaml"
        ),
    )

    parser.add_argument(
        "--version",
        default="v1",
        help=(
            "Artifact version name. "
            "Default: v1"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete and regenerate existing "
            "version directories."
        ),
    )

    args = parser.parse_args()

    prepare_data(
        config_path=args.config,
        version=args.version,
        force=args.force,
    )


if __name__ == "__main__":
    main()