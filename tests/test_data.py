"""Tests for src.data.

These tests verify the shared dataset-loading layer without depending on the
real MovieLens downloads being present.

A temporary miniature MovieLens Latest Small-style dataset is constructed
inside pytest's temporary directory. This makes the tests portable and keeps
the real raw datasets untouched.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data import load_dataset, load_yaml


def _write_yaml(
    path: Path,
    content: dict,
) -> None:
    """Write a YAML mapping to disk."""
    path.write_text(
        yaml.safe_dump(
            content,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _build_temporary_dataset(
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Create a tiny self-contained dataset/config pair."""
    raw_root = tmp_path / "raw"
    dataset_root = raw_root / "ml-latest-small"

    raw_root.mkdir()
    dataset_root.mkdir()

    ratings = pd.DataFrame(
        [
            [1, 1, 5.0, 100],
            [1, 2, 4.0, 101],
            [2, 1, 3.0, 102],
            [2, 3, 4.5, 103],
        ],
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
    )

    movies = pd.DataFrame(
        [
            [1, "Movie A", "Drama"],
            [2, "Movie B", "Comedy"],
            [3, "Movie C", "Action"],
        ],
        columns=[
            "movieId",
            "title",
            "genres",
        ],
    )

    ratings.to_csv(
        dataset_root / "ratings.csv",
        index=False,
    )

    movies.to_csv(
        dataset_root / "movies.csv",
        index=False,
    )

    # The loader verifies archive identity, but does not need to extract it.
    # A deterministic miniature placeholder archive is sufficient here.
    archive_path = raw_root / "ml-latest-small.zip"
    archive_bytes = b"paper2 synthetic archive fixture"

    archive_path.write_bytes(
        archive_bytes
    )

    md5 = hashlib.md5(
        archive_bytes
    ).hexdigest()

    sha256 = hashlib.sha256(
        archive_bytes
    ).hexdigest()

    manifest = {
        "dataset": {
            "id": "ml-latest-small",
            "name": "Synthetic MovieLens Fixture",
            "provider": "pytest",
            "role": "test",
            "publication_primary": False,
        },
        "source": {
            "archive_filename": "ml-latest-small.zip",
            "extracted_directory": "ml-latest-small",
        },
        "integrity": {
            "archive_size_bytes": len(
                archive_bytes
            ),
            "official_md5": md5,
            "sha256": sha256,
            "checksum_verified": True,
        },
        "format": {
            "ratings_file": "ratings.csv",
            "movies_file": "movies.csv",
            "delimiter": ",",
            "encoding": "utf-8",
        },
        "ratings_schema": {
            "columns": [
                "userId",
                "movieId",
                "rating",
                "timestamp",
            ],
        },
        "raw_statistics": {
            "ratings": 4,
            "users": 2,
            "movies_metadata": 3,
            "rated_movies": 3,
            "rating_min": 3.0,
            "rating_max": 5.0,
            "missing_values_in_ratings": 0,
        },
        "usage": {
            "intended_use": [
                "testing",
            ],
            "use_for_primary_publication_results": False,
        },
        "preprocessing": {
            "status": "not_started",
        },
        "split": {
            "status": "not_created",
        },
        "notes": [
            "Synthetic pytest fixture."
        ],
    }

    manifest_path = (
        tmp_path
        / "manifest.yaml"
    )

    _write_yaml(
        manifest_path,
        manifest,
    )

    config = {
        "dataset": {
            "id": "ml-latest-small",
            "manifest": str(
                manifest_path
            ),
            "raw_root": str(
                raw_root
            ),
        },
    }

    config_path = (
        tmp_path
        / "config.yaml"
    )

    _write_yaml(
        config_path,
        config,
    )

    return (
        config_path,
        manifest_path,
    )


def test_load_yaml_reads_mapping(
    tmp_path: Path,
) -> None:
    """A normal YAML mapping should load successfully."""
    path = tmp_path / "example.yaml"

    _write_yaml(
        path,
        {
            "hello": "world",
            "value": 42,
        },
    )

    result = load_yaml(path)

    assert result == {
        "hello": "world",
        "value": 42,
    }


def test_load_yaml_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Missing YAML files must fail loudly."""
    path = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError
    ):
        load_yaml(path)


def test_load_dataset_success(
    tmp_path: Path,
) -> None:
    """The shared loader should read and validate a correct dataset."""
    config_path, _ = _build_temporary_dataset(
        tmp_path
    )

    bundle = load_dataset(
        config_path
    )

    assert bundle.dataset_id == (
        "ml-latest-small"
    )

    assert len(bundle.ratings) == 4
    assert len(bundle.movies) == 3
    assert bundle.users is None

    assert list(
        bundle.ratings.columns
    ) == [
        "userId",
        "movieId",
        "rating",
        "timestamp",
    ]

    assert (
        bundle.ratings["userId"].dtype
        == "int64"
    )

    assert (
        bundle.ratings["movieId"].dtype
        == "int64"
    )

    assert (
        bundle.ratings["rating"].dtype
        == "float64"
    )

    assert (
        bundle.ratings["timestamp"].dtype
        == "int64"
    )


def test_load_dataset_rejects_manifest_mismatch(
    tmp_path: Path,
) -> None:
    """Config and manifest dataset IDs must agree."""
    config_path, _ = _build_temporary_dataset(
        tmp_path
    )

    config = load_yaml(
        config_path
    )

    config["dataset"]["id"] = "ml-1m"

    _write_yaml(
        config_path,
        config,
    )

    with pytest.raises(
        ValueError,
        match="Dataset/config mismatch",
    ):
        load_dataset(
            config_path
        )


def test_load_dataset_rejects_corrupted_archive(
    tmp_path: Path,
) -> None:
    """Archive corruption must be detected before data are loaded."""
    config_path, _ = _build_temporary_dataset(
        tmp_path
    )

    config = load_yaml(
        config_path
    )

    raw_root = Path(
        config["dataset"]["raw_root"]
    )

    archive_path = (
        raw_root
        / "ml-latest-small.zip"
    )

    archive_path.write_bytes(
        b"corrupted archive"
    )

    with pytest.raises(
        ValueError,
        match="Archive size mismatch|MD5 mismatch|SHA-256 mismatch",
    ):
        load_dataset(
            config_path
        )


def test_load_dataset_rejects_wrong_raw_statistics(
    tmp_path: Path,
) -> None:
    """Manifest statistics must match the actual raw dataset."""
    config_path, manifest_path = (
        _build_temporary_dataset(
            tmp_path
        )
    )

    manifest = load_yaml(
        manifest_path
    )

    manifest[
        "raw_statistics"
    ]["ratings"] = 999

    _write_yaml(
        manifest_path,
        manifest,
    )

    with pytest.raises(
        ValueError,
        match="Raw statistic mismatch",
    ):
        load_dataset(
            config_path
        )