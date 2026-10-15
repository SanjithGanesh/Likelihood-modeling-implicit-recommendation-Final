"""Dataset loading utilities for the Paper 2 experiment pipeline.

This module provides one authoritative path for loading MovieLens data.

Responsibilities
----------------
- Load an experiment configuration.
- Load and validate the corresponding dataset manifest.
- Verify raw archive integrity.
- Load MovieLens Latest Small or MovieLens 1M.
- Normalize the core ratings schema.
- Validate raw dataset statistics against the manifest.

This module intentionally does NOT perform:
- train/validation/test splitting,
- filtering,
- user/item reindexing,
- negative sampling,
- ranking evaluation.

Those responsibilities belong to later pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetBundle:
    """Container for one loaded raw MovieLens dataset."""

    dataset_id: str
    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame | None
    manifest: dict[str, Any]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at its top level."""
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping at top level: {path}"
        )

    return data


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve a project-relative path against the project root."""
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _hash_file(path: Path, algorithm: str) -> str:
    """Compute a hexadecimal file hash without loading the whole file at once."""
    hasher = hashlib.new(algorithm)

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_archive_integrity(
    manifest: dict[str, Any],
    raw_root: Path,
) -> None:
    """Verify archive size, MD5, and SHA-256 against the manifest."""
    source = manifest["source"]
    integrity = manifest["integrity"]

    archive_path = raw_root / source["archive_filename"]

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Raw archive is missing: {archive_path}"
        )

    expected_size = int(integrity["archive_size_bytes"])
    actual_size = archive_path.stat().st_size

    if actual_size != expected_size:
        raise ValueError(
            f"Archive size mismatch for {archive_path.name}: "
            f"expected {expected_size}, got {actual_size}"
        )

    expected_md5 = str(integrity["official_md5"]).lower()
    actual_md5 = _hash_file(archive_path, "md5")

    if actual_md5 != expected_md5:
        raise ValueError(
            f"MD5 mismatch for {archive_path.name}: "
            f"expected {expected_md5}, got {actual_md5}"
        )

    expected_sha256 = str(integrity["sha256"]).lower()
    actual_sha256 = _hash_file(archive_path, "sha256")

    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {archive_path.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def _normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate the common MovieLens ratings schema."""
    required_columns = [
        "userId",
        "movieId",
        "rating",
        "timestamp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in ratings.columns
    ]

    if missing:
        raise ValueError(
            f"Ratings data is missing required columns: {missing}"
        )

    ratings = ratings[required_columns].copy()

    ratings["userId"] = ratings["userId"].astype("int64")
    ratings["movieId"] = ratings["movieId"].astype("int64")
    ratings["rating"] = ratings["rating"].astype("float64")
    ratings["timestamp"] = ratings["timestamp"].astype("int64")

    if ratings.isna().any().any():
        raise ValueError("Ratings data contains missing values")

    return ratings


def _load_latest_small(
    dataset_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, None]:
    """Load MovieLens Latest Small."""
    ratings_path = dataset_root / "ratings.csv"
    movies_path = dataset_root / "movies.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Missing ratings file: {ratings_path}"
        )

    if not movies_path.exists():
        raise FileNotFoundError(
            f"Missing movies file: {movies_path}"
        )

    ratings = pd.read_csv(
        ratings_path,
        encoding="utf-8",
    )

    movies = pd.read_csv(
        movies_path,
        encoding="utf-8",
    )

    return ratings, movies, None


def _load_ml_1m(
    dataset_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 1M."""
    ratings_path = dataset_root / "ratings.dat"
    movies_path = dataset_root / "movies.dat"
    users_path = dataset_root / "users.dat"

    for path in (ratings_path, movies_path, users_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing MovieLens 1M file: {path}"
            )

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=[
            "movieId",
            "title",
            "genres",
        ],
        encoding="latin-1",
    )

    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=[
            "userId",
            "gender",
            "age",
            "occupation",
            "zip",
        ],
        encoding="latin-1",
    )

    return ratings, movies, users


def _validate_raw_statistics(
    bundle: DatasetBundle,
) -> None:
    """Validate loaded raw statistics against the dataset manifest."""
    stats = bundle.manifest["raw_statistics"]

    checks = {
        "ratings": len(bundle.ratings),
        "users": bundle.ratings["userId"].nunique(),
        "movies_metadata": len(bundle.movies),
        "rated_movies": bundle.ratings["movieId"].nunique(),
        "rating_min": float(bundle.ratings["rating"].min()),
        "rating_max": float(bundle.ratings["rating"].max()),
        "missing_values_in_ratings": int(
            bundle.ratings.isna().sum().sum()
        ),
    }

    for key, actual in checks.items():
        if key not in stats:
            raise ValueError(
                f"Manifest is missing raw statistic: {key}"
            )

        expected = stats[key]

        if actual != expected:
            raise ValueError(
                f"Raw statistic mismatch for '{key}': "
                f"expected {expected}, got {actual}"
            )

    if bundle.users is not None:
        expected_users_file = stats.get(
            "missing_values_in_users"
        )

        if expected_users_file is not None:
            actual_missing_users = int(
                bundle.users.isna().sum().sum()
            )

            if actual_missing_users != expected_users_file:
                raise ValueError(
                    "Raw statistic mismatch for "
                    "'missing_values_in_users': "
                    f"expected {expected_users_file}, "
                    f"got {actual_missing_users}"
                )

    expected_missing_movies = stats.get(
        "missing_values_in_movies"
    )

    if expected_missing_movies is not None:
        actual_missing_movies = int(
            bundle.movies.isna().sum().sum()
        )

        if actual_missing_movies != expected_missing_movies:
            raise ValueError(
                "Raw statistic mismatch for "
                "'missing_values_in_movies': "
                f"expected {expected_missing_movies}, "
                f"got {actual_missing_movies}"
            )


def load_dataset(
    config_path: str | Path,
    verify_integrity: bool = True,
) -> DatasetBundle:
    """Load and validate the raw dataset specified by a config file."""
    config = load_yaml(config_path)

    if "dataset" not in config:
        raise ValueError(
            "Configuration is missing the 'dataset' section"
        )

    dataset_config = config["dataset"]

    required_dataset_fields = [
        "id",
        "manifest",
        "raw_root",
    ]

    missing_fields = [
        field
        for field in required_dataset_fields
        if field not in dataset_config
    ]

    if missing_fields:
        raise ValueError(
            "Configuration dataset section is missing fields: "
            f"{missing_fields}"
        )

    dataset_id = str(dataset_config["id"])

    manifest = load_yaml(dataset_config["manifest"])

    manifest_dataset_id = str(
        manifest["dataset"]["id"]
    )

    if dataset_id != manifest_dataset_id:
        raise ValueError(
            "Dataset/config mismatch: "
            f"config requests '{dataset_id}' but manifest "
            f"describes '{manifest_dataset_id}'"
        )

    raw_root = _resolve_project_path(
        dataset_config["raw_root"]
    )

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw data root does not exist: {raw_root}"
        )

    if verify_integrity:
        verify_archive_integrity(
            manifest=manifest,
            raw_root=raw_root,
        )

    extracted_directory = manifest["source"][
        "extracted_directory"
    ]

    dataset_root = raw_root / extracted_directory

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Extracted dataset directory is missing: "
            f"{dataset_root}"
        )

    if dataset_id == "ml-latest-small":
        ratings, movies, users = _load_latest_small(
            dataset_root
        )
    elif dataset_id == "ml-1m":
        ratings, movies, users = _load_ml_1m(
            dataset_root
        )
    else:
        raise ValueError(
            f"Unsupported dataset id: {dataset_id}"
        )

    ratings = _normalize_ratings(ratings)

    bundle = DatasetBundle(
        dataset_id=dataset_id,
        ratings=ratings,
        movies=movies,
        users=users,
        manifest=manifest,
    )

    _validate_raw_statistics(bundle)

    return bundle