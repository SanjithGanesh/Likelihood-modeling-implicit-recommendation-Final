"""Structured result storage and provenance management for Paper 2.

This module owns the filesystem contract for experiment outputs.

Run layout
----------
results/
└── <run_id>/
    ├── run_manifest.json
    └── experiments/
        └── <experiment_id>/
            ├── result.json
            ├── metrics.csv
            └── training_history.json

Design rules
------------
- every run has one immutable provenance manifest;
- run IDs are unique;
- experiment IDs are deterministic within a run;
- experiment directories are never silently overwritten;
- JSON is UTF-8, deterministic, and rejects NaN/Infinity;
- metrics.csv uses one strict long-format schema;
- stale/mixed provenance is rejected;
- result writes are staged and moved atomically;
- verified completed experiment directories are safe resume checkpoints;
- processed-data and split files are integrity-checked before
  provenance is accepted;
- code_version is a deterministic SHA-256 over src/ and scripts/
  Python source because this project is not currently a Git repository.

This module does not aggregate experiments or produce figures.
Those responsibilities belong to later pipeline stages.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import yaml

from src.experiment import (
    ExperimentResult,
    ExperimentSpec,
)


# ============================================================
# Constants
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


RESULT_SCHEMA_VERSION = "1.0"


RUN_MANIFEST_FILENAME = (
    "run_manifest.json"
)


EXPERIMENTS_DIRECTORY_NAME = (
    "experiments"
)


RESULT_FILENAME = (
    "result.json"
)


METRICS_FILENAME = (
    "metrics.csv"
)


TRAINING_HISTORY_FILENAME = (
    "training_history.json"
)


RUN_MANIFEST_KEYS = {
    "schema_version",
    "run_id",
    "dataset_id",
    "artifact_version",
    "config_hash",
    "dataset_hash",
    "split_hash",
    "code_version",
    "created_at_utc",
}


METRICS_COLUMNS = [
    "schema_version",
    "run_id",
    "experiment_id",
    "model_name",
    "family",
    "variant",
    "seed",
    "target_split",
    "metric_scope",
    "metric_name",
    "metric_value",
]


_HASH_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9]"
    r"[A-Za-z0-9._-]{0,191}$"
)


# ============================================================
# Dataclasses
# ============================================================


@dataclass(
    frozen=True,
)
class RunProvenance:
    """Immutable scientific provenance for one run."""

    dataset_id: str

    artifact_version: str

    config_hash: str

    dataset_hash: str

    split_hash: str

    code_version: str

    def __post_init__(
        self,
    ) -> None:

        _validate_nonempty_string(
            self.dataset_id,
            name="dataset_id",
        )

        _validate_nonempty_string(
            self.artifact_version,
            name="artifact_version",
        )

        for name in [
            "config_hash",
            "dataset_hash",
            "split_hash",
            "code_version",
        ]:
            _validate_sha256(
                getattr(
                    self,
                    name,
                ),
                name=name,
            )

    def to_dict(
        self,
    ) -> dict[
        str,
        str,
    ]:

        return {
            "dataset_id": self.dataset_id,
            "artifact_version": (
                self.artifact_version
            ),
            "config_hash": self.config_hash,
            "dataset_hash": (
                self.dataset_hash
            ),
            "split_hash": self.split_hash,
            "code_version": (
                self.code_version
            ),
        }


@dataclass(
    frozen=True,
)
class RunContext:
    """Filesystem identity and provenance of one result run."""

    run_id: str

    run_dir: Path

    provenance: RunProvenance

    created_at_utc: str

    def __post_init__(
        self,
    ) -> None:

        _validate_identifier(
            self.run_id,
            name="run_id",
        )

        if not isinstance(
            self.run_dir,
            Path,
        ):
            raise TypeError(
                "run_dir must be a pathlib.Path"
            )

        if not isinstance(
            self.provenance,
            RunProvenance,
        ):
            raise TypeError(
                "provenance must be RunProvenance"
            )

        _validate_utc_timestamp(
            self.created_at_utc
        )

    @property
    def manifest_path(
        self,
    ) -> Path:

        return (
            self.run_dir
            /
            RUN_MANIFEST_FILENAME
        )

    @property
    def experiments_dir(
        self,
    ) -> Path:

        return (
            self.run_dir
            /
            EXPERIMENTS_DIRECTORY_NAME
        )


@dataclass(
    frozen=True,
)
class ExperimentArtifacts:
    """Paths written for one stored experiment."""

    experiment_id: str

    experiment_dir: Path

    result_path: Path

    metrics_path: Path

    training_history_path: Path


# ============================================================
# Generic validation helpers
# ============================================================


def _validate_nonempty_string(
    value: Any,
    *,
    name: str,
) -> None:

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string"
        )

    if not value.strip():
        raise ValueError(
            f"{name} must not be empty"
        )


def _validate_sha256(
    value: Any,
    *,
    name: str,
) -> None:

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string"
        )

    if (
        _HASH_PATTERN.fullmatch(
            value
        )
        is None
    ):
        raise ValueError(
            f"{name} must be a lowercase "
            "64-character SHA-256 hex digest"
        )


def _validate_identifier(
    value: Any,
    *,
    name: str,
) -> None:

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string"
        )

    if (
        _IDENTIFIER_PATTERN.fullmatch(
            value
        )
        is None
    ):
        raise ValueError(
            f"{name} contains invalid characters "
            "or has invalid length"
        )

    if (
        ".."
        in value
    ):
        raise ValueError(
            f"{name} must not contain '..'"
        )


def _validate_utc_timestamp(
    value: Any,
) -> None:

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "created_at_utc must be a string"
        )

    if not value.endswith(
        "Z"
    ):
        raise ValueError(
            "created_at_utc must end with 'Z'"
        )

    try:
        parsed = (
            datetime.fromisoformat(
                value[
                    :-1
                ]
                +
                "+00:00"
            )
        )
    except ValueError as exc:
        raise ValueError(
            "created_at_utc is not a valid "
            "ISO-8601 UTC timestamp"
        ) from exc

    if (
        parsed.utcoffset()
        is None
        or parsed.utcoffset().total_seconds()
        !=
        0.0
    ):
        raise ValueError(
            "created_at_utc must represent UTC"
        )


# ============================================================
# JSON normalization
# ============================================================


def _json_safe(
    value: Any,
) -> Any:
    """Convert common scientific-Python values to JSON-safe objects."""

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
            float,
        ),
    ):

        if isinstance(
            value,
            float,
        ):
            if not np.isfinite(
                value
            ):
                raise ValueError(
                    "JSON payload contains "
                    "NaN or infinite float"
                )

        return value

    if isinstance(
        value,
        np.generic,
    ):
        return _json_safe(
            value.item()
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            _json_safe(
                item
            )
            for item
            in value.tolist()
        ]

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        Mapping,
    ):
        normalized = {}

        for key, child in value.items():

            if not isinstance(
                key,
                str,
            ):
                raise TypeError(
                    "JSON object keys must be strings"
                )

            normalized[
                key
            ] = _json_safe(
                child
            )

        return normalized

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(
                child
            )
            for child
            in value
        ]

    raise TypeError(
        "Unsupported JSON value type: "
        f"{type(value).__name__}"
    )


def _canonical_json_bytes(
    value: Any,
) -> bytes:

    safe = _json_safe(
        value
    )

    return json.dumps(
        safe,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    ).encode(
        "utf-8"
    )


def _pretty_json_text(
    value: Any,
) -> str:

    safe = _json_safe(
        value
    )

    return (
        json.dumps(
            safe,
            sort_keys=False,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    )


# ============================================================
# Hashing
# ============================================================


def sha256_file(
    path: str | Path,
) -> str:
    """Compute SHA-256 of one file."""

    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    hasher = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024
                *
                1024
            ),
            b"",
        ):
            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def semantic_yaml_hash(
    data: Mapping[
        str,
        Any,
    ],
) -> str:
    """Hash YAML semantics exactly like prepare_data.py."""

    if not isinstance(
        data,
        Mapping,
    ):
        raise TypeError(
            "data must be a mapping"
        )

    canonical = yaml.safe_dump(
        dict(
            data
        ),
        sort_keys=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        canonical
    ).hexdigest()


def source_tree_hash(
    project_root: str | Path = PROJECT_ROOT,
) -> str:
    """Hash Paper 2 Python source deterministically.

    Only production Python code under src/ and scripts/ contributes.
    Tests, data, result files, caches, and virtual environments do not.
    """

    project_root = Path(
        project_root
    ).resolve()

    source_roots = [
        project_root
        /
        "src",
        project_root
        /
        "scripts",
    ]

    source_files: list[
        Path
    ] = []

    for root in source_roots:

        if not root.is_dir():
            raise FileNotFoundError(
                f"Source directory does not exist: {root}"
            )

        for path in root.rglob(
            "*.py"
        ):

            if (
                "__pycache__"
                in path.parts
            ):
                continue

            source_files.append(
                path
            )

    source_files = sorted(
        source_files,
        key=lambda path: (
            path.relative_to(
                project_root
            )
            .as_posix()
        ),
    )

    if not source_files:
        raise ValueError(
            "No Python source files found"
        )

    hasher = hashlib.sha256()

    for path in source_files:

        relative = (
            path.relative_to(
                project_root
            )
            .as_posix()
        )

        hasher.update(
            relative.encode(
                "utf-8"
            )
        )

        hasher.update(
            b"\0"
        )

        with path.open(
            "rb"
        ) as file:

            for chunk in iter(
                lambda: file.read(
                    1024
                    *
                    1024
                ),
                b"",
            ):
                hasher.update(
                    chunk
                )

        hasher.update(
            b"\0"
        )

    return hasher.hexdigest()


# ============================================================
# YAML / path helpers
# ============================================================


def _load_yaml_mapping(
    path: str | Path,
) -> dict[
    str,
    Any,
]:

    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"YAML file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            f"YAML root must be a mapping: {path}"
        )

    return data


def _resolve_project_path(
    raw_path: str | Path,
    *,
    project_root: Path,
) -> Path:
    """Resolve manifest paths across Windows/POSIX separators."""

    if isinstance(
        raw_path,
        Path,
    ):
        path = raw_path
    elif isinstance(
        raw_path,
        str,
    ):
        path = Path(
            raw_path.replace(
                "\\",
                "/",
            )
        )
    else:
        raise TypeError(
            "Manifest path must be a string or Path"
        )

    if path.is_absolute():
        return path

    return (
        project_root
        /
        path
    )


def _nested_mapping(
    data: Mapping[
        str,
        Any,
    ],
    key: str,
    *,
    source_name: str,
) -> Mapping[
    str,
    Any,
]:

    value = data.get(
        key
    )

    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(
            f"{source_name}.{key} must be a mapping"
        )

    return value


def _required_string(
    data: Mapping[
        str,
        Any,
    ],
    key: str,
    *,
    source_name: str,
) -> str:

    if key not in data:
        raise KeyError(
            f"Missing field: "
            f"{source_name}.{key}"
        )

    value = data[
        key
    ]

    _validate_nonempty_string(
        value,
        name=(
            f"{source_name}.{key}"
        ),
    )

    return value


def _required_hash(
    data: Mapping[
        str,
        Any,
    ],
    key: str,
    *,
    source_name: str,
) -> str:

    value = _required_string(
        data,
        key,
        source_name=source_name,
    )

    _validate_sha256(
        value,
        name=(
            f"{source_name}.{key}"
        ),
    )

    return value


# ============================================================
# Provenance construction
# ============================================================


def build_run_provenance(
    *,
    config_path: str | Path,
    preparation_manifest_path: str | Path,
    split_manifest_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    verify_files: bool = True,
) -> RunProvenance:
    """Build and validate scientific provenance for one run."""

    if not isinstance(
        verify_files,
        bool,
    ):
        raise TypeError(
            "verify_files must be a boolean"
        )

    project_root = Path(
        project_root
    ).resolve()

    config = (
        _load_yaml_mapping(
            config_path
        )
    )

    preparation = (
        _load_yaml_mapping(
            preparation_manifest_path
        )
    )

    split = (
        _load_yaml_mapping(
            split_manifest_path
        )
    )

    prep_artifact = (
        _nested_mapping(
            preparation,
            "artifact",
            source_name=(
                "preparation_manifest"
            ),
        )
    )

    prep_source = (
        _nested_mapping(
            preparation,
            "source",
            source_name=(
                "preparation_manifest"
            ),
        )
    )

    prep_files = (
        _nested_mapping(
            preparation,
            "files",
            source_name=(
                "preparation_manifest"
            ),
        )
    )

    split_artifact = (
        _nested_mapping(
            split,
            "artifact",
            source_name=(
                "split_manifest"
            ),
        )
    )

    split_source = (
        _nested_mapping(
            split,
            "source",
            source_name=(
                "split_manifest"
            ),
        )
    )

    split_files = (
        _nested_mapping(
            split,
            "files",
            source_name=(
                "split_manifest"
            ),
        )
    )

    prep_type = (
        _required_string(
            prep_artifact,
            "type",
            source_name=(
                "preparation_manifest.artifact"
            ),
        )
    )

    if (
        prep_type
        !=
        "processed_dataset"
    ):
        raise ValueError(
            "preparation manifest artifact.type "
            "must be 'processed_dataset'"
        )

    split_type = (
        _required_string(
            split_artifact,
            "type",
            source_name=(
                "split_manifest.artifact"
            ),
        )
    )

    if (
        split_type
        !=
        "canonical_split"
    ):
        raise ValueError(
            "split manifest artifact.type "
            "must be 'canonical_split'"
        )

    dataset_id = (
        _required_string(
            prep_artifact,
            "dataset_id",
            source_name=(
                "preparation_manifest.artifact"
            ),
        )
    )

    split_dataset_id = (
        _required_string(
            split_artifact,
            "dataset_id",
            source_name=(
                "split_manifest.artifact"
            ),
        )
    )

    if (
        dataset_id
        !=
        split_dataset_id
    ):
        raise ValueError(
            "Preparation and split manifests "
            "refer to different dataset IDs"
        )

    artifact_version = (
        _required_string(
            prep_artifact,
            "version",
            source_name=(
                "preparation_manifest.artifact"
            ),
        )
    )

    split_version = (
        _required_string(
            split_artifact,
            "version",
            source_name=(
                "split_manifest.artifact"
            ),
        )
    )

    if (
        artifact_version
        !=
        split_version
    ):
        raise ValueError(
            "Preparation and split manifests "
            "refer to different artifact versions"
        )

    prep_config_hash = (
        _required_hash(
            prep_source,
            "config_sha256",
            source_name=(
                "preparation_manifest.source"
            ),
        )
    )

    split_config_hash = (
        _required_hash(
            split_source,
            "config_sha256",
            source_name=(
                "split_manifest.source"
            ),
        )
    )

    if (
        prep_config_hash
        !=
        split_config_hash
    ):
        raise ValueError(
            "Preparation and split manifests "
            "contain different config hashes"
        )

    actual_config_hash = (
        semantic_yaml_hash(
            config
        )
    )

    if (
        actual_config_hash
        !=
        prep_config_hash
    ):
        raise ValueError(
            "Current configuration does not match "
            "the frozen artifact config hash"
        )

    ratings_file = (
        _nested_mapping(
            prep_files,
            "ratings",
            source_name=(
                "preparation_manifest.files"
            ),
        )
    )

    dataset_hash = (
        _required_hash(
            ratings_file,
            "sha256",
            source_name=(
                "preparation_manifest.files.ratings"
            ),
        )
    )

    split_dataset_hash = (
        _required_hash(
            split_source,
            "processed_dataset_sha256",
            source_name=(
                "split_manifest.source"
            ),
        )
    )

    if (
        dataset_hash
        !=
        split_dataset_hash
    ):
        raise ValueError(
            "Split manifest does not reference "
            "the preparation manifest dataset hash"
        )

    split_hash = (
        _required_hash(
            split,
            "split_sha256",
            source_name=(
                "split_manifest"
            ),
        )
    )

    if verify_files:

        ratings_path_raw = (
            _required_string(
                ratings_file,
                "path",
                source_name=(
                    "preparation_manifest.files.ratings"
                ),
            )
        )

        ratings_path = (
            _resolve_project_path(
                ratings_path_raw,
                project_root=(
                    project_root
                ),
            )
        )

        actual_dataset_hash = (
            sha256_file(
                ratings_path
            )
        )

        if (
            actual_dataset_hash
            !=
            dataset_hash
        ):
            raise ValueError(
                "Processed ratings file SHA-256 "
                "does not match its manifest"
            )

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            split_file = (
                _nested_mapping(
                    split_files,
                    split_name,
                    source_name=(
                        "split_manifest.files"
                    ),
                )
            )

            expected_hash = (
                _required_hash(
                    split_file,
                    "sha256",
                    source_name=(
                        "split_manifest.files."
                        f"{split_name}"
                    ),
                )
            )

            path_raw = (
                _required_string(
                    split_file,
                    "path",
                    source_name=(
                        "split_manifest.files."
                        f"{split_name}"
                    ),
                )
            )

            actual_hash = (
                sha256_file(
                    _resolve_project_path(
                        path_raw,
                        project_root=(
                            project_root
                        ),
                    )
                )
            )

            if (
                actual_hash
                !=
                expected_hash
            ):
                raise ValueError(
                    f"{split_name} split file SHA-256 "
                    "does not match its manifest"
                )

    code_version = (
        source_tree_hash(
            project_root
        )
    )

    return RunProvenance(
        dataset_id=dataset_id,
        artifact_version=(
            artifact_version
        ),
        config_hash=(
            actual_config_hash
        ),
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        code_version=code_version,
    )


# ============================================================
# Run IDs / experiment IDs
# ============================================================


def _slug(
    value: str,
) -> str:

    value = value.strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    value = value.strip(
        "-"
    )

    if not value:
        return "x"

    return value[
        :48
    ]


def _utc_now_string() -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    )


def generate_run_id(
    provenance: RunProvenance,
) -> str:
    """Generate one unique, provenance-readable run ID."""

    if not isinstance(
        provenance,
        RunProvenance,
    ):
        raise TypeError(
            "provenance must be RunProvenance"
        )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )

    identifier = (
        "run-"
        f"{timestamp}-"
        f"{_slug(provenance.dataset_id)}-"
        f"{provenance.split_hash[:8]}-"
        f"{provenance.code_version[:8]}-"
        f"{uuid4().hex[:8]}"
    )

    _validate_identifier(
        identifier,
        name="run_id",
    )

    return identifier


def _experiment_id_from_identity(
    *,
    model_name: str,
    family: str,
    variant: str,
    seed: int,
    target_split: str,
    hyperparameters: Mapping[
        str,
        Any,
    ],
) -> str:
    """Create the canonical deterministic experiment identifier."""

    for name, value in [
        (
            "model_name",
            model_name,
        ),
        (
            "family",
            family,
        ),
        (
            "variant",
            variant,
        ),
        (
            "target_split",
            target_split,
        ),
    ]:
        _validate_nonempty_string(
            value,
            name=name,
        )

    if (
        not isinstance(
            seed,
            int,
        )
        or isinstance(
            seed,
            bool,
        )
    ):
        raise TypeError(
            "seed must be an integer"
        )

    if not isinstance(
        hyperparameters,
        Mapping,
    ):
        raise TypeError(
            "hyperparameters must be a mapping"
        )

    identity = {
        "model_name": model_name,
        "family": family,
        "variant": variant,
        "seed": seed,
        "target_split": target_split,
        "hyperparameters": dict(
            hyperparameters
        ),
    }

    digest = hashlib.sha256(
        _canonical_json_bytes(
            identity
        )
    ).hexdigest()

    identifier = (
        f"{_slug(model_name)}-"
        f"{_slug(variant)}-"
        f"s{seed}-"
        f"{_slug(target_split)}-"
        f"{digest[:12]}"
    )

    _validate_identifier(
        identifier,
        name="experiment_id",
    )

    return identifier


def experiment_id_for_spec(
    spec: ExperimentSpec,
    *,
    target_split: str,
) -> str:
    """Create the expected experiment ID before fitting a model."""

    if not isinstance(
        spec,
        ExperimentSpec,
    ):
        raise TypeError(
            "spec must be an ExperimentSpec"
        )

    return _experiment_id_from_identity(
        model_name=spec.model_name,
        family=spec.family,
        variant=spec.variant,
        seed=spec.seed,
        target_split=target_split,
        hyperparameters=(
            spec.hyperparameters
        ),
    )


def experiment_id_for_result(
    result: ExperimentResult,
) -> str:
    """Create a deterministic experiment ID from scientific identity."""

    if not isinstance(
        result,
        ExperimentResult,
    ):
        raise TypeError(
            "result must be an ExperimentResult"
        )

    hyperparameters = (
        result.metadata.get(
            "hyperparameters",
            {},
        )
    )

    if not isinstance(
        hyperparameters,
        Mapping,
    ):
        raise TypeError(
            "result.metadata['hyperparameters'] "
            "must be a mapping when present"
        )

    return _experiment_id_from_identity(
        model_name=result.model_name,
        family=result.family,
        variant=result.variant,
        seed=result.seed,
        target_split=(
            result.target_split
        ),
        hyperparameters=hyperparameters,
    )


# ============================================================
# Atomic filesystem helpers
# ============================================================


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:

    if not isinstance(
        content,
        str,
    ):
        raise TypeError(
            "content must be a string"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.parent
        /
        (
            "."
            f"{path.name}"
            ".tmp-"
            f"{uuid4().hex}"
        )
    )

    try:

        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:

            file.write(
                content
            )

            file.flush()

        temporary.replace(
            path
        )

    finally:

        if temporary.exists():
            temporary.unlink()


def _write_json(
    path: Path,
    payload: Any,
) -> None:

    _atomic_write_text(
        path,
        _pretty_json_text(
            payload
        ),
    )


# ============================================================
# Run manifest handling
# ============================================================


def _run_manifest_payload(
    *,
    run_id: str,
    provenance: RunProvenance,
    created_at_utc: str,
) -> dict[
    str,
    Any,
]:

    payload = {
        "schema_version": (
            RESULT_SCHEMA_VERSION
        ),
        "run_id": run_id,
        "dataset_id": (
            provenance.dataset_id
        ),
        "artifact_version": (
            provenance.artifact_version
        ),
        "config_hash": (
            provenance.config_hash
        ),
        "dataset_hash": (
            provenance.dataset_hash
        ),
        "split_hash": (
            provenance.split_hash
        ),
        "code_version": (
            provenance.code_version
        ),
        "created_at_utc": (
            created_at_utc
        ),
    }

    _validate_run_manifest_payload(
        payload
    )

    return payload


def _validate_run_manifest_payload(
    payload: Any,
) -> None:

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "run manifest must be a dictionary"
        )

    keys = set(
        payload.keys()
    )

    if (
        keys
        !=
        RUN_MANIFEST_KEYS
    ):
        missing = (
            RUN_MANIFEST_KEYS
            -
            keys
        )

        extra = (
            keys
            -
            RUN_MANIFEST_KEYS
        )

        raise ValueError(
            "Run manifest schema mismatch. "
            f"Missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    if (
        payload[
            "schema_version"
        ]
        !=
        RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported result schema version"
        )

    _validate_identifier(
        payload[
            "run_id"
        ],
        name="run_id",
    )

    _validate_nonempty_string(
        payload[
            "dataset_id"
        ],
        name="dataset_id",
    )

    _validate_nonempty_string(
        payload[
            "artifact_version"
        ],
        name="artifact_version",
    )

    for name in [
        "config_hash",
        "dataset_hash",
        "split_hash",
        "code_version",
    ]:
        _validate_sha256(
            payload[
                name
            ],
            name=name,
        )

    _validate_utc_timestamp(
        payload[
            "created_at_utc"
        ]
    )


def create_run(
    *,
    provenance: RunProvenance,
    output_root: str | Path = "results",
    run_id: str | None = None,
    created_at_utc: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> RunContext:
    """Create one clean versioned run directory."""

    if not isinstance(
        provenance,
        RunProvenance,
    ):
        raise TypeError(
            "provenance must be RunProvenance"
        )

    project_root = Path(
        project_root
    ).resolve()

    output_root_path = Path(
        output_root
    )

    if not output_root_path.is_absolute():
        output_root_path = (
            project_root
            /
            output_root_path
        )

    if run_id is None:
        run_id = (
            generate_run_id(
                provenance
            )
        )

    _validate_identifier(
        run_id,
        name="run_id",
    )

    if created_at_utc is None:
        created_at_utc = (
            _utc_now_string()
        )

    _validate_utc_timestamp(
        created_at_utc
    )

    final_run_dir = (
        output_root_path
        /
        run_id
    )

    if final_run_dir.exists():
        raise FileExistsError(
            "Run directory already exists: "
            f"{final_run_dir}"
        )

    output_root_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = (
        output_root_path
        /
        (
            "."
            f"{run_id}"
            ".tmp-"
            f"{uuid4().hex}"
        )
    )

    if staging_dir.exists():
        raise FileExistsError(
            f"Staging directory already exists: "
            f"{staging_dir}"
        )

    try:

        staging_dir.mkdir(
            parents=False,
            exist_ok=False,
        )

        (
            staging_dir
            /
            EXPERIMENTS_DIRECTORY_NAME
        ).mkdir(
            parents=False,
            exist_ok=False,
        )

        manifest_payload = (
            _run_manifest_payload(
                run_id=run_id,
                provenance=provenance,
                created_at_utc=(
                    created_at_utc
                ),
            )
        )

        _write_json(
            staging_dir
            /
            RUN_MANIFEST_FILENAME,
            manifest_payload,
        )

        staging_dir.replace(
            final_run_dir
        )

    except Exception:

        if staging_dir.exists():
            shutil.rmtree(
                staging_dir
            )

        raise

    return RunContext(
        run_id=run_id,
        run_dir=final_run_dir,
        provenance=provenance,
        created_at_utc=(
            created_at_utc
        ),
    )


def create_run_from_artifacts(
    *,
    config_path: str | Path,
    preparation_manifest_path: str | Path,
    split_manifest_path: str | Path,
    output_root: str | Path = "results",
    run_id: str | None = None,
    created_at_utc: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
    verify_files: bool = True,
) -> RunContext:
    """Validate provenance then create one clean run."""

    provenance = (
        build_run_provenance(
            config_path=config_path,
            preparation_manifest_path=(
                preparation_manifest_path
            ),
            split_manifest_path=(
                split_manifest_path
            ),
            project_root=project_root,
            verify_files=verify_files,
        )
    )

    return create_run(
        provenance=provenance,
        output_root=output_root,
        run_id=run_id,
        created_at_utc=created_at_utc,
        project_root=project_root,
    )


def load_run_context(
    run_dir: str | Path,
) -> RunContext:
    """Open and validate an existing run directory."""

    run_dir = Path(
        run_dir
    ).resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}"
        )

    manifest_path = (
        run_dir
        /
        RUN_MANIFEST_FILENAME
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Run manifest does not exist: "
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )

    _validate_run_manifest_payload(
        payload
    )

    if (
        payload[
            "run_id"
        ]
        !=
        run_dir.name
    ):
        raise ValueError(
            "Run manifest run_id does not match "
            "its directory name"
        )

    experiments_dir = (
        run_dir
        /
        EXPERIMENTS_DIRECTORY_NAME
    )

    if not experiments_dir.is_dir():
        raise FileNotFoundError(
            "Run experiments directory "
            "does not exist"
        )

    provenance = RunProvenance(
        dataset_id=(
            payload[
                "dataset_id"
            ]
        ),
        artifact_version=(
            payload[
                "artifact_version"
            ]
        ),
        config_hash=(
            payload[
                "config_hash"
            ]
        ),
        dataset_hash=(
            payload[
                "dataset_hash"
            ]
        ),
        split_hash=(
            payload[
                "split_hash"
            ]
        ),
        code_version=(
            payload[
                "code_version"
            ]
        ),
    )

    return RunContext(
        run_id=(
            payload[
                "run_id"
            ]
        ),
        run_dir=run_dir,
        provenance=provenance,
        created_at_utc=(
            payload[
                "created_at_utc"
            ]
        ),
    )


def _assert_run_context_matches_disk(
    run: RunContext,
) -> None:
    """Prevent writing into a run whose manifest changed."""

    disk = load_run_context(
        run.run_dir
    )

    if (
        disk.run_id
        !=
        run.run_id
    ):
        raise ValueError(
            "Run ID mismatch"
        )

    if (
        disk.provenance
        !=
        run.provenance
    ):
        raise ValueError(
            "Run provenance differs from "
            "the on-disk manifest"
        )

    if (
        disk.created_at_utc
        !=
        run.created_at_utc
    ):
        raise ValueError(
            "Run timestamp differs from "
            "the on-disk manifest"
        )


# ============================================================
# Experiment storage payloads
# ============================================================


def _result_payload(
    *,
    run: RunContext,
    experiment_id: str,
    result: ExperimentResult,
) -> dict[
    str,
    Any,
]:

    return {
        "schema_version": (
            RESULT_SCHEMA_VERSION
        ),
        "run_id": (
            run.run_id
        ),
        "experiment_id": (
            experiment_id
        ),
        "config_hash": (
            run.provenance.config_hash
        ),
        "dataset_hash": (
            run.provenance.dataset_hash
        ),
        "split_hash": (
            run.provenance.split_hash
        ),
        "code_version": (
            run.provenance.code_version
        ),
        "result": (
            result.to_dict()
        ),
    }


def _training_history_payload(
    *,
    run: RunContext,
    experiment_id: str,
    result: ExperimentResult,
) -> dict[
    str,
    Any,
]:

    return {
        "schema_version": (
            RESULT_SCHEMA_VERSION
        ),
        "run_id": (
            run.run_id
        ),
        "experiment_id": (
            experiment_id
        ),
        "records": (
            result.training_history
        ),
    }


def _metric_rows(
    *,
    run: RunContext,
    experiment_id: str,
    result: ExperimentResult,
) -> list[
    dict[
        str,
        Any,
    ]
]:

    rows = []

    for scope, metrics in [
        (
            "ranking",
            result.ranking_metrics,
        ),
        (
            "native",
            result.native_metrics,
        ),
    ]:

        for metric_name in sorted(
            metrics.keys()
        ):

            value = float(
                metrics[
                    metric_name
                ]
            )

            if not np.isfinite(
                value
            ):
                raise ValueError(
                    "Metric values must be finite"
                )

            rows.append(
                {
                    "schema_version": (
                        RESULT_SCHEMA_VERSION
                    ),
                    "run_id": (
                        run.run_id
                    ),
                    "experiment_id": (
                        experiment_id
                    ),
                    "model_name": (
                        result.model_name
                    ),
                    "family": (
                        result.family
                    ),
                    "variant": (
                        result.variant
                    ),
                    "seed": (
                        int(
                            result.seed
                        )
                    ),
                    "target_split": (
                        result.target_split
                    ),
                    "metric_scope": (
                        scope
                    ),
                    "metric_name": (
                        metric_name
                    ),
                    "metric_value": (
                        value
                    ),
                }
            )

    return rows


def _write_metrics_csv(
    path: Path,
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:

    for row in rows:

        if (
            set(
                row.keys()
            )
            !=
            set(
                METRICS_COLUMNS
            )
        ):
            raise ValueError(
                "Metric row schema mismatch"
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.parent
        /
        (
            "."
            f"{path.name}"
            ".tmp-"
            f"{uuid4().hex}"
        )
    )

    try:

        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = (
                csv.DictWriter(
                    file,
                    fieldnames=(
                        METRICS_COLUMNS
                    ),
                    extrasaction="raise",
                )
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    row
                )

            file.flush()

        temporary.replace(
            path
        )

    finally:

        if temporary.exists():
            temporary.unlink()


# ============================================================
# Completed-experiment checkpoint / resume loading
# ============================================================


def _load_json_file(
    path: Path,
    *,
    description: str,
) -> Any:
    """Load one required JSON artifact with a clear corruption error."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(
                file
            )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError(
            f"Invalid {description}: {path}"
        ) from exc


def _experiment_result_from_payload(
    payload: Any,
) -> ExperimentResult:
    """Reconstruct one validated ExperimentResult from stored JSON."""

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            "Stored result.result must be a mapping"
        )

    try:
        result = ExperimentResult(
            **dict(
                payload
            )
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Stored ExperimentResult payload is invalid"
        ) from exc

    if result.status != "completed":
        raise ValueError(
            "Only completed experiments may be resumed"
        )

    return result


def _assert_metrics_csv_matches(
    *,
    path: Path,
    expected_rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """Require metrics.csv to match the stored ExperimentResult exactly."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing experiment metrics CSV: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames != METRICS_COLUMNS:
            raise ValueError(
                "Stored metrics.csv schema does not match "
                "the current strict schema"
            )

        actual_rows = list(
            reader
        )

    expected_text_rows = [
        {
            column: str(
                row[
                    column
                ]
            )
            for column in METRICS_COLUMNS
        }
        for row in expected_rows
    ]

    if actual_rows != expected_text_rows:
        raise ValueError(
            "Stored metrics.csv does not match result.json"
        )


def load_completed_experiment(
    *,
    run: RunContext,
    spec: ExperimentSpec,
    target_split: str,
) -> tuple[
    ExperimentResult,
    ExperimentArtifacts,
] | None:
    """Load a verified completed experiment checkpoint if it exists.

    A missing deterministic experiment directory means the experiment has not
    completed and returns ``None``. If the directory exists, every persisted
    artifact must be present and exactly consistent with the current run,
    expected scientific identity, and one another. Corrupt, incomplete, or
    mismatched checkpoints fail closed and are never silently overwritten.
    """

    if not isinstance(
        run,
        RunContext,
    ):
        raise TypeError(
            "run must be a RunContext"
        )

    if not isinstance(
        spec,
        ExperimentSpec,
    ):
        raise TypeError(
            "spec must be an ExperimentSpec"
        )

    experiment_id = (
        experiment_id_for_spec(
            spec,
            target_split=target_split,
        )
    )

    experiment_dir = (
        run.experiments_dir
        /
        experiment_id
    )

    # Absence means there is no completed experiment checkpoint to resume.
    # The writer remains responsible for validating the run context before a
    # new result is committed. This ordering also keeps the loader usable in
    # isolated runner tests that inject an in-memory/fake RunContext.
    if not experiment_dir.exists():
        return None

    # Once a candidate checkpoint exists, fail closed: the run manifest and
    # provenance must be valid before any persisted experiment is trusted.
    _assert_run_context_matches_disk(
        run
    )

    if not experiment_dir.is_dir():
        raise ValueError(
            "Expected experiment checkpoint path is not a directory: "
            f"{experiment_dir}"
        )

    artifacts = ExperimentArtifacts(
        experiment_id=experiment_id,
        experiment_dir=(
            experiment_dir
        ),
        result_path=(
            experiment_dir
            /
            RESULT_FILENAME
        ),
        metrics_path=(
            experiment_dir
            /
            METRICS_FILENAME
        ),
        training_history_path=(
            experiment_dir
            /
            TRAINING_HISTORY_FILENAME
        ),
    )

    result_payload = _load_json_file(
        artifacts.result_path,
        description=(
            "experiment result JSON"
        ),
    )

    if not isinstance(
        result_payload,
        Mapping,
    ):
        raise TypeError(
            "Stored result.json root must be a mapping"
        )

    stored_result = (
        result_payload.get(
            "result"
        )
    )

    result = (
        _experiment_result_from_payload(
            stored_result
        )
    )

    stored_hyperparameters = (
        result.metadata.get(
            "hyperparameters",
            {},
        )
    )

    if not isinstance(
        stored_hyperparameters,
        Mapping,
    ):
        raise TypeError(
            "Stored result metadata.hyperparameters must be a mapping"
        )

    if (
        result.model_name
        !=
        spec.model_name
        or result.family
        !=
        spec.family
        or result.variant
        !=
        spec.variant
        or result.seed
        !=
        spec.seed
        or result.target_split
        !=
        target_split
        or _canonical_json_bytes(
            dict(
                stored_hyperparameters
            )
        )
        !=
        _canonical_json_bytes(
            dict(
                spec.hyperparameters
            )
        )
    ):
        raise ValueError(
            "Stored experiment scientific identity does not match "
            "the requested ExperimentSpec"
        )

    if (
        experiment_id_for_result(
            result
        )
        !=
        experiment_id
    ):
        raise ValueError(
            "Stored experiment ID does not match its result identity"
        )

    expected_result_payload = (
        _result_payload(
            run=run,
            experiment_id=(
                experiment_id
            ),
            result=result,
        )
    )

    if (
        _canonical_json_bytes(
            result_payload
        )
        !=
        _canonical_json_bytes(
            expected_result_payload
        )
    ):
        raise ValueError(
            "Stored result.json provenance or payload does not match "
            "the current run"
        )

    history_payload = _load_json_file(
        artifacts.training_history_path,
        description=(
            "experiment training-history JSON"
        ),
    )

    expected_history_payload = (
        _training_history_payload(
            run=run,
            experiment_id=(
                experiment_id
            ),
            result=result,
        )
    )

    if (
        _canonical_json_bytes(
            history_payload
        )
        !=
        _canonical_json_bytes(
            expected_history_payload
        )
    ):
        raise ValueError(
            "Stored training_history.json does not match result.json"
        )

    _assert_metrics_csv_matches(
        path=artifacts.metrics_path,
        expected_rows=(
            _metric_rows(
                run=run,
                experiment_id=(
                    experiment_id
                ),
                result=result,
            )
        ),
    )

    return (
        result,
        artifacts,
    )


# ============================================================
# Public experiment write API
# ============================================================


def write_experiment_result(
    *,
    run: RunContext,
    result: ExperimentResult,
) -> ExperimentArtifacts:
    """Persist one experiment without allowing overwrite/mixing."""

    if not isinstance(
        run,
        RunContext,
    ):
        raise TypeError(
            "run must be a RunContext"
        )

    if not isinstance(
        result,
        ExperimentResult,
    ):
        raise TypeError(
            "result must be an ExperimentResult"
        )

    _assert_run_context_matches_disk(
        run
    )

    experiment_id = (
        experiment_id_for_result(
            result
        )
    )

    final_directory = (
        run.experiments_dir
        /
        experiment_id
    )

    if final_directory.exists():
        raise FileExistsError(
            "Experiment directory already exists: "
            f"{final_directory}"
        )

    staging_directory = (
        run.experiments_dir
        /
        (
            "."
            f"{experiment_id}"
            ".tmp-"
            f"{uuid4().hex}"
        )
    )

    if staging_directory.exists():
        raise FileExistsError(
            "Experiment staging directory "
            "already exists"
        )

    try:

        staging_directory.mkdir(
            parents=False,
            exist_ok=False,
        )

        result_path = (
            staging_directory
            /
            RESULT_FILENAME
        )

        metrics_path = (
            staging_directory
            /
            METRICS_FILENAME
        )

        training_history_path = (
            staging_directory
            /
            TRAINING_HISTORY_FILENAME
        )

        _write_json(
            result_path,
            _result_payload(
                run=run,
                experiment_id=(
                    experiment_id
                ),
                result=result,
            ),
        )

        _write_metrics_csv(
            metrics_path,
            _metric_rows(
                run=run,
                experiment_id=(
                    experiment_id
                ),
                result=result,
            ),
        )

        _write_json(
            training_history_path,
            _training_history_payload(
                run=run,
                experiment_id=(
                    experiment_id
                ),
                result=result,
            ),
        )

        staging_directory.replace(
            final_directory
        )

    except Exception:

        if staging_directory.exists():
            shutil.rmtree(
                staging_directory
            )

        raise

    return ExperimentArtifacts(
        experiment_id=(
            experiment_id
        ),
        experiment_dir=(
            final_directory
        ),
        result_path=(
            final_directory
            /
            RESULT_FILENAME
        ),
        metrics_path=(
            final_directory
            /
            METRICS_FILENAME
        ),
        training_history_path=(
            final_directory
            /
            TRAINING_HISTORY_FILENAME
        ),
    )