"""Permanent tests for Paper 2 structured result storage."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.experiment import ExperimentResult
from src.results import (
    METRICS_COLUMNS,
    RESULT_SCHEMA_VERSION,
    RunProvenance,
    build_run_provenance,
    create_run,
    experiment_id_for_result,
    generate_run_id,
    load_run_context,
    semantic_yaml_hash,
    sha256_file,
    source_tree_hash,
    write_experiment_result,
)


# ============================================================
# Helpers
# ============================================================


def _hash(
    character: str,
) -> str:
    """Create one deterministic valid SHA-256-looking digest."""

    return (
        character
        *
        64
    )


def _provenance() -> RunProvenance:
    """Return deterministic synthetic run provenance."""

    return RunProvenance(
        dataset_id="synthetic-dataset",
        artifact_version="v1",
        config_hash=_hash("a"),
        dataset_hash=_hash("b"),
        split_hash=_hash("c"),
        code_version=_hash("d"),
    )


def _result(
    *,
    model_name: str = "SyntheticModel",
    family: str = "synthetic",
    variant: str = "primary",
    seed: int = 2026,
    target_split: str = "validation",
    ranking_metrics=None,
    native_metrics=None,
    training_history=None,
    hyperparameters=None,
) -> ExperimentResult:
    """Construct one valid structured experiment result."""

    if ranking_metrics is None:
        ranking_metrics = {
            "ndcg_at_10": 0.42,
            "recall_at_10": 0.50,
        }

    if native_metrics is None:
        native_metrics = {
            "rmse": 0.80,
        }

    if training_history is None:
        training_history = [
            {
                "epoch": 1,
                "loss": 0.25,
            }
        ]

    if hyperparameters is None:
        hyperparameters = {
            "n_factors": 4,
            "learning_rate": 0.01,
        }

    return ExperimentResult(
        model_name=model_name,
        family=family,
        variant=variant,
        seed=seed,
        target_split=target_split,
        ranking_metrics=ranking_metrics,
        native_metrics=native_metrics,
        training_history=training_history,
        runtime_seconds=1.25,
        status="completed",
        error_message=None,
        metadata={
            "hyperparameters": hyperparameters,
        },
    )


def _write_yaml(
    path: Path,
    payload,
) -> None:
    """Write deterministic test YAML."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _build_synthetic_artifact_tree(
    root: Path,
):
    """Create a complete synthetic Paper 2 artifact tree."""

    # --------------------------------------------------------
    # Production source tree required by source_tree_hash()
    # --------------------------------------------------------

    src_dir = (
        root
        /
        "src"
    )

    scripts_dir = (
        root
        /
        "scripts"
    )

    src_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scripts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        src_dir
        /
        "alpha.py"
    ).write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    (
        scripts_dir
        /
        "beta.py"
    ).write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = {
        "dataset": {
            "name": "synthetic",
        },
        "evaluation": {
            "k_values": [
                5,
                10,
                20,
            ],
        },
    }

    config_path = (
        root
        /
        "config"
        /
        "dev.yaml"
    )

    _write_yaml(
        config_path,
        config,
    )

    config_hash = (
        semantic_yaml_hash(
            config
        )
    )

    # --------------------------------------------------------
    # Data files
    # --------------------------------------------------------

    ratings_path = (
        root
        /
        "data"
        /
        "processed"
        /
        "synthetic"
        /
        "v1"
        /
        "ratings.csv"
    )

    train_path = (
        root
        /
        "data"
        /
        "splits"
        /
        "synthetic"
        /
        "v1"
        /
        "train.csv"
    )

    validation_path = (
        root
        /
        "data"
        /
        "splits"
        /
        "synthetic"
        /
        "v1"
        /
        "validation.csv"
    )

    test_path = (
        root
        /
        "data"
        /
        "splits"
        /
        "synthetic"
        /
        "v1"
        /
        "test.csv"
    )

    for path, content in [
        (
            ratings_path,
            "userId,movieId,rating\n1,10,5.0\n",
        ),
        (
            train_path,
            "userId,movieId,rating\n1,10,5.0\n",
        ),
        (
            validation_path,
            "userId,movieId,rating\n1,20,4.0\n",
        ),
        (
            test_path,
            "userId,movieId,rating\n1,30,3.0\n",
        ),
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

    dataset_hash = (
        sha256_file(
            ratings_path
        )
    )

    train_hash = (
        sha256_file(
            train_path
        )
    )

    validation_hash = (
        sha256_file(
            validation_path
        )
    )

    test_hash = (
        sha256_file(
            test_path
        )
    )

    split_hash = _hash(
        "e"
    )

    # --------------------------------------------------------
    # Preparation manifest
    # --------------------------------------------------------

    preparation_manifest = {
        "artifact": {
            "type": "processed_dataset",
            "version": "v1",
            "dataset_id": "synthetic",
        },
        "source": {
            "config_sha256": config_hash,
        },
        "files": {
            "ratings": {
                "path": (
                    "data\\processed\\synthetic\\v1\\ratings.csv"
                ),
                "sha256": dataset_hash,
            },
        },
    }

    preparation_manifest_path = (
        root
        /
        "data"
        /
        "processed"
        /
        "synthetic"
        /
        "v1"
        /
        "preparation_manifest.yaml"
    )

    _write_yaml(
        preparation_manifest_path,
        preparation_manifest,
    )

    # --------------------------------------------------------
    # Split manifest
    # --------------------------------------------------------

    split_manifest = {
        "artifact": {
            "type": "canonical_split",
            "version": "v1",
            "dataset_id": "synthetic",
        },
        "source": {
            "processed_dataset_sha256": dataset_hash,
            "config_sha256": config_hash,
        },
        "files": {
            "train": {
                "path": (
                    "data\\splits\\synthetic\\v1\\train.csv"
                ),
                "sha256": train_hash,
            },
            "validation": {
                "path": (
                    "data\\splits\\synthetic\\v1\\validation.csv"
                ),
                "sha256": validation_hash,
            },
            "test": {
                "path": (
                    "data\\splits\\synthetic\\v1\\test.csv"
                ),
                "sha256": test_hash,
            },
        },
        "split_sha256": split_hash,
    }

    split_manifest_path = (
        root
        /
        "data"
        /
        "splits"
        /
        "synthetic"
        /
        "v1"
        /
        "split_manifest.yaml"
    )

    _write_yaml(
        split_manifest_path,
        split_manifest,
    )

    return {
        "config": config,
        "config_path": config_path,
        "config_hash": config_hash,
        "ratings_path": ratings_path,
        "dataset_hash": dataset_hash,
        "train_path": train_path,
        "validation_path": validation_path,
        "test_path": test_path,
        "preparation_manifest_path": (
            preparation_manifest_path
        ),
        "split_manifest_path": (
            split_manifest_path
        ),
        "split_hash": split_hash,
    }


# ============================================================
# Constants / basic provenance validation
# ============================================================


def test_result_schema_and_metrics_columns_are_frozen():
    assert (
        RESULT_SCHEMA_VERSION
        ==
        "1.0"
    )

    assert METRICS_COLUMNS == [
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


@pytest.mark.parametrize(
    "field_name",
    [
        "config_hash",
        "dataset_hash",
        "split_hash",
        "code_version",
    ],
)
def test_run_provenance_rejects_invalid_hash(
    field_name,
):
    kwargs = {
        "dataset_id": "synthetic",
        "artifact_version": "v1",
        "config_hash": _hash("a"),
        "dataset_hash": _hash("b"),
        "split_hash": _hash("c"),
        "code_version": _hash("d"),
    }

    kwargs[
        field_name
    ] = "not-a-valid-hash"

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        RunProvenance(
            **kwargs
        )


def test_run_provenance_rejects_empty_dataset_id():
    with pytest.raises(
        ValueError,
        match="dataset_id",
    ):
        RunProvenance(
            dataset_id="",
            artifact_version="v1",
            config_hash=_hash("a"),
            dataset_hash=_hash("b"),
            split_hash=_hash("c"),
            code_version=_hash("d"),
        )


# ============================================================
# Hashing
# ============================================================


def test_semantic_yaml_hash_ignores_mapping_order():
    first = {
        "b": 2,
        "a": {
            "y": 2,
            "x": 1,
        },
    }

    second = {
        "a": {
            "x": 1,
            "y": 2,
        },
        "b": 2,
    }

    assert (
        semantic_yaml_hash(
            first
        )
        ==
        semantic_yaml_hash(
            second
        )
    )


def test_sha256_file_changes_when_file_content_changes(
    tmp_path,
):
    path = (
        tmp_path
        /
        "value.txt"
    )

    path.write_text(
        "alpha",
        encoding="utf-8",
    )

    first = (
        sha256_file(
            path
        )
    )

    path.write_text(
        "beta",
        encoding="utf-8",
    )

    second = (
        sha256_file(
            path
        )
    )

    assert (
        first
        !=
        second
    )


def test_source_tree_hash_is_deterministic(
    tmp_path,
):
    tree = (
        _build_synthetic_artifact_tree(
            tmp_path
        )
    )

    assert tree is not None

    first = (
        source_tree_hash(
            tmp_path
        )
    )

    second = (
        source_tree_hash(
            tmp_path
        )
    )

    assert first == second

    assert len(
        first
    ) == 64


def test_source_tree_hash_changes_with_production_source(
    tmp_path,
):
    _build_synthetic_artifact_tree(
        tmp_path
    )

    before = (
        source_tree_hash(
            tmp_path
        )
    )

    (
        tmp_path
        /
        "src"
        /
        "alpha.py"
    ).write_text(
        "VALUE = 999\n",
        encoding="utf-8",
    )

    after = (
        source_tree_hash(
            tmp_path
        )
    )

    assert (
        before
        !=
        after
    )


def test_source_tree_hash_ignores_tests_and_results(
    tmp_path,
):
    _build_synthetic_artifact_tree(
        tmp_path
    )

    before = (
        source_tree_hash(
            tmp_path
        )
    )

    tests_dir = (
        tmp_path
        /
        "tests"
    )

    results_dir = (
        tmp_path
        /
        "results"
    )

    tests_dir.mkdir()
    results_dir.mkdir()

    (
        tests_dir
        /
        "test_something.py"
    ).write_text(
        "assert False\n",
        encoding="utf-8",
    )

    (
        results_dir
        /
        "junk.py"
    ).write_text(
        "VALUE = 12345\n",
        encoding="utf-8",
    )

    after = (
        source_tree_hash(
            tmp_path
        )
    )

    assert (
        before
        ==
        after
    )


# ============================================================
# Provenance construction
# ============================================================


def test_build_run_provenance_from_valid_artifacts(
    tmp_path,
):
    tree = (
        _build_synthetic_artifact_tree(
            tmp_path
        )
    )

    provenance = (
        build_run_provenance(
            config_path=(
                tree[
                    "config_path"
                ]
            ),
            preparation_manifest_path=(
                tree[
                    "preparation_manifest_path"
                ]
            ),
            split_manifest_path=(
                tree[
                    "split_manifest_path"
                ]
            ),
            project_root=tmp_path,
            verify_files=True,
        )
    )

    assert (
        provenance.dataset_id
        ==
        "synthetic"
    )

    assert (
        provenance.artifact_version
        ==
        "v1"
    )

    assert (
        provenance.config_hash
        ==
        tree[
            "config_hash"
        ]
    )

    assert (
        provenance.dataset_hash
        ==
        tree[
            "dataset_hash"
        ]
    )

    assert (
        provenance.split_hash
        ==
        tree[
            "split_hash"
        ]
    )

    assert (
        provenance.code_version
        ==
        source_tree_hash(
            tmp_path
        )
    )


def test_build_run_provenance_rejects_config_mismatch(
    tmp_path,
):
    tree = (
        _build_synthetic_artifact_tree(
            tmp_path
        )
    )

    changed_config = {
        "dataset": {
            "name": "changed",
        },
    }

    _write_yaml(
        tree[
            "config_path"
        ],
        changed_config,
    )

    with pytest.raises(
        ValueError,
        match="configuration",
    ):
        build_run_provenance(
            config_path=(
                tree[
                    "config_path"
                ]
            ),
            preparation_manifest_path=(
                tree[
                    "preparation_manifest_path"
                ]
            ),
            split_manifest_path=(
                tree[
                    "split_manifest_path"
                ]
            ),
            project_root=tmp_path,
            verify_files=True,
        )


def test_build_run_provenance_rejects_dataset_manifest_mismatch(
    tmp_path,
):
    tree = (
        _build_synthetic_artifact_tree(
            tmp_path
        )
    )

    split_manifest = (
        yaml.safe_load(
            tree[
                "split_manifest_path"
            ].read_text(
                encoding="utf-8"
            )
        )
    )

    split_manifest[
        "source"
    ][
        "processed_dataset_sha256"
    ] = _hash(
        "f"
    )

    _write_yaml(
        tree[
            "split_manifest_path"
        ],
        split_manifest,
    )

    with pytest.raises(
        ValueError,
        match="dataset hash",
    ):
        build_run_provenance(
            config_path=(
                tree[
                    "config_path"
                ]
            ),
            preparation_manifest_path=(
                tree[
                    "preparation_manifest_path"
                ]
            ),
            split_manifest_path=(
                tree[
                    "split_manifest_path"
                ]
            ),
            project_root=tmp_path,
            verify_files=True,
        )


def test_build_run_provenance_detects_tampered_processed_file(
    tmp_path,
):
    tree = (
        _build_synthetic_artifact_tree(
            tmp_path
        )
    )

    tree[
        "ratings_path"
    ].write_text(
        "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Processed ratings file SHA-256",
    ):
        build_run_provenance(
            config_path=(
                tree[
                    "config_path"
                ]
            ),
            preparation_manifest_path=(
                tree[
                    "preparation_manifest_path"
                ]
            ),
            split_manifest_path=(
                tree[
                    "split_manifest_path"
                ]
            ),
            project_root=tmp_path,
            verify_files=True,
        )


@pytest.mark.parametrize(
    "split_name",
    [
        "train",
        "validation",
        "test",
    ],
)
def test_build_run_provenance_detects_tampered_split_file(
    tmp_path,
    split_name,
):
    tree = (
        _build_synthetic_artifact_tree(
            tmp_path
        )
    )

    tree[
        f"{split_name}_path"
    ].write_text(
        "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            f"{split_name} split file SHA-256"
        ),
    ):
        build_run_provenance(
            config_path=(
                tree[
                    "config_path"
                ]
            ),
            preparation_manifest_path=(
                tree[
                    "preparation_manifest_path"
                ]
            ),
            split_manifest_path=(
                tree[
                    "split_manifest_path"
                ]
            ),
            project_root=tmp_path,
            verify_files=True,
        )


def test_build_run_provenance_verify_files_false_skips_file_hash_check(
    tmp_path,
):
    tree = (
        _build_synthetic_artifact_tree(
            tmp_path
        )
    )

    tree[
        "ratings_path"
    ].write_text(
        "tampered\n",
        encoding="utf-8",
    )

    provenance = (
        build_run_provenance(
            config_path=(
                tree[
                    "config_path"
                ]
            ),
            preparation_manifest_path=(
                tree[
                    "preparation_manifest_path"
                ]
            ),
            split_manifest_path=(
                tree[
                    "split_manifest_path"
                ]
            ),
            project_root=tmp_path,
            verify_files=False,
        )
    )

    assert (
        provenance.dataset_hash
        ==
        tree[
            "dataset_hash"
        ]
    )


# ============================================================
# Run creation / loading
# ============================================================


def test_generate_run_id_is_valid_and_unique():
    provenance = (
        _provenance()
    )

    first = (
        generate_run_id(
            provenance
        )
    )

    second = (
        generate_run_id(
            provenance
        )
    )

    assert (
        first
        !=
        second
    )

    assert first.startswith(
        "run-"
    )

    assert (
        ".."
        not in first
    )


def test_create_run_writes_exact_manifest(
    tmp_path,
):
    provenance = (
        _provenance()
    )

    run = create_run(
        provenance=provenance,
        output_root=tmp_path,
        run_id="storage-test-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    assert (
        run.run_dir
        ==
        tmp_path
        /
        "storage-test-run"
    )

    assert (
        run.experiments_dir.is_dir()
    )

    payload = json.loads(
        run.manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert set(
        payload.keys()
    ) == {
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

    assert (
        payload[
            "schema_version"
        ]
        ==
        "1.0"
    )

    assert (
        payload[
            "run_id"
        ]
        ==
        "storage-test-run"
    )

    assert (
        payload[
            "dataset_hash"
        ]
        ==
        provenance.dataset_hash
    )


def test_create_run_refuses_existing_directory(
    tmp_path,
):
    provenance = (
        _provenance()
    )

    kwargs = {
        "provenance": provenance,
        "output_root": tmp_path,
        "run_id": "duplicate-run",
        "created_at_utc": (
            "2026-10-06T20:00:00.000000Z"
        ),
    }

    create_run(
        **kwargs
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        create_run(
            **kwargs
        )


def test_load_run_context_round_trip(
    tmp_path,
):
    provenance = (
        _provenance()
    )

    created = create_run(
        provenance=provenance,
        output_root=tmp_path,
        run_id="roundtrip-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    loaded = (
        load_run_context(
            created.run_dir
        )
    )

    assert (
        loaded.run_id
        ==
        created.run_id
    )

    assert (
        loaded.provenance
        ==
        provenance
    )

    assert (
        loaded.created_at_utc
        ==
        created.created_at_utc
    )


def test_load_run_context_rejects_directory_name_mismatch(
    tmp_path,
):
    provenance = (
        _provenance()
    )

    run = create_run(
        provenance=provenance,
        output_root=tmp_path,
        run_id="original-name",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    renamed = (
        tmp_path
        /
        "different-name"
    )

    run.run_dir.rename(
        renamed
    )

    with pytest.raises(
        ValueError,
        match="directory name",
    ):
        load_run_context(
            renamed
        )


# ============================================================
# Experiment IDs
# ============================================================


def test_experiment_id_is_deterministic():
    first = (
        experiment_id_for_result(
            _result()
        )
    )

    second = (
        experiment_id_for_result(
            _result()
        )
    )

    assert (
        first
        ==
        second
    )


def test_experiment_id_changes_with_hyperparameters():
    first = (
        experiment_id_for_result(
            _result(
                hyperparameters={
                    "n_factors": 4,
                }
            )
        )
    )

    second = (
        experiment_id_for_result(
            _result(
                hyperparameters={
                    "n_factors": 8,
                }
            )
        )
    )

    assert (
        first
        !=
        second
    )


def test_experiment_id_changes_with_seed():
    first = (
        experiment_id_for_result(
            _result(
                seed=2026
            )
        )
    )

    second = (
        experiment_id_for_result(
            _result(
                seed=2027
            )
        )
    )

    assert (
        first
        !=
        second
    )


# ============================================================
# Experiment persistence
# ============================================================


def test_write_experiment_result_creates_all_artifacts(
    tmp_path,
):
    run = create_run(
        provenance=_provenance(),
        output_root=tmp_path,
        run_id="artifact-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    result = (
        _result()
    )

    artifacts = (
        write_experiment_result(
            run=run,
            result=result,
        )
    )

    assert (
        artifacts.experiment_dir.is_dir()
    )

    assert (
        artifacts.result_path.is_file()
    )

    assert (
        artifacts.metrics_path.is_file()
    )

    assert (
        artifacts.training_history_path.is_file()
    )


def test_result_json_contains_run_provenance_and_result(
    tmp_path,
):
    provenance = (
        _provenance()
    )

    run = create_run(
        provenance=provenance,
        output_root=tmp_path,
        run_id="result-json-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    result = (
        _result()
    )

    artifacts = (
        write_experiment_result(
            run=run,
            result=result,
        )
    )

    payload = json.loads(
        artifacts.result_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload[
            "schema_version"
        ]
        ==
        RESULT_SCHEMA_VERSION
    )

    assert (
        payload[
            "run_id"
        ]
        ==
        run.run_id
    )

    assert (
        payload[
            "experiment_id"
        ]
        ==
        artifacts.experiment_id
    )

    assert (
        payload[
            "config_hash"
        ]
        ==
        provenance.config_hash
    )

    assert (
        payload[
            "dataset_hash"
        ]
        ==
        provenance.dataset_hash
    )

    assert (
        payload[
            "split_hash"
        ]
        ==
        provenance.split_hash
    )

    assert (
        payload[
            "code_version"
        ]
        ==
        provenance.code_version
    )

    assert (
        payload[
            "result"
        ][
            "model_name"
        ]
        ==
        result.model_name
    )


def test_metrics_csv_has_strict_long_schema_and_sorted_names(
    tmp_path,
):
    run = create_run(
        provenance=_provenance(),
        output_root=tmp_path,
        run_id="metrics-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    result = _result(
        ranking_metrics={
            "recall_at_10": 0.50,
            "ndcg_at_10": 0.42,
        },
        native_metrics={
            "rmse": 0.80,
            "mae": 0.60,
        },
    )

    artifacts = (
        write_experiment_result(
            run=run,
            result=result,
        )
    )

    with artifacts.metrics_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = list(
            reader
        )

        assert (
            reader.fieldnames
            ==
            METRICS_COLUMNS
        )

    assert len(
        rows
    ) == 4

    ranking_names = [
        row[
            "metric_name"
        ]
        for row in rows
        if (
            row[
                "metric_scope"
            ]
            ==
            "ranking"
        )
    ]

    native_names = [
        row[
            "metric_name"
        ]
        for row in rows
        if (
            row[
                "metric_scope"
            ]
            ==
            "native"
        )
    ]

    assert ranking_names == [
        "ndcg_at_10",
        "recall_at_10",
    ]

    assert native_names == [
        "mae",
        "rmse",
    ]

    assert all(
        row[
            "run_id"
        ]
        ==
        run.run_id
        for row in rows
    )


def test_metrics_csv_handles_empty_native_metrics(
    tmp_path,
):
    run = create_run(
        provenance=_provenance(),
        output_root=tmp_path,
        run_id="ranking-only-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    result = _result(
        native_metrics={},
    )

    artifacts = (
        write_experiment_result(
            run=run,
            result=result,
        )
    )

    with artifacts.metrics_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    assert len(
        rows
    ) == 2

    assert {
        row[
            "metric_scope"
        ]
        for row in rows
    } == {
        "ranking",
    }


def test_training_history_json_matches_result(
    tmp_path,
):
    run = create_run(
        provenance=_provenance(),
        output_root=tmp_path,
        run_id="history-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    history = [
        {
            "epoch": 1,
            "loss": 0.50,
        },
        {
            "epoch": 2,
            "loss": 0.25,
        },
    ]

    result = _result(
        training_history=history,
    )

    artifacts = (
        write_experiment_result(
            run=run,
            result=result,
        )
    )

    payload = json.loads(
        artifacts.training_history_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload[
            "schema_version"
        ]
        ==
        RESULT_SCHEMA_VERSION
    )

    assert (
        payload[
            "run_id"
        ]
        ==
        run.run_id
    )

    assert (
        payload[
            "experiment_id"
        ]
        ==
        artifacts.experiment_id
    )

    assert (
        payload[
            "records"
        ]
        ==
        history
    )


def test_duplicate_experiment_write_is_blocked(
    tmp_path,
):
    run = create_run(
        provenance=_provenance(),
        output_root=tmp_path,
        run_id="duplicate-experiment-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    result = (
        _result()
    )

    write_experiment_result(
        run=run,
        result=result,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        write_experiment_result(
            run=run,
            result=result,
        )


def test_manifest_tampering_blocks_future_experiment_write(
    tmp_path,
):
    run = create_run(
        provenance=_provenance(),
        output_root=tmp_path,
        run_id="tamper-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    payload = json.loads(
        run.manifest_path.read_text(
            encoding="utf-8"
        )
    )

    payload[
        "dataset_hash"
    ] = _hash(
        "f"
    )

    run.manifest_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="provenance",
    ):
        write_experiment_result(
            run=run,
            result=_result(),
        )


def test_nonfinite_training_history_fails_without_partial_experiment(
    tmp_path,
):
    run = create_run(
        provenance=_provenance(),
        output_root=tmp_path,
        run_id="nonfinite-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    result = _result(
        training_history=[
            {
                "epoch": 1,
                "loss": np.nan,
            }
        ]
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

    with pytest.raises(
        ValueError,
        match=(
            "NaN or infinite"
        ),
    ):
        write_experiment_result(
            run=run,
            result=result,
        )

    assert (
        not final_directory.exists()
    )

    leftovers = [
        path
        for path
        in run.experiments_dir.iterdir()
        if (
            experiment_id
            in path.name
        )
    ]

    assert leftovers == []


def test_two_distinct_experiments_can_coexist_in_one_run(
    tmp_path,
):
    run = create_run(
        provenance=_provenance(),
        output_root=tmp_path,
        run_id="multi-experiment-run",
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    first = (
        write_experiment_result(
            run=run,
            result=_result(
                model_name="ModelA",
                hyperparameters={
                    "k": 4,
                },
            ),
        )
    )

    second = (
        write_experiment_result(
            run=run,
            result=_result(
                model_name="ModelB",
                hyperparameters={
                    "k": 8,
                },
            ),
        )
    )

    assert (
        first.experiment_id
        !=
        second.experiment_id
    )

    assert (
        first.experiment_dir.is_dir()
    )

    assert (
        second.experiment_dir.is_dir()
    )

    stored = sorted(
        path.name
        for path
        in run.experiments_dir.iterdir()
        if path.is_dir()
    )

    assert stored == sorted(
        [
            first.experiment_id,
            second.experiment_id,
        ]
    )