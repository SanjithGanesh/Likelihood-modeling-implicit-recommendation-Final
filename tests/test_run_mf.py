"""Permanent regression tests for scripts/run_mf.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import scripts.run_mf as run_mf_module

from scripts.run_mf import (
    MODEL_CHOICES,
    MODEL_METADATA,
    MODEL_ORDER,
    artifact_paths,
    build_experiment_spec,
    build_model,
    load_frozen_split,
    prepare_run_context,
    resolve_model_keys,
    resolve_seed,
    run_selected_models,
)

from src.experiment import (
    ExperimentResult,
    ExperimentSpec,
)

from src.results import (
    RunContext,
    RunProvenance,
)

from src.split import (
    SplitResult,
)


# ============================================================
# Helpers
# ============================================================


def _synthetic_split() -> SplitResult:
    """Return a tiny canonical-style split."""

    train = pd.DataFrame(
        {
            "userId": [
                1,
                1,
                2,
                2,
            ],
            "movieId": [
                10,
                20,
                10,
                30,
            ],
            "rating": [
                5,
                4,
                4,
                3,
            ],
            "timestamp": [
                1,
                2,
                3,
                4,
            ],
            "user_idx": [
                0,
                0,
                1,
                1,
            ],
            "item_idx": [
                0,
                1,
                0,
                2,
            ],
        }
    )

    validation = pd.DataFrame(
        {
            "userId": [
                1,
                2,
            ],
            "movieId": [
                30,
                20,
            ],
            "rating": [
                5,
                4,
            ],
            "timestamp": [
                5,
                6,
            ],
            "user_idx": [
                0,
                1,
            ],
            "item_idx": [
                2,
                1,
            ],
        }
    )

    test = pd.DataFrame(
        {
            "userId": [
                1,
                2,
            ],
            "movieId": [
                40,
                40,
            ],
            "rating": [
                4,
                5,
            ],
            "timestamp": [
                7,
                8,
            ],
            "user_idx": [
                0,
                1,
            ],
            "item_idx": [
                3,
                3,
            ],
        }
    )

    return SplitResult(
        train=train,
        validation=validation,
        test=test,
        seed=2026,
        attempt=1,
        enforce_train_item_coverage=True,
    )


def _config() -> dict:
    """Return runner-compatible synthetic configuration."""

    return {
        "dataset": {
            "id": "synthetic",
            "processed_root": "data/processed",
            "split_root": "data/splits",
        },
        "training": {
            "batch_size": 1024,
            "default_epochs": 5,
            "default_learning_rate": 0.01,
        },
        "runtime": {
            "device": "cpu",
        },
        "reproducibility": {
            "global_seed": 2026,
            "seeds": [
                2026,
            ],
        },
        "evaluation": {
            "k_values": [
                5,
                10,
                20,
            ],
            "primary_selection_metric": (
                "ndcg_at_10"
            ),
        },
        "paths": {
            "results_root": "results",
        },
    }


def _provenance() -> RunProvenance:
    """Return valid synthetic run provenance."""

    return RunProvenance(
        dataset_id="synthetic",
        artifact_version="v1",
        config_hash="a" * 64,
        dataset_hash="b" * 64,
        split_hash="c" * 64,
        code_version="d" * 64,
    )


def _run_context(
    root: Path,
    *,
    run_id: str = "synthetic-run",
) -> RunContext:
    """Return a lightweight RunContext."""

    return RunContext(
        run_id=run_id,
        run_dir=(
            root
            /
            run_id
        ),
        provenance=_provenance(),
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )


def _result(
    *,
    model_name: str,
    family: str,
    seed: int = 2026,
    target_split: str = "validation",
) -> ExperimentResult:
    """Return one valid synthetic experiment result."""

    return ExperimentResult(
        model_name=model_name,
        family=family,
        variant="primary",
        seed=seed,
        target_split=target_split,
        ranking_metrics={
            "ndcg_at_10": 0.5,
        },
        native_metrics={},
        training_history=[],
        runtime_seconds=0.1,
        status="completed",
        error_message=None,
        metadata={
            "hyperparameters": {},
        },
    )


# ============================================================
# Frozen runner inventory
# ============================================================


def test_model_inventory_is_frozen():
    assert MODEL_ORDER == [
        "pca_svd",
        "bernoulli_mf",
        "poisson_mf",
        "gaussian_mf",
        "logistic_pca",
        "hpf",
    ]

    assert MODEL_CHOICES == [
        "pca_svd",
        "bernoulli_mf",
        "poisson_mf",
        "gaussian_mf",
        "logistic_pca",
        "hpf",
        "all",
    ]


def test_model_metadata_matches_public_model_names():
    assert (
        MODEL_METADATA[
            "pca_svd"
        ][
            "model_name"
        ]
        ==
        "PCA/SVD"
    )

    assert (
        MODEL_METADATA[
            "bernoulli_mf"
        ][
            "model_name"
        ]
        ==
        "BernoulliMF"
    )

    assert (
        MODEL_METADATA[
            "poisson_mf"
        ][
            "model_name"
        ]
        ==
        "PoissonMF"
    )

    assert (
        MODEL_METADATA[
            "gaussian_mf"
        ][
            "model_name"
        ]
        ==
        "GaussianMF"
    )

    assert (
        MODEL_METADATA[
            "logistic_pca"
        ][
            "model_name"
        ]
        ==
        "LogisticPCA"
    )

    assert (
        MODEL_METADATA[
            "hpf"
        ][
            "model_name"
        ]
        ==
        "HPF"
    )


# ============================================================
# Model-selection / seed helpers
# ============================================================


@pytest.mark.parametrize(
    "model_key",
    MODEL_ORDER,
)
def test_resolve_model_keys_single_model(
    model_key,
):
    assert (
        resolve_model_keys(
            model_key
        )
        ==
        [
            model_key,
        ]
    )


def test_resolve_model_keys_all_preserves_frozen_order():
    assert (
        resolve_model_keys(
            "all"
        )
        ==
        MODEL_ORDER
    )


def test_resolve_model_keys_rejects_unknown():
    with pytest.raises(
        ValueError,
        match="Unknown model selection",
    ):
        resolve_model_keys(
            "not-a-model"
        )


def test_resolve_seed_uses_global_seed_by_default():
    assert (
        resolve_seed(
            _config(),
            None,
        )
        ==
        2026
    )


def test_resolve_seed_allows_explicit_override():
    assert (
        resolve_seed(
            _config(),
            2030,
        )
        ==
        2030
    )


def test_resolve_seed_rejects_boolean():
    with pytest.raises(
        ValueError,
        match="integer",
    ):
        resolve_seed(
            _config(),
            True,
        )


# ============================================================
# Artifact paths
# ============================================================


def test_artifact_paths_follow_dataset_and_version(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        run_mf_module,
        "PROJECT_ROOT",
        tmp_path,
    )

    paths = artifact_paths(
        _config(),
        version="v7",
    )

    assert (
        paths[
            "processed_dir"
        ]
        ==
        tmp_path
        /
        "data"
        /
        "processed"
        /
        "synthetic"
        /
        "v7"
    )

    assert (
        paths[
            "split_dir"
        ]
        ==
        tmp_path
        /
        "data"
        /
        "splits"
        /
        "synthetic"
        /
        "v7"
    )

    assert (
        paths[
            "preparation_manifest"
        ].name
        ==
        "preparation_manifest.yaml"
    )

    assert (
        paths[
            "split_manifest"
        ].name
        ==
        "split_manifest.yaml"
    )


def test_artifact_paths_rejects_empty_version():
    with pytest.raises(
        ValueError,
        match="version",
    ):
        artifact_paths(
            _config(),
            version="",
        )


# ============================================================
# Frozen split loading
# ============================================================


def test_load_frozen_split_reconstructs_manifest_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        run_mf_module,
        "PROJECT_ROOT",
        tmp_path,
    )

    split_dir = (
        tmp_path
        /
        "data"
        /
        "splits"
        /
        "synthetic"
        /
        "v1"
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split = _synthetic_split()

    for name in [
        "train",
        "validation",
        "test",
    ]:
        frame = getattr(
            split,
            name,
        )

        frame.to_csv(
            split_dir
            /
            f"{name}.csv",
            index=False,
        )

    manifest = {
        "artifact": {
            "type": "canonical_split",
            "version": "v1",
            "dataset_id": "synthetic",
        },
        "protocol": {
            "seed": 2026,
            "constrained_attempt": 1,
            "enforce_train_item_coverage": True,
        },
        "statistics": {
            "train_rows": len(
                split.train
            ),
            "validation_rows": len(
                split.validation
            ),
            "test_rows": len(
                split.test
            ),
        },
        "files": {
            "train": {
                "path": (
                    "data/splits/synthetic/"
                    "v1/train.csv"
                ),
            },
            "validation": {
                "path": (
                    "data/splits/synthetic/"
                    "v1/validation.csv"
                ),
            },
            "test": {
                "path": (
                    "data/splits/synthetic/"
                    "v1/test.csv"
                ),
            },
        },
    }

    (
        split_dir
        /
        "split_manifest.yaml"
    ).write_text(
        yaml.safe_dump(
            manifest,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_frozen_split(
        _config(),
        version="v1",
    )

    assert (
        loaded.seed
        ==
        2026
    )

    assert (
        loaded.attempt
        ==
        1
    )

    assert (
        loaded.enforce_train_item_coverage
        is True
    )

    pd.testing.assert_frame_equal(
        loaded.train,
        split.train,
    )

    pd.testing.assert_frame_equal(
        loaded.validation,
        split.validation,
    )

    pd.testing.assert_frame_equal(
        loaded.test,
        split.test,
    )


def test_load_frozen_split_rejects_row_count_mismatch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        run_mf_module,
        "PROJECT_ROOT",
        tmp_path,
    )

    split_dir = (
        tmp_path
        /
        "data"
        /
        "splits"
        /
        "synthetic"
        /
        "v1"
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split = _synthetic_split()

    for name in [
        "train",
        "validation",
        "test",
    ]:
        getattr(
            split,
            name,
        ).to_csv(
            split_dir
            /
            f"{name}.csv",
            index=False,
        )

    manifest = {
        "artifact": {
            "type": "canonical_split",
            "version": "v1",
            "dataset_id": "synthetic",
        },
        "protocol": {
            "seed": 2026,
            "constrained_attempt": 1,
            "enforce_train_item_coverage": True,
        },
        "statistics": {
            "train_rows": 999,
            "validation_rows": len(
                split.validation
            ),
            "test_rows": len(
                split.test
            ),
        },
        "files": {
            name: {
                "path": (
                    f"data/splits/synthetic/"
                    f"v1/{name}.csv"
                ),
            }
            for name in [
                "train",
                "validation",
                "test",
            ]
        },
    }

    (
        split_dir
        /
        "split_manifest.yaml"
    ).write_text(
        yaml.safe_dump(
            manifest,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="row count mismatch",
    ):
        load_frozen_split(
            _config(),
            version="v1",
        )


# ============================================================
# ExperimentSpec construction
# ============================================================


@pytest.mark.parametrize(
    "model_key",
    MODEL_ORDER,
)
def test_build_experiment_spec_matches_metadata(
    model_key,
):
    hyperparameters = {
        "example": 1,
    }

    spec = build_experiment_spec(
        model_key,
        seed=2026,
        hyperparameters=(
            hyperparameters
        ),
    )

    expected = (
        MODEL_METADATA[
            model_key
        ]
    )

    assert isinstance(
        spec,
        ExperimentSpec,
    )

    assert (
        spec.model_name
        ==
        expected[
            "model_name"
        ]
    )

    assert (
        spec.family
        ==
        expected[
            "family"
        ]
    )

    assert (
        spec.variant
        ==
        expected[
            "variant"
        ]
    )

    assert (
        spec.seed
        ==
        2026
    )

    assert (
        spec.hyperparameters
        ==
        hyperparameters
    )


# ============================================================
# build_model constructor wiring
# ============================================================


@pytest.mark.parametrize(
    (
        "model_key",
        "class_attribute",
    ),
    [
        (
            "pca_svd",
            "PCASVDBaseline",
        ),
        (
            "bernoulli_mf",
            "BernoulliMF",
        ),
        (
            "poisson_mf",
            "PoissonMF",
        ),
        (
            "gaussian_mf",
            "GaussianMF",
        ),
        (
            "logistic_pca",
            "LogisticPCA",
        ),
        (
            "hpf",
            "HPF",
        ),
    ],
)
def test_build_model_constructs_expected_class(
    monkeypatch,
    model_key,
    class_attribute,
):
    calls = []

    class FakeModel:

        def __init__(
            self,
            *args,
            **kwargs,
        ):
            calls.append(
                {
                    "args": args,
                    "kwargs": kwargs,
                }
            )

    monkeypatch.setattr(
        run_mf_module,
        class_attribute,
        FakeModel,
    )

    model, hyperparameters = (
        build_model(
            model_key,
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )
    )

    assert isinstance(
        model,
        FakeModel,
    )

    assert len(
        calls
    ) == 1

    assert isinstance(
        hyperparameters,
        dict,
    )

    assert (
        hyperparameters[
            "seed"
        ]
        ==
        2026
    )


def test_bernoulli_build_model_uses_leakage_safe_sampler(
    monkeypatch,
):
    captured = {}

    class FakeSampler:

        def __init__(
            self,
            *,
            train,
            all_observed,
            seed,
            namespace,
            popularity_alpha,
        ):
            captured[
                "train"
            ] = train

            captured[
                "all_observed"
            ] = all_observed

            captured[
                "seed"
            ] = seed

            captured[
                "namespace"
            ] = namespace

            captured[
                "popularity_alpha"
            ] = popularity_alpha

    class FakeBernoulliMF:

        def __init__(
            self,
            *,
            negative_sampler,
            **kwargs,
        ):
            captured[
                "sampler"
            ] = negative_sampler

            captured[
                "model_kwargs"
            ] = kwargs

    monkeypatch.setattr(
        run_mf_module,
        "NegativeSampler",
        FakeSampler,
    )

    monkeypatch.setattr(
        run_mf_module,
        "BernoulliMF",
        FakeBernoulliMF,
    )

    split = _synthetic_split()

    build_model(
        "bernoulli_mf",
        split=split,
        config=_config(),
        seed=2026,
    )

    assert (
        captured[
            "train"
        ]
        is split.train
    )

    assert len(
        captured[
            "all_observed"
        ]
    ) == (
        len(
            split.train
        )
        +
        len(
            split.validation
        )
        +
        len(
            split.test
        )
    )

    assert (
        captured[
            "seed"
        ]
        ==
        2026
    )

    assert (
        captured[
            "namespace"
        ]
        ==
        "bernoulli_mf_negative_sampling"
    )

    assert (
        captured[
            "model_kwargs"
        ][
            "negative_strategy"
        ]
        ==
        "uniform"
    )


def test_poisson_build_model_uses_leakage_safe_sampler(
    monkeypatch,
):
    captured = {}

    class FakeSampler:

        def __init__(
            self,
            *,
            train,
            all_observed,
            seed,
            namespace,
            popularity_alpha,
        ):
            captured[
                "train"
            ] = train

            captured[
                "all_observed"
            ] = all_observed

            captured[
                "seed"
            ] = seed

            captured[
                "namespace"
            ] = namespace

    class FakePoissonMF:

        def __init__(
            self,
            *,
            negative_sampler,
            **kwargs,
        ):
            captured[
                "sampler"
            ] = negative_sampler

            captured[
                "model_kwargs"
            ] = kwargs

    monkeypatch.setattr(
        run_mf_module,
        "NegativeSampler",
        FakeSampler,
    )

    monkeypatch.setattr(
        run_mf_module,
        "PoissonMF",
        FakePoissonMF,
    )

    split = _synthetic_split()

    build_model(
        "poisson_mf",
        split=split,
        config=_config(),
        seed=2026,
    )

    assert (
        captured[
            "train"
        ]
        is split.train
    )

    assert len(
        captured[
            "all_observed"
        ]
    ) == (
        len(
            split.train
        )
        +
        len(
            split.validation
        )
        +
        len(
            split.test
        )
    )

    assert (
        captured[
            "namespace"
        ]
        ==
        "poisson_mf_zero_sampling"
    )

    assert (
        captured[
            "model_kwargs"
        ][
            "zero_strategy"
        ]
        ==
        "uniform"
    )


@pytest.mark.parametrize(
    (
        "model_key",
        "class_attribute",
    ),
    [
        (
            "logistic_pca",
            "LogisticPCA",
        ),
        (
            "hpf",
            "HPF",
        ),
    ],
)
def test_literature_models_receive_validation_plus_test_mask(
    monkeypatch,
    model_key,
    class_attribute,
):
    captured = {}

    class FakeModel:

        def __init__(
            self,
            *,
            heldout_interactions,
            **kwargs,
        ):
            captured[
                "heldout"
            ] = heldout_interactions

            captured[
                "kwargs"
            ] = kwargs

    monkeypatch.setattr(
        run_mf_module,
        class_attribute,
        FakeModel,
    )

    split = _synthetic_split()

    build_model(
        model_key,
        split=split,
        config=_config(),
        seed=2026,
    )

    expected = pd.concat(
        [
            split.validation,
            split.test,
        ],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(
        captured[
            "heldout"
        ].reset_index(
            drop=True
        ),
        expected,
    )


# ============================================================
# Test-access protection
# ============================================================


def test_run_selected_models_blocks_test_before_split_load(
    monkeypatch,
):
    split_called = False

    monkeypatch.setattr(
        run_mf_module,
        "load_yaml",
        lambda path: _config(),
    )

    def forbidden_split(
        *args,
        **kwargs,
    ):
        nonlocal split_called

        split_called = True

        raise AssertionError(
            "Split must not be loaded"
        )

    monkeypatch.setattr(
        run_mf_module,
        "load_frozen_split",
        forbidden_split,
    )

    with pytest.raises(
        PermissionError,
        match="Test evaluation is disabled",
    ):
        run_selected_models(
            config_path="config/dev.yaml",
            version="v1",
            model_selection="pca_svd",
            target_split="test",
            allow_test=False,
        )

    assert (
        split_called
        is False
    )


# ============================================================
# run_selected_models orchestration
# ============================================================


def test_run_selected_models_executes_one_selected_model(
    tmp_path,
    monkeypatch,
):
    split = _synthetic_split()

    run = _run_context(
        tmp_path
    )

    calls = {
        "build_model": [],
        "run_experiment": [],
        "write": [],
    }

    monkeypatch.setattr(
        run_mf_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        run_mf_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        run_mf_module,
        "prepare_run_context",
        lambda **kwargs: run,
    )

    fake_model = (
        SimpleNamespace(
            model_name="PCA/SVD"
        )
    )

    def fake_build_model(
        model_key,
        *,
        split,
        config,
        seed,
    ):
        calls[
            "build_model"
        ].append(
            model_key
        )

        return (
            fake_model,
            {
                "n_components": 32,
                "seed": seed,
                "n_iter": 7,
            },
        )

    monkeypatch.setattr(
        run_mf_module,
        "build_model",
        fake_build_model,
    )

    def fake_run_experiment(
        *,
        model,
        split,
        config,
        spec,
        target_split,
        allow_test,
    ):
        calls[
            "run_experiment"
        ].append(
            {
                "model": model,
                "spec": spec,
                "target_split": (
                    target_split
                ),
                "allow_test": (
                    allow_test
                ),
            }
        )

        return _result(
            model_name="PCA/SVD",
            family="baseline",
        )

    monkeypatch.setattr(
        run_mf_module,
        "run_experiment",
        fake_run_experiment,
    )

    def fake_write(
        *,
        run,
        result,
    ):
        calls[
            "write"
        ].append(
            result
        )

        return SimpleNamespace(
            experiment_id="exp-1",
            experiment_dir=(
                tmp_path
                /
                "exp-1"
            ),
        )

    monkeypatch.setattr(
        run_mf_module,
        "write_experiment_result",
        fake_write,
    )

    returned_run, results = (
        run_selected_models(
            config_path="config/dev.yaml",
            version="v1",
            model_selection="pca_svd",
            seed=2026,
            target_split="validation",
            allow_test=False,
        )
    )

    assert (
        returned_run
        is run
    )

    assert (
        calls[
            "build_model"
        ]
        ==
        [
            "pca_svd",
        ]
    )

    assert len(
        calls[
            "run_experiment"
        ]
    ) == 1

    assert len(
        calls[
            "write"
        ]
    ) == 1

    assert len(
        results
    ) == 1

    assert (
        results[
            0
        ].model_name
        ==
        "PCA/SVD"
    )


def test_run_selected_models_all_uses_one_run_and_frozen_order(
    tmp_path,
    monkeypatch,
):
    split = _synthetic_split()

    run = _run_context(
        tmp_path
    )

    built = []

    written = []

    monkeypatch.setattr(
        run_mf_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        run_mf_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    prepare_calls = []

    def fake_prepare_run_context(
        **kwargs,
    ):
        prepare_calls.append(
            kwargs
        )

        return run

    monkeypatch.setattr(
        run_mf_module,
        "prepare_run_context",
        fake_prepare_run_context,
    )

    def fake_build_model(
        model_key,
        *,
        split,
        config,
        seed,
    ):
        built.append(
            model_key
        )

        return (
            SimpleNamespace(
                model_name=(
                    MODEL_METADATA[
                        model_key
                    ][
                        "model_name"
                    ]
                )
            ),
            {
                "seed": seed,
            },
        )

    monkeypatch.setattr(
        run_mf_module,
        "build_model",
        fake_build_model,
    )

    def fake_run_experiment(
        *,
        model,
        split,
        config,
        spec,
        target_split,
        allow_test,
    ):
        return _result(
            model_name=(
                spec.model_name
            ),
            family=(
                spec.family
            ),
            seed=(
                spec.seed
            ),
            target_split=(
                target_split
            ),
        )

    monkeypatch.setattr(
        run_mf_module,
        "run_experiment",
        fake_run_experiment,
    )

    def fake_write(
        *,
        run,
        result,
    ):
        written.append(
            result.model_name
        )

        return SimpleNamespace(
            experiment_id=(
                "exp-"
                +
                result.model_name.replace(
                    "/",
                    "-"
                )
            ),
            experiment_dir=(
                tmp_path
                /
                "stored"
            ),
        )

    monkeypatch.setattr(
        run_mf_module,
        "write_experiment_result",
        fake_write,
    )

    returned_run, results = (
        run_selected_models(
            config_path="config/dev.yaml",
            version="v1",
            model_selection="all",
            seed=2026,
            target_split="validation",
            allow_test=False,
        )
    )

    assert (
        returned_run
        is run
    )

    assert len(
        prepare_calls
    ) == 1

    assert (
        built
        ==
        MODEL_ORDER
    )

    expected_names = [
        MODEL_METADATA[
            key
        ][
            "model_name"
        ]
        for key in MODEL_ORDER
    ]

    assert (
        written
        ==
        expected_names
    )

    assert [
        result.model_name
        for result in results
    ] == expected_names


def test_authorized_test_selection_reaches_experiment_layer(
    tmp_path,
    monkeypatch,
):
    split = _synthetic_split()

    run = _run_context(
        tmp_path
    )

    captured = {}

    monkeypatch.setattr(
        run_mf_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        run_mf_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        run_mf_module,
        "prepare_run_context",
        lambda **kwargs: run,
    )

    monkeypatch.setattr(
        run_mf_module,
        "build_model",
        lambda *args, **kwargs: (
            SimpleNamespace(
                model_name="PCA/SVD"
            ),
            {
                "seed": 2026,
            },
        ),
    )

    def fake_run_experiment(
        *,
        model,
        split,
        config,
        spec,
        target_split,
        allow_test,
    ):
        captured[
            "target_split"
        ] = target_split

        captured[
            "allow_test"
        ] = allow_test

        return _result(
            model_name="PCA/SVD",
            family="baseline",
            target_split="test",
        )

    monkeypatch.setattr(
        run_mf_module,
        "run_experiment",
        fake_run_experiment,
    )

    monkeypatch.setattr(
        run_mf_module,
        "write_experiment_result",
        lambda **kwargs: SimpleNamespace(
            experiment_id="exp-test",
            experiment_dir=(
                tmp_path
                /
                "exp-test"
            ),
        ),
    )

    _, results = run_selected_models(
        config_path="config/dev.yaml",
        version="v1",
        model_selection="pca_svd",
        seed=2026,
        target_split="test",
        allow_test=True,
    )

    assert (
        captured[
            "target_split"
        ]
        ==
        "test"
    )

    assert (
        captured[
            "allow_test"
        ]
        is True
    )

    assert (
        results[
            0
        ].target_split
        ==
        "test"
    )


# ============================================================
# Run-context reopening protection
# ============================================================


def test_prepare_run_context_reopens_matching_existing_run(
    tmp_path,
    monkeypatch,
):
    config = _config()

    provenance = (
        _provenance()
    )

    existing = (
        _run_context(
            tmp_path,
            run_id="existing-run",
        )
    )

    candidate = (
        tmp_path
        /
        "existing-run"
    )

    candidate.mkdir(
        parents=True,
        exist_ok=True,
    )

    monkeypatch.setattr(
        run_mf_module,
        "PROJECT_ROOT",
        tmp_path,
    )

    monkeypatch.setattr(
        run_mf_module,
        "build_run_provenance",
        lambda **kwargs: provenance,
    )

    monkeypatch.setattr(
        run_mf_module,
        "load_run_context",
        lambda path: existing,
    )

    create_called = False

    def forbidden_create(
        **kwargs,
    ):
        nonlocal create_called

        create_called = True

        raise AssertionError(
            "create_run must not be called"
        )

    monkeypatch.setattr(
        run_mf_module,
        "create_run",
        forbidden_create,
    )

    reopened = prepare_run_context(
        config_path=(
            tmp_path
            /
            "config"
            /
            "dev.yaml"
        ),
        config=config,
        version="v1",
        results_root=tmp_path,
        run_id="existing-run",
    )

    assert (
        reopened
        is existing
    )

    assert (
        create_called
        is False
    )


def test_prepare_run_context_rejects_existing_provenance_mismatch(
    tmp_path,
    monkeypatch,
):
    config = _config()

    current = (
        _provenance()
    )

    mismatched = RunProvenance(
        dataset_id="synthetic",
        artifact_version="v1",
        config_hash="f" * 64,
        dataset_hash="b" * 64,
        split_hash="c" * 64,
        code_version="d" * 64,
    )

    existing = RunContext(
        run_id="existing-run",
        run_dir=(
            tmp_path
            /
            "existing-run"
        ),
        provenance=mismatched,
        created_at_utc=(
            "2026-10-06T20:00:00.000000Z"
        ),
    )

    existing.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    monkeypatch.setattr(
        run_mf_module,
        "PROJECT_ROOT",
        tmp_path,
    )

    monkeypatch.setattr(
        run_mf_module,
        "build_run_provenance",
        lambda **kwargs: current,
    )

    monkeypatch.setattr(
        run_mf_module,
        "load_run_context",
        lambda path: existing,
    )

    with pytest.raises(
        ValueError,
        match="provenance",
    ):
        prepare_run_context(
            config_path=(
                tmp_path
                /
                "config"
                /
                "dev.yaml"
            ),
            config=config,
            version="v1",
            results_root=tmp_path,
            run_id="existing-run",
        )