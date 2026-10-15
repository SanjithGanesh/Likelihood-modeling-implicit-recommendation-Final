"""Permanent regression tests for scripts/run_baselines.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_baselines as baseline_module

from scripts.run_baselines import (
    BASELINE_CHOICES,
    BASELINE_METADATA,
    BASELINE_ORDER,
    build_baseline,
    build_experiment_spec,
    resolve_baseline_keys,
    run_selected_baselines,
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
    """Return minimal baseline-runner configuration."""

    return {
        "dataset": {
            "id": "synthetic",
        },
        "evaluation": {
            "primary_selection_metric": (
                "ndcg_at_10"
            ),
        },
        "reproducibility": {
            "global_seed": 2026,
            "seeds": [
                2026,
            ],
        },
        "paths": {
            "results_root": "results",
        },
    }


def _provenance() -> RunProvenance:
    """Return deterministic synthetic provenance."""

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
    run_id: str = "baseline-test-run",
) -> RunContext:
    """Return one lightweight RunContext."""

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
    seed: int = 2026,
    target_split: str = "validation",
    hyperparameters=None,
) -> ExperimentResult:
    """Return one valid baseline ExperimentResult."""

    if hyperparameters is None:
        hyperparameters = {}

    return ExperimentResult(
        model_name=model_name,
        family="baseline",
        variant="primary",
        seed=seed,
        target_split=target_split,
        ranking_metrics={
            "ndcg_at_10": 0.25,
        },
        native_metrics={},
        training_history=[],
        runtime_seconds=0.1,
        status="completed",
        error_message=None,
        metadata={
            "hyperparameters": (
                hyperparameters
            ),
        },
    )


# ============================================================
# Frozen inventory
# ============================================================


def test_baseline_inventory_is_frozen():

    assert BASELINE_ORDER == [
        "popularity",
        "random",
    ]

    assert BASELINE_CHOICES == [
        "popularity",
        "random",
        "all",
    ]


def test_baseline_inventory_does_not_duplicate_pca_svd():

    assert (
        "pca_svd"
        not in BASELINE_ORDER
    )

    assert (
        "PCA/SVD"
        not in [
            metadata[
                "model_name"
            ]
            for metadata
            in BASELINE_METADATA.values()
        ]
    )


def test_baseline_metadata_matches_public_names():

    assert (
        BASELINE_METADATA[
            "popularity"
        ]
        ==
        {
            "model_name": "Popularity",
            "family": "baseline",
            "variant": "primary",
        }
    )

    assert (
        BASELINE_METADATA[
            "random"
        ]
        ==
        {
            "model_name": "Random",
            "family": "baseline",
            "variant": "primary",
        }
    )


def test_baseline_model_names_are_unique():

    names = [
        metadata[
            "model_name"
        ]
        for metadata
        in BASELINE_METADATA.values()
    ]

    assert len(
        names
    ) == len(
        set(
            names
        )
    )


# ============================================================
# Selection helpers
# ============================================================


@pytest.mark.parametrize(
    "baseline_key",
    BASELINE_ORDER,
)
def test_resolve_baseline_keys_single(
    baseline_key,
):

    assert (
        resolve_baseline_keys(
            baseline_key
        )
        ==
        [
            baseline_key,
        ]
    )


def test_resolve_baseline_keys_all_preserves_order():

    assert (
        resolve_baseline_keys(
            "all"
        )
        ==
        BASELINE_ORDER
    )


def test_resolve_baseline_keys_returns_copy_for_all():

    resolved = (
        resolve_baseline_keys(
            "all"
        )
    )

    resolved.append(
        "junk"
    )

    assert (
        BASELINE_ORDER
        ==
        [
            "popularity",
            "random",
        ]
    )


def test_resolve_baseline_keys_rejects_unknown():

    with pytest.raises(
        ValueError,
        match="Unknown baseline selection",
    ):
        resolve_baseline_keys(
            "pca_svd"
        )


# ============================================================
# Baseline construction
# ============================================================


def test_build_popularity_constructs_canonical_class(
    monkeypatch,
):

    calls = []

    class FakePopularity:

        def __init__(
            self,
        ):
            calls.append(
                True
            )

    monkeypatch.setattr(
        baseline_module,
        "PopularityBaseline",
        FakePopularity,
    )

    model, hyperparameters = (
        build_baseline(
            "popularity",
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )
    )

    assert isinstance(
        model,
        FakePopularity,
    )

    assert calls == [
        True,
    ]

    assert (
        hyperparameters
        ==
        {}
    )


def test_build_random_constructs_seeded_canonical_class(
    monkeypatch,
):

    captured = {}

    class FakeRandom:

        def __init__(
            self,
            *,
            seed,
            namespace,
        ):

            captured[
                "seed"
            ] = seed

            captured[
                "namespace"
            ] = namespace

    monkeypatch.setattr(
        baseline_module,
        "RandomBaseline",
        FakeRandom,
    )

    model, hyperparameters = (
        build_baseline(
            "random",
            split=_synthetic_split(),
            config=_config(),
            seed=2027,
        )
    )

    assert isinstance(
        model,
        FakeRandom,
    )

    assert (
        captured[
            "seed"
        ]
        ==
        2027
    )

    assert (
        captured[
            "namespace"
        ]
        ==
        "random_baseline"
    )

    assert (
        hyperparameters
        ==
        {
            "seed": 2027,
            "namespace": (
                "random_baseline"
            ),
        }
    )


def test_build_baseline_rejects_unknown():

    with pytest.raises(
        ValueError,
        match="Unknown baseline key",
    ):
        build_baseline(
            "pca_svd",
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )


def test_build_baseline_rejects_non_split():

    with pytest.raises(
        TypeError,
        match="SplitResult",
    ):
        build_baseline(
            "popularity",
            split=None,
            config=_config(),
            seed=2026,
        )


def test_build_baseline_rejects_non_dict_config():

    with pytest.raises(
        TypeError,
        match="dictionary",
    ):
        build_baseline(
            "popularity",
            split=_synthetic_split(),
            config=None,
            seed=2026,
        )


def test_build_baseline_rejects_boolean_seed():

    with pytest.raises(
        ValueError,
        match="integer",
    ):
        build_baseline(
            "random",
            split=_synthetic_split(),
            config=_config(),
            seed=True,
        )


# ============================================================
# ExperimentSpec
# ============================================================


@pytest.mark.parametrize(
    "baseline_key",
    BASELINE_ORDER,
)
def test_build_experiment_spec_matches_metadata(
    baseline_key,
):

    hyperparameters = {
        "example": 1,
    }

    spec = (
        build_experiment_spec(
            baseline_key,
            seed=2026,
            hyperparameters=(
                hyperparameters
            ),
        )
    )

    metadata = (
        BASELINE_METADATA[
            baseline_key
        ]
    )

    assert isinstance(
        spec,
        ExperimentSpec,
    )

    assert (
        spec.model_name
        ==
        metadata[
            "model_name"
        ]
    )

    assert (
        spec.family
        ==
        "baseline"
    )

    assert (
        spec.variant
        ==
        "primary"
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


def test_build_experiment_spec_rejects_unknown():

    with pytest.raises(
        ValueError,
        match="Unknown baseline key",
    ):
        build_experiment_spec(
            "pca_svd",
            seed=2026,
            hyperparameters={},
        )


# ============================================================
# Test-access protection
# ============================================================


def test_run_selected_baselines_blocks_test_before_split_load(
    monkeypatch,
):

    split_called = False

    monkeypatch.setattr(
        baseline_module,
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
            "Frozen split must not be loaded"
        )

    monkeypatch.setattr(
        baseline_module,
        "load_frozen_split",
        forbidden_split,
    )

    with pytest.raises(
        PermissionError,
        match="Test evaluation is disabled",
    ):
        run_selected_baselines(
            config_path="config/dev.yaml",
            version="v1",
            baseline_selection="popularity",
            target_split="test",
            allow_test=False,
        )

    assert (
        split_called
        is False
    )


def test_run_selected_baselines_rejects_invalid_target(
    monkeypatch,
):

    monkeypatch.setattr(
        baseline_module,
        "load_yaml",
        lambda path: _config(),
    )

    with pytest.raises(
        ValueError,
        match="target_split",
    ):
        run_selected_baselines(
            config_path="config/dev.yaml",
            version="v1",
            baseline_selection="popularity",
            target_split="training",
        )


# ============================================================
# Runner orchestration
# ============================================================


def test_run_selected_baselines_executes_one_baseline(
    tmp_path,
    monkeypatch,
):

    split = (
        _synthetic_split()
    )

    run = (
        _run_context(
            tmp_path
        )
    )

    calls = {
        "build": [],
        "experiment": [],
        "write": [],
    }

    monkeypatch.setattr(
        baseline_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        baseline_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        baseline_module,
        "resolve_seed",
        lambda config, seed: 2026,
    )

    monkeypatch.setattr(
        baseline_module,
        "prepare_run_context",
        lambda **kwargs: run,
    )

    fake_model = (
        SimpleNamespace(
            model_name="Popularity"
        )
    )

    def fake_build(
        baseline_key,
        *,
        split,
        config,
        seed,
    ):

        calls[
            "build"
        ].append(
            baseline_key
        )

        return (
            fake_model,
            {},
        )

    monkeypatch.setattr(
        baseline_module,
        "build_baseline",
        fake_build,
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
            "experiment"
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
            model_name="Popularity",
        )

    monkeypatch.setattr(
        baseline_module,
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
            result.model_name
        )

        return SimpleNamespace(
            experiment_id=(
                "popularity-exp"
            ),
            experiment_dir=(
                tmp_path
                /
                "popularity-exp"
            ),
        )

    monkeypatch.setattr(
        baseline_module,
        "write_experiment_result",
        fake_write,
    )

    returned_run, results = (
        run_selected_baselines(
            config_path="config/dev.yaml",
            version="v1",
            baseline_selection=(
                "popularity"
            ),
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
            "build"
        ]
        ==
        [
            "popularity",
        ]
    )

    assert len(
        calls[
            "experiment"
        ]
    ) == 1

    assert (
        calls[
            "experiment"
        ][
            0
        ][
            "target_split"
        ]
        ==
        "validation"
    )

    assert (
        calls[
            "experiment"
        ][
            0
        ][
            "allow_test"
        ]
        is False
    )

    assert (
        calls[
            "write"
        ]
        ==
        [
            "Popularity",
        ]
    )

    assert len(
        results
    ) == 1

    assert (
        results[
            0
        ].model_name
        ==
        "Popularity"
    )


def test_run_selected_baselines_all_uses_one_run_and_frozen_order(
    tmp_path,
    monkeypatch,
):

    split = (
        _synthetic_split()
    )

    run = (
        _run_context(
            tmp_path
        )
    )

    built = []
    written = []
    prepare_calls = []

    monkeypatch.setattr(
        baseline_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        baseline_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        baseline_module,
        "resolve_seed",
        lambda config, seed: 2026,
    )

    def fake_prepare(
        **kwargs,
    ):
        prepare_calls.append(
            kwargs
        )

        return run

    monkeypatch.setattr(
        baseline_module,
        "prepare_run_context",
        fake_prepare,
    )

    def fake_build(
        baseline_key,
        *,
        split,
        config,
        seed,
    ):

        built.append(
            baseline_key
        )

        name = (
            BASELINE_METADATA[
                baseline_key
            ][
                "model_name"
            ]
        )

        hyperparameters = (
            {}
            if baseline_key
            ==
            "popularity"
            else {
                "seed": seed,
                "namespace": (
                    "random_baseline"
                ),
            }
        )

        return (
            SimpleNamespace(
                model_name=name
            ),
            hyperparameters,
        )

    monkeypatch.setattr(
        baseline_module,
        "build_baseline",
        fake_build,
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
            seed=(
                spec.seed
            ),
            target_split=(
                target_split
            ),
            hyperparameters=(
                spec.hyperparameters
            ),
        )

    monkeypatch.setattr(
        baseline_module,
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
                result.model_name.lower()
            ),
            experiment_dir=(
                tmp_path
                /
                "stored"
            ),
        )

    monkeypatch.setattr(
        baseline_module,
        "write_experiment_result",
        fake_write,
    )

    returned_run, results = (
        run_selected_baselines(
            config_path="config/dev.yaml",
            version="v1",
            baseline_selection="all",
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
        [
            "popularity",
            "random",
        ]
    )

    assert (
        written
        ==
        [
            "Popularity",
            "Random",
        ]
    )

    assert [
        result.model_name
        for result
        in results
    ] == [
        "Popularity",
        "Random",
    ]


def test_all_selection_never_constructs_pca_svd(
    tmp_path,
    monkeypatch,
):

    split = (
        _synthetic_split()
    )

    run = (
        _run_context(
            tmp_path
        )
    )

    requested = []

    monkeypatch.setattr(
        baseline_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        baseline_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        baseline_module,
        "resolve_seed",
        lambda config, seed: 2026,
    )

    monkeypatch.setattr(
        baseline_module,
        "prepare_run_context",
        lambda **kwargs: run,
    )

    def fake_build(
        baseline_key,
        **kwargs,
    ):

        requested.append(
            baseline_key
        )

        name = (
            BASELINE_METADATA[
                baseline_key
            ][
                "model_name"
            ]
        )

        return (
            SimpleNamespace(
                model_name=name
            ),
            {},
        )

    monkeypatch.setattr(
        baseline_module,
        "build_baseline",
        fake_build,
    )

    monkeypatch.setattr(
        baseline_module,
        "run_experiment",
        lambda *,
        model,
        split,
        config,
        spec,
        target_split,
        allow_test: _result(
            model_name=(
                spec.model_name
            ),
        ),
    )

    monkeypatch.setattr(
        baseline_module,
        "write_experiment_result",
        lambda **kwargs: (
            SimpleNamespace(
                experiment_id="exp",
                experiment_dir=(
                    tmp_path
                    /
                    "exp"
                ),
            )
        ),
    )

    run_selected_baselines(
        config_path="config/dev.yaml",
        version="v1",
        baseline_selection="all",
        seed=2026,
    )

    assert (
        requested
        ==
        [
            "popularity",
            "random",
        ]
    )

    assert (
        "pca_svd"
        not in requested
    )


def test_authorized_test_reaches_experiment_layer(
    tmp_path,
    monkeypatch,
):

    split = (
        _synthetic_split()
    )

    run = (
        _run_context(
            tmp_path
        )
    )

    captured = {}

    monkeypatch.setattr(
        baseline_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        baseline_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        baseline_module,
        "resolve_seed",
        lambda config, seed: 2026,
    )

    monkeypatch.setattr(
        baseline_module,
        "prepare_run_context",
        lambda **kwargs: run,
    )

    monkeypatch.setattr(
        baseline_module,
        "build_baseline",
        lambda *args, **kwargs: (
            SimpleNamespace(
                model_name="Popularity"
            ),
            {},
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
            model_name="Popularity",
            target_split="test",
        )

    monkeypatch.setattr(
        baseline_module,
        "run_experiment",
        fake_run_experiment,
    )

    monkeypatch.setattr(
        baseline_module,
        "write_experiment_result",
        lambda **kwargs: (
            SimpleNamespace(
                experiment_id="test-exp",
                experiment_dir=(
                    tmp_path
                    /
                    "test-exp"
                ),
            )
        ),
    )

    _, results = (
        run_selected_baselines(
            config_path="config/dev.yaml",
            version="v1",
            baseline_selection=(
                "popularity"
            ),
            seed=2026,
            target_split="test",
            allow_test=True,
        )
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
# Shared run handoff
# ============================================================


def test_runner_forwards_explicit_run_id_to_shared_run_layer(
    tmp_path,
    monkeypatch,
):

    split = (
        _synthetic_split()
    )

    run = (
        _run_context(
            tmp_path,
            run_id="shared-paper2-run",
        )
    )

    captured = {}

    monkeypatch.setattr(
        baseline_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        baseline_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        baseline_module,
        "resolve_seed",
        lambda config, seed: 2026,
    )

    def fake_prepare_run_context(
        **kwargs,
    ):

        captured.update(
            kwargs
        )

        return run

    monkeypatch.setattr(
        baseline_module,
        "prepare_run_context",
        fake_prepare_run_context,
    )

    monkeypatch.setattr(
        baseline_module,
        "build_baseline",
        lambda *args, **kwargs: (
            SimpleNamespace(
                model_name="Popularity"
            ),
            {},
        ),
    )

    monkeypatch.setattr(
        baseline_module,
        "run_experiment",
        lambda **kwargs: _result(
            model_name="Popularity",
        ),
    )

    monkeypatch.setattr(
        baseline_module,
        "write_experiment_result",
        lambda **kwargs: (
            SimpleNamespace(
                experiment_id="exp",
                experiment_dir=(
                    tmp_path
                    /
                    "exp"
                ),
            )
        ),
    )

    returned_run, _ = (
        run_selected_baselines(
            config_path="config/dev.yaml",
            version="v1",
            baseline_selection=(
                "popularity"
            ),
            seed=2026,
            run_id=(
                "shared-paper2-run"
            ),
            results_root=(
                tmp_path
            ),
        )
    )

    assert (
        returned_run
        is run
    )

    assert (
        captured[
            "run_id"
        ]
        ==
        "shared-paper2-run"
    )

    assert (
        Path(
            captured[
                "results_root"
            ]
        )
        ==
        tmp_path
    )


# ============================================================
# CLI surface
# ============================================================


def test_cli_model_choices_are_exact():

    parser = (
        baseline_module
        .build_argument_parser()
    )

    model_action = next(
        action
        for action
        in parser._actions
        if action.dest
        ==
        "model"
    )

    assert list(
        model_action.choices
    ) == [
        "popularity",
        "random",
        "all",
    ]