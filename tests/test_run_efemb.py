"""Permanent regression tests for scripts/run_efemb.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_efemb as ef_module

from scripts.run_efemb import (
    EF_MODEL_CHOICES,
    EF_MODEL_METADATA,
    EF_MODEL_ORDER,
    PRIMARY_CONTEXT_NORMALIZATION,
    build_ef_model,
    build_experiment_spec,
    resolve_ef_model_keys,
    run_selected_ef_models,
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
    """Return EF-runner-compatible synthetic configuration."""

    return {
        "dataset": {
            "id": "synthetic",
        },
        "training": {
            "default_epochs": 5,
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
            "primary_selection_metric": (
                "ndcg_at_10"
            ),
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
    run_id: str = "ef-test-run",
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
    family: str,
    seed: int = 2026,
    target_split: str = "validation",
    hyperparameters=None,
    native_metrics=None,
) -> ExperimentResult:
    """Return one valid synthetic EF result."""

    if hyperparameters is None:
        hyperparameters = {}

    if native_metrics is None:
        native_metrics = {}

    return ExperimentResult(
        model_name=model_name,
        family=family,
        variant="primary",
        seed=seed,
        target_split=target_split,
        ranking_metrics={
            "ndcg_at_10": 0.25,
        },
        native_metrics=native_metrics,
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


class _FakeSampler:
    """Minimal sampler stand-in used for constructor tests."""

    def __init__(
        self,
        *,
        train,
        all_observed,
        seed,
        namespace,
        popularity_alpha,
    ):
        self.train = train
        self.all_observed = all_observed
        self.seed = seed
        self.namespace = namespace
        self.popularity_alpha = popularity_alpha


# ============================================================
# Frozen inventory
# ============================================================


def test_ef_inventory_is_frozen():

    assert EF_MODEL_ORDER == [
        "b_emb",
        "p_emb",
        "ap_emb",
        "g_emb",
        "ng_emb",
    ]

    assert EF_MODEL_CHOICES == [
        "b_emb",
        "p_emb",
        "ap_emb",
        "g_emb",
        "ng_emb",
        "all",
    ]


def test_ef_public_model_names_are_frozen():

    assert [
        EF_MODEL_METADATA[
            key
        ][
            "model_name"
        ]
        for key
        in EF_MODEL_ORDER
    ] == [
        "b-emb",
        "p-emb",
        "ap-emb",
        "g-emb",
        "ng-emb",
    ]


def test_ef_family_assignments_are_frozen():

    assert {
        key: (
            EF_MODEL_METADATA[
                key
            ][
                "family"
            ]
        )
        for key
        in EF_MODEL_ORDER
    } == {
        "b_emb": "bernoulli",
        "p_emb": "poisson",
        "ap_emb": "poisson",
        "g_emb": "gaussian",
        "ng_emb": "gaussian",
    }


def test_ef_public_model_names_are_unique():

    names = [
        metadata[
            "model_name"
        ]
        for metadata
        in EF_MODEL_METADATA.values()
    ]

    assert len(
        names
    ) == len(
        set(
            names
        )
    )


def test_primary_context_normalization_is_mean():

    assert (
        PRIMARY_CONTEXT_NORMALIZATION
        ==
        "mean"
    )


# ============================================================
# Selection helpers
# ============================================================


@pytest.mark.parametrize(
    "model_key",
    EF_MODEL_ORDER,
)
def test_resolve_ef_model_keys_single(
    model_key,
):

    assert (
        resolve_ef_model_keys(
            model_key
        )
        ==
        [
            model_key,
        ]
    )


def test_resolve_ef_model_keys_all_preserves_order():

    assert (
        resolve_ef_model_keys(
            "all"
        )
        ==
        EF_MODEL_ORDER
    )


def test_resolve_ef_model_keys_returns_copy_for_all():

    resolved = (
        resolve_ef_model_keys(
            "all"
        )
    )

    resolved.append(
        "junk"
    )

    assert EF_MODEL_ORDER == [
        "b_emb",
        "p_emb",
        "ap_emb",
        "g_emb",
        "ng_emb",
    ]


def test_resolve_ef_model_keys_rejects_unknown():

    with pytest.raises(
        ValueError,
        match="Unknown EF model selection",
    ):
        resolve_ef_model_keys(
            "unknown"
        )


# ============================================================
# ExperimentSpec
# ============================================================


@pytest.mark.parametrize(
    "model_key",
    EF_MODEL_ORDER,
)
def test_build_experiment_spec_matches_metadata(
    model_key,
):

    hyperparameters = {
        "example": 1,
    }

    spec = (
        build_experiment_spec(
            model_key,
            seed=2026,
            hyperparameters=(
                hyperparameters
            ),
        )
    )

    expected = (
        EF_MODEL_METADATA[
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
        match="Unknown EF model key",
    ):
        build_experiment_spec(
            "unknown",
            seed=2026,
            hyperparameters={},
        )


# ============================================================
# Generic constructor wiring
# ============================================================


@pytest.mark.parametrize(
    (
        "model_key",
        "class_attribute",
    ),
    [
        (
            "b_emb",
            "BernoulliEFEmb",
        ),
        (
            "p_emb",
            "PoissonEFEmb",
        ),
        (
            "ap_emb",
            "AdditivePoissonEFEmb",
        ),
        (
            "g_emb",
            "GaussianEFEmb",
        ),
        (
            "ng_emb",
            "NonnegativeGaussianEFEmb",
        ),
    ],
)
def test_build_ef_model_constructs_expected_class(
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
        ef_module,
        class_attribute,
        FakeModel,
    )

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        _FakeSampler,
    )

    model, hyperparameters = (
        build_ef_model(
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

    assert (
        hyperparameters[
            "context_normalization"
        ]
        ==
        "mean"
    )


def test_build_ef_model_rejects_unknown():

    with pytest.raises(
        ValueError,
        match="Unknown EF model key",
    ):
        build_ef_model(
            "unknown",
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )


def test_build_ef_model_rejects_non_split():

    with pytest.raises(
        TypeError,
        match="SplitResult",
    ):
        build_ef_model(
            "g_emb",
            split=None,
            config=_config(),
            seed=2026,
        )


def test_build_ef_model_rejects_non_dict_config():

    with pytest.raises(
        TypeError,
        match="dictionary",
    ):
        build_ef_model(
            "g_emb",
            split=_synthetic_split(),
            config=None,
            seed=2026,
        )


# ============================================================
# Leakage-safe sampler wiring
# ============================================================


def test_b_emb_uses_all_observed_negative_sampler(
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

    class FakeModel:

        def __init__(
            self,
            *,
            negative_sampler,
            heldout_interactions,
            **kwargs,
        ):

            captured[
                "sampler"
            ] = negative_sampler

            captured[
                "heldout"
            ] = heldout_interactions

            captured[
                "kwargs"
            ] = kwargs

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        FakeSampler,
    )

    monkeypatch.setattr(
        ef_module,
        "BernoulliEFEmb",
        FakeModel,
    )

    split = (
        _synthetic_split()
    )

    build_ef_model(
        "b_emb",
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
        "b_emb_negative_sampling"
    )

    assert (
        captured[
            "kwargs"
        ][
            "negative_strategy"
        ]
        ==
        "uniform"
    )

    assert (
        captured[
            "kwargs"
        ][
            "negative_ratio"
        ]
        ==
        5
    )


@pytest.mark.parametrize(
    (
        "model_key",
        "class_attribute",
        "expected_namespace",
    ),
    [
        (
            "p_emb",
            "PoissonEFEmb",
            "p_emb_zero_sampling",
        ),
        (
            "ap_emb",
            "AdditivePoissonEFEmb",
            "ap_emb_zero_sampling",
        ),
    ],
)
def test_poisson_ef_models_use_all_observed_zero_sampler(
    monkeypatch,
    model_key,
    class_attribute,
    expected_namespace,
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
                "namespace"
            ] = namespace

    class FakeModel:

        def __init__(
            self,
            *,
            zero_sampler,
            heldout_interactions,
            **kwargs,
        ):

            captured[
                "sampler"
            ] = zero_sampler

            captured[
                "heldout"
            ] = heldout_interactions

            captured[
                "kwargs"
            ] = kwargs

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        FakeSampler,
    )

    monkeypatch.setattr(
        ef_module,
        class_attribute,
        FakeModel,
    )

    split = (
        _synthetic_split()
    )

    build_ef_model(
        model_key,
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
        expected_namespace
    )

    assert (
        captured[
            "kwargs"
        ][
            "zero_sampling_strategy"
        ]
        ==
        "uniform"
    )

    assert (
        captured[
            "kwargs"
        ][
            "zero_ratio"
        ]
        ==
        5
    )


# ============================================================
# Heldout context-mask wiring
# ============================================================


@pytest.mark.parametrize(
    (
        "model_key",
        "class_attribute",
    ),
    [
        (
            "b_emb",
            "BernoulliEFEmb",
        ),
        (
            "p_emb",
            "PoissonEFEmb",
        ),
        (
            "ap_emb",
            "AdditivePoissonEFEmb",
        ),
        (
            "g_emb",
            "GaussianEFEmb",
        ),
        (
            "ng_emb",
            "NonnegativeGaussianEFEmb",
        ),
    ],
)
def test_all_ef_models_receive_validation_plus_test_mask(
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
        ef_module,
        class_attribute,
        FakeModel,
    )

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        _FakeSampler,
    )

    split = (
        _synthetic_split()
    )

    build_ef_model(
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
# Explicit primary hyperparameters
# ============================================================


@pytest.mark.parametrize(
    "model_key",
    EF_MODEL_ORDER,
)
def test_all_primary_ef_models_use_mean_context_normalization(
    monkeypatch,
    model_key,
):

    class FakeModel:

        def __init__(
            self,
            **kwargs,
        ):
            pass

    mapping = {
        "b_emb": "BernoulliEFEmb",
        "p_emb": "PoissonEFEmb",
        "ap_emb": "AdditivePoissonEFEmb",
        "g_emb": "GaussianEFEmb",
        "ng_emb": "NonnegativeGaussianEFEmb",
    }

    monkeypatch.setattr(
        ef_module,
        mapping[
            model_key
        ],
        FakeModel,
    )

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        _FakeSampler,
    )

    _, hyperparameters = (
        build_ef_model(
            model_key,
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )
    )

    assert (
        hyperparameters[
            "context_normalization"
        ]
        ==
        PRIMARY_CONTEXT_NORMALIZATION
    )


@pytest.mark.parametrize(
    "model_key",
    EF_MODEL_ORDER,
)
def test_all_primary_ef_models_use_frozen_factor_count(
    monkeypatch,
    model_key,
):

    class FakeModel:

        def __init__(
            self,
            **kwargs,
        ):
            pass

    mapping = {
        "b_emb": "BernoulliEFEmb",
        "p_emb": "PoissonEFEmb",
        "ap_emb": "AdditivePoissonEFEmb",
        "g_emb": "GaussianEFEmb",
        "ng_emb": "NonnegativeGaussianEFEmb",
    }

    monkeypatch.setattr(
        ef_module,
        mapping[
            model_key
        ],
        FakeModel,
    )

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        _FakeSampler,
    )

    _, hyperparameters = (
        build_ef_model(
            model_key,
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )
    )

    assert (
        hyperparameters[
            "n_factors"
        ]
        ==
        32
    )


@pytest.mark.parametrize(
    "model_key",
    EF_MODEL_ORDER,
)
def test_all_primary_ef_models_use_frozen_learning_rate(
    monkeypatch,
    model_key,
):

    class FakeModel:

        def __init__(
            self,
            **kwargs,
        ):
            pass

    mapping = {
        "b_emb": "BernoulliEFEmb",
        "p_emb": "PoissonEFEmb",
        "ap_emb": "AdditivePoissonEFEmb",
        "g_emb": "GaussianEFEmb",
        "ng_emb": "NonnegativeGaussianEFEmb",
    }

    monkeypatch.setattr(
        ef_module,
        mapping[
            model_key
        ],
        FakeModel,
    )

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        _FakeSampler,
    )

    _, hyperparameters = (
        build_ef_model(
            model_key,
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )
    )

    assert (
        hyperparameters[
            "learning_rate"
        ]
        ==
        0.05
    )


@pytest.mark.parametrize(
    "model_key",
    EF_MODEL_ORDER,
)
def test_all_primary_ef_models_use_frozen_batch_size(
    monkeypatch,
    model_key,
):

    class FakeModel:

        def __init__(
            self,
            **kwargs,
        ):
            pass

    mapping = {
        "b_emb": "BernoulliEFEmb",
        "p_emb": "PoissonEFEmb",
        "ap_emb": "AdditivePoissonEFEmb",
        "g_emb": "GaussianEFEmb",
        "ng_emb": "NonnegativeGaussianEFEmb",
    }

    monkeypatch.setattr(
        ef_module,
        mapping[
            model_key
        ],
        FakeModel,
    )

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        _FakeSampler,
    )

    _, hyperparameters = (
        build_ef_model(
            model_key,
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )
    )

    assert (
        hyperparameters[
            "batch_size"
        ]
        ==
        2048
    )


def test_ap_emb_has_explicit_additive_parameter(
    monkeypatch,
):

    class FakeModel:

        def __init__(
            self,
            **kwargs,
        ):
            pass

    monkeypatch.setattr(
        ef_module,
        "AdditivePoissonEFEmb",
        FakeModel,
    )

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        _FakeSampler,
    )

    _, hyperparameters = (
        build_ef_model(
            "ap_emb",
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )
    )

    assert (
        hyperparameters[
            "initial_interaction_rate"
        ]
        ==
        0.01
    )


def test_p_emb_and_ap_emb_have_distinct_primary_hyperparameters(
    monkeypatch,
):

    class FakeModel:

        def __init__(
            self,
            **kwargs,
        ):
            pass

    monkeypatch.setattr(
        ef_module,
        "PoissonEFEmb",
        FakeModel,
    )

    monkeypatch.setattr(
        ef_module,
        "AdditivePoissonEFEmb",
        FakeModel,
    )

    monkeypatch.setattr(
        ef_module,
        "NegativeSampler",
        _FakeSampler,
    )

    _, p_hyperparameters = (
        build_ef_model(
            "p_emb",
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )
    )

    _, ap_hyperparameters = (
        build_ef_model(
            "ap_emb",
            split=_synthetic_split(),
            config=_config(),
            seed=2026,
        )
    )

    assert (
        "initial_interaction_rate"
        not in p_hyperparameters
    )

    assert (
        ap_hyperparameters[
            "initial_interaction_rate"
        ]
        ==
        0.01
    )


# ============================================================
# Test-access protection
# ============================================================


def test_run_selected_ef_models_blocks_test_before_split_load(
    monkeypatch,
):

    split_called = False

    monkeypatch.setattr(
        ef_module,
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
        ef_module,
        "load_frozen_split",
        forbidden_split,
    )

    with pytest.raises(
        PermissionError,
        match="Test evaluation is disabled",
    ):
        run_selected_ef_models(
            config_path="config/dev.yaml",
            version="v1",
            model_selection="b_emb",
            target_split="test",
            allow_test=False,
        )

    assert (
        split_called
        is False
    )


def test_run_selected_ef_models_rejects_invalid_target(
    monkeypatch,
):

    monkeypatch.setattr(
        ef_module,
        "load_yaml",
        lambda path: _config(),
    )

    with pytest.raises(
        ValueError,
        match="target_split",
    ):
        run_selected_ef_models(
            config_path="config/dev.yaml",
            version="v1",
            model_selection="b_emb",
            target_split="training",
        )


# ============================================================
# Runner orchestration
# ============================================================


def test_run_selected_ef_models_executes_one_model(
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
        ef_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        ef_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        ef_module,
        "resolve_seed",
        lambda config, seed: 2026,
    )

    monkeypatch.setattr(
        ef_module,
        "prepare_run_context",
        lambda **kwargs: run,
    )

    def fake_build(
        model_key,
        *,
        split,
        config,
        seed,
    ):

        calls[
            "build"
        ].append(
            model_key
        )

        return (
            SimpleNamespace(
                model_name="b-emb"
            ),
            {
                "seed": seed,
                "context_normalization": (
                    "mean"
                ),
            },
        )

    monkeypatch.setattr(
        ef_module,
        "build_ef_model",
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
            model_name="b-emb",
            family="bernoulli",
            hyperparameters=(
                spec.hyperparameters
            ),
        )

    monkeypatch.setattr(
        ef_module,
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
            experiment_id="b-exp",
            experiment_dir=(
                tmp_path
                /
                "b-exp"
            ),
        )

    monkeypatch.setattr(
        ef_module,
        "write_experiment_result",
        fake_write,
    )

    returned_run, results = (
        run_selected_ef_models(
            config_path="config/dev.yaml",
            version="v1",
            model_selection="b_emb",
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
            "b_emb",
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
            "write"
        ]
        ==
        [
            "b-emb",
        ]
    )

    assert len(
        results
    ) == 1


def test_run_selected_ef_models_all_uses_one_run_and_frozen_order(
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

    prepare_calls = []
    built = []
    written = []

    monkeypatch.setattr(
        ef_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        ef_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        ef_module,
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
        ef_module,
        "prepare_run_context",
        fake_prepare,
    )

    def fake_build(
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
                    EF_MODEL_METADATA[
                        model_key
                    ][
                        "model_name"
                    ]
                )
            ),
            {
                "seed": seed,
                "context_normalization": (
                    "mean"
                ),
            },
        )

    monkeypatch.setattr(
        ef_module,
        "build_ef_model",
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
            family=(
                spec.family
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
        ef_module,
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
                result.model_name
            ),
            experiment_dir=(
                tmp_path
                /
                "stored"
            ),
        )

    monkeypatch.setattr(
        ef_module,
        "write_experiment_result",
        fake_write,
    )

    returned_run, results = (
        run_selected_ef_models(
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
        EF_MODEL_ORDER
    )

    expected_names = [
        EF_MODEL_METADATA[
            key
        ][
            "model_name"
        ]
        for key
        in EF_MODEL_ORDER
    ]

    assert (
        written
        ==
        expected_names
    )

    assert [
        result.model_name
        for result
        in results
    ] == expected_names


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
        ef_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        ef_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        ef_module,
        "resolve_seed",
        lambda config, seed: 2026,
    )

    monkeypatch.setattr(
        ef_module,
        "prepare_run_context",
        lambda **kwargs: run,
    )

    monkeypatch.setattr(
        ef_module,
        "build_ef_model",
        lambda *args, **kwargs: (
            SimpleNamespace(
                model_name="g-emb"
            ),
            {
                "seed": 2026,
                "context_normalization": (
                    "mean"
                ),
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
            model_name="g-emb",
            family="gaussian",
            target_split="test",
            hyperparameters=(
                spec.hyperparameters
            ),
            native_metrics={
                "rmse": 1.0,
                "mae": 0.8,
            },
        )

    monkeypatch.setattr(
        ef_module,
        "run_experiment",
        fake_run_experiment,
    )

    monkeypatch.setattr(
        ef_module,
        "write_experiment_result",
        lambda **kwargs: (
            SimpleNamespace(
                experiment_id="g-test",
                experiment_dir=(
                    tmp_path
                    /
                    "g-test"
                ),
            )
        ),
    )

    _, results = (
        run_selected_ef_models(
            config_path="config/dev.yaml",
            version="v1",
            model_selection="g_emb",
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
# Shared-run handoff
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
        ef_module,
        "load_yaml",
        lambda path: _config(),
    )

    monkeypatch.setattr(
        ef_module,
        "load_frozen_split",
        lambda config, version: split,
    )

    monkeypatch.setattr(
        ef_module,
        "resolve_seed",
        lambda config, seed: 2026,
    )

    def fake_prepare(
        **kwargs,
    ):

        captured.update(
            kwargs
        )

        return run

    monkeypatch.setattr(
        ef_module,
        "prepare_run_context",
        fake_prepare,
    )

    monkeypatch.setattr(
        ef_module,
        "build_ef_model",
        lambda *args, **kwargs: (
            SimpleNamespace(
                model_name="b-emb"
            ),
            {
                "seed": 2026,
                "context_normalization": (
                    "mean"
                ),
            },
        ),
    )

    monkeypatch.setattr(
        ef_module,
        "run_experiment",
        lambda *,
        model,
        split,
        config,
        spec,
        target_split,
        allow_test: _result(
            model_name="b-emb",
            family="bernoulli",
            hyperparameters=(
                spec.hyperparameters
            ),
        ),
    )

    monkeypatch.setattr(
        ef_module,
        "write_experiment_result",
        lambda **kwargs: (
            SimpleNamespace(
                experiment_id="b-exp",
                experiment_dir=(
                    tmp_path
                    /
                    "b-exp"
                ),
            )
        ),
    )

    returned_run, _ = (
        run_selected_ef_models(
            config_path="config/dev.yaml",
            version="v1",
            model_selection="b_emb",
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
        ef_module
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
        "b_emb",
        "p_emb",
        "ap_emb",
        "g_emb",
        "ng_emb",
        "all",
    ]