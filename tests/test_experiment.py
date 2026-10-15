"""Permanent tests for the Paper 2 unified experiment layer."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import src.experiment as experiment_module
from src.evaluation import EvaluationResult
from src.experiment import (
    ExperimentResult,
    ExperimentSpec,
    run_experiment,
)
from src.split import SplitResult


# ============================================================
# Synthetic split
# ============================================================


@pytest.fixture
def synthetic_split() -> SplitResult:
    """Tiny nonempty canonical-style split for orchestration tests."""

    train = pd.DataFrame(
        {
            "userId": [
                1,
                2,
                3,
            ],
            "movieId": [
                10,
                20,
                30,
            ],
            "rating": [
                5.0,
                4.0,
                3.0,
            ],
        }
    )

    validation = pd.DataFrame(
        {
            "userId": [
                1,
                2,
                3,
            ],
            "movieId": [
                20,
                30,
                10,
            ],
            "rating": [
                4.0,
                3.5,
                4.5,
            ],
        }
    )

    test = pd.DataFrame(
        {
            "userId": [
                1,
                2,
                3,
            ],
            "movieId": [
                30,
                10,
                20,
            ],
            "rating": [
                3.5,
                5.0,
                4.0,
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


# ============================================================
# Dummy model implementations
# ============================================================


class DummyModel:
    """Minimal Paper 2 model contract for experiment tests."""

    model_name = "DummyModel"

    def __init__(
        self,
        *,
        seed: int = 2026,
    ) -> None:

        self.seed = seed

        self.fit_calls = 0

        self.fit_frame = None

        self.training_history = [
            {
                "epoch": 1,
                "loss": 0.25,
            }
        ]

    def fit(
        self,
        train_data: pd.DataFrame,
    ):
        self.fit_calls += 1

        self.fit_frame = train_data

        return self

    def score(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:

        items = np.asarray(
            candidate_items,
            dtype=np.float64,
        )

        return (
            items
            +
            float(
                user_id
            )
            * 1e-6
        )


class NativeDummyModel(
    DummyModel
):
    """Dummy model exposing family-native diagnostics."""

    model_name = "NativeDummyModel"

    def __init__(
        self,
        *,
        seed: int = 2026,
    ) -> None:

        super().__init__(
            seed=seed
        )

        self.native_frame = None

    def native_metrics(
        self,
        evaluation_frame: pd.DataFrame,
    ):
        self.native_frame = (
            evaluation_frame
        )

        return {
            "rmse": np.float64(
                0.5
            ),
            "mae": np.float32(
                0.4
            ),
        }


class FailingFitModel(
    DummyModel
):
    """Model whose fit failure must propagate."""

    model_name = "FailingFitModel"

    def fit(
        self,
        train_data: pd.DataFrame,
    ):
        self.fit_calls += 1

        raise RuntimeError(
            "intentional fit failure"
        )


# ============================================================
# Helpers
# ============================================================


def _spec(
    *,
    model_name: str = "DummyModel",
    family: str = "synthetic",
    seed: int = 2026,
    variant: str = "primary",
) -> ExperimentSpec:

    return ExperimentSpec(
        model_name=model_name,
        family=family,
        seed=seed,
        variant=variant,
        hyperparameters={
            "n_factors": 2,
            "learning_rate": 0.01,
        },
    )


def _evaluation_result(
    *,
    target_split: str = "validation",
) -> EvaluationResult:

    per_user = pd.DataFrame(
        {
            "userId": [
                1,
                2,
                3,
            ],
            "ndcg_at_10": [
                0.5,
                0.4,
                0.3,
            ],
        }
    )

    return EvaluationResult(
        target_split=target_split,
        catalog_size=3,
        users_evaluated=3,
        per_user=per_user,
        aggregate={
            "ndcg_at_10": np.float64(
                0.4
            ),
            "recall_at_10": np.float32(
                0.5
            ),
        },
    )


def _install_fake_evaluator(
    monkeypatch,
    *,
    returned_target: str = "validation",
):
    """Install a deterministic evaluator and return its call record."""

    calls = []

    def fake_evaluate_from_config(
        *,
        split,
        config,
        score_function,
        target_split="validation",
        allow_test=False,
    ):
        calls.append(
            {
                "split": split,
                "config": config,
                "score_function": (
                    score_function
                ),
                "target_split": (
                    target_split
                ),
                "allow_test": (
                    allow_test
                ),
            }
        )

        return _evaluation_result(
            target_split=(
                returned_target
            )
        )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_from_config",
        fake_evaluate_from_config,
    )

    return calls


# ============================================================
# ExperimentSpec
# ============================================================


def test_experiment_spec_defensively_copies_hyperparameters():
    original = {
        "n_factors": 4,
        "nested": {
            "alpha": 1.0,
        },
    }

    spec = ExperimentSpec(
        model_name="DummyModel",
        family="synthetic",
        seed=2026,
        hyperparameters=original,
    )

    original[
        "n_factors"
    ] = 999

    original[
        "nested"
    ][
        "alpha"
    ] = 999.0

    assert (
        spec.hyperparameters[
            "n_factors"
        ]
        ==
        4
    )

    assert (
        spec.hyperparameters[
            "nested"
        ][
            "alpha"
        ]
        ==
        1.0
    )


@pytest.mark.parametrize(
    (
        "kwargs",
        "exception_type",
    ),
    [
        (
            {
                "model_name": "",
                "family": "synthetic",
                "seed": 2026,
            },
            ValueError,
        ),
        (
            {
                "model_name": "DummyModel",
                "family": "",
                "seed": 2026,
            },
            ValueError,
        ),
        (
            {
                "model_name": "DummyModel",
                "family": "synthetic",
                "seed": 2026,
                "variant": "",
            },
            ValueError,
        ),
        (
            {
                "model_name": "DummyModel",
                "family": "synthetic",
                "seed": True,
            },
            TypeError,
        ),
        (
            {
                "model_name": "DummyModel",
                "family": "synthetic",
                "seed": 2026,
                "hyperparameters": [
                    1,
                    2,
                ],
            },
            TypeError,
        ),
    ],
)
def test_experiment_spec_rejects_invalid_values(
    kwargs,
    exception_type,
):
    with pytest.raises(
        exception_type
    ):
        ExperimentSpec(
            **kwargs
        )


# ============================================================
# Validation run
# ============================================================


def test_run_experiment_defaults_to_validation_and_fits_train_only(
    synthetic_split,
    monkeypatch,
):
    calls = (
        _install_fake_evaluator(
            monkeypatch,
            returned_target="validation",
        )
    )

    model = DummyModel()

    config = {
        "evaluation": {
            "k_values": [
                5,
                10,
                20,
            ],
        }
    }

    result = run_experiment(
        model=model,
        split=synthetic_split,
        config=config,
        spec=_spec(),
    )

    # Exactly one fit.
    assert model.fit_calls == 1

    # Fitting receives the canonical train DataFrame,
    # not validation or test.
    assert (
        model.fit_frame
        is synthetic_split.train
    )

    # Shared evaluator called exactly once.
    assert len(
        calls
    ) == 1

    call = calls[
        0
    ]

    assert (
        call[
            "split"
        ]
        is synthetic_split
    )

    assert (
        call[
            "config"
        ]
        is config
    )

    assert (
        call[
            "target_split"
        ]
        ==
        "validation"
    )

    assert (
        call[
            "allow_test"
        ]
        is False
    )

    assert (
        call[
            "score_function"
        ].__self__
        is model
    )

    # Structured result.
    assert (
        result.model_name
        ==
        "DummyModel"
    )

    assert (
        result.family
        ==
        "synthetic"
    )

    assert (
        result.variant
        ==
        "primary"
    )

    assert (
        result.seed
        ==
        2026
    )

    assert (
        result.target_split
        ==
        "validation"
    )

    assert (
        result.status
        ==
        "completed"
    )

    assert (
        result.error_message
        is None
    )

    assert (
        result.native_metrics
        ==
        {}
    )

    assert (
        result.ranking_metrics
        ==
        {
            "ndcg_at_10": 0.4,
            "recall_at_10": 0.5,
        }
    )

    assert (
        result.runtime_seconds
        >=
        0.0
    )

    assert np.isfinite(
        result.runtime_seconds
    )


# ============================================================
# Metadata
# ============================================================


def test_run_experiment_records_structured_metadata(
    synthetic_split,
    monkeypatch,
):
    _install_fake_evaluator(
        monkeypatch,
        returned_target="validation",
    )

    model = DummyModel()

    result = run_experiment(
        model=model,
        split=synthetic_split,
        config={},
        spec=_spec(),
    )

    metadata = (
        result.metadata
    )

    assert metadata[
        "model_class"
    ].endswith(
        ".DummyModel"
    )

    assert metadata[
        "hyperparameters"
    ] == {
        "n_factors": 2,
        "learning_rate": 0.01,
    }

    assert (
        metadata[
            "catalog_size"
        ]
        ==
        3
    )

    assert (
        metadata[
            "users_evaluated"
        ]
        ==
        3
    )

    assert (
        metadata[
            "per_user_metric_rows"
        ]
        ==
        3
    )

    assert (
        metadata[
            "split_seed"
        ]
        ==
        2026
    )

    assert (
        metadata[
            "split_attempt"
        ]
        ==
        1
    )

    assert (
        metadata[
            "enforce_train_item_coverage"
        ]
        is True
    )

    assert (
        metadata[
            "train_rows"
        ]
        ==
        3
    )

    assert (
        metadata[
            "validation_rows"
        ]
        ==
        3
    )

    assert (
        metadata[
            "test_rows"
        ]
        ==
        3
    )


# ============================================================
# Native diagnostics
# ============================================================


def test_native_metrics_receive_validation_frame(
    synthetic_split,
    monkeypatch,
):
    _install_fake_evaluator(
        monkeypatch,
        returned_target="validation",
    )

    model = NativeDummyModel()

    result = run_experiment(
        model=model,
        split=synthetic_split,
        config={},
        spec=_spec(
            model_name=(
                "NativeDummyModel"
            ),
            family="gaussian",
        ),
    )

    assert (
        model.native_frame
        is synthetic_split.validation
    )

    assert result.native_metrics == {
        "rmse": 0.5,
        "mae": pytest.approx(
            float(
                np.float32(
                    0.4
                )
            )
        ),
    }


# ============================================================
# Test-access gate / B75 / B89
# ============================================================


def test_unauthorized_test_access_is_blocked_before_fit(
    synthetic_split,
    monkeypatch,
):
    evaluator_called = False

    def forbidden_evaluator(
        **kwargs,
    ):
        nonlocal evaluator_called

        evaluator_called = True

        raise AssertionError(
            "Evaluator must not be called"
        )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_from_config",
        forbidden_evaluator,
    )

    model = DummyModel()

    with pytest.raises(
        PermissionError,
        match="Test evaluation is disabled",
    ):
        run_experiment(
            model=model,
            split=synthetic_split,
            config={},
            spec=_spec(),
            target_split="test",
            allow_test=False,
        )

    assert (
        model.fit_calls
        ==
        0
    )

    assert (
        evaluator_called
        is False
    )


def test_authorized_test_run_uses_test_frame(
    synthetic_split,
    monkeypatch,
):
    calls = []

    def fake_evaluator(
        *,
        split,
        config,
        score_function,
        target_split="validation",
        allow_test=False,
    ):
        calls.append(
            {
                "target_split": (
                    target_split
                ),
                "allow_test": (
                    allow_test
                ),
            }
        )

        return _evaluation_result(
            target_split="test"
        )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_from_config",
        fake_evaluator,
    )

    model = NativeDummyModel()

    result = run_experiment(
        model=model,
        split=synthetic_split,
        config={},
        spec=_spec(
            model_name=(
                "NativeDummyModel"
            ),
            family="gaussian",
        ),
        target_split="test",
        allow_test=True,
    )

    assert (
        model.fit_calls
        ==
        1
    )

    assert (
        model.fit_frame
        is synthetic_split.train
    )

    assert (
        model.native_frame
        is synthetic_split.test
    )

    assert (
        result.target_split
        ==
        "test"
    )

    assert calls == [
        {
            "target_split": "test",
            "allow_test": True,
        }
    ]


# ============================================================
# Identity protection
# ============================================================


def test_model_name_mismatch_fails_before_fit(
    synthetic_split,
):
    model = DummyModel()

    spec = _spec(
        model_name="WrongName"
    )

    with pytest.raises(
        ValueError,
        match="model_name",
    ):
        run_experiment(
            model=model,
            split=synthetic_split,
            config={},
            spec=spec,
        )

    assert (
        model.fit_calls
        ==
        0
    )


def test_model_seed_mismatch_fails_before_fit(
    synthetic_split,
):
    model = DummyModel(
        seed=123
    )

    spec = _spec(
        seed=2026
    )

    with pytest.raises(
        ValueError,
        match="seed",
    ):
        run_experiment(
            model=model,
            split=synthetic_split,
            config={},
            spec=spec,
        )

    assert (
        model.fit_calls
        ==
        0
    )


# ============================================================
# Evaluator integrity
# ============================================================


def test_evaluator_target_mismatch_is_rejected(
    synthetic_split,
    monkeypatch,
):
    _install_fake_evaluator(
        monkeypatch,
        returned_target="test",
    )

    model = DummyModel()

    with pytest.raises(
        RuntimeError,
        match="different target_split",
    ):
        run_experiment(
            model=model,
            split=synthetic_split,
            config={},
            spec=_spec(),
            target_split="validation",
            allow_test=False,
        )


def test_nonfinite_ranking_metric_is_rejected(
    synthetic_split,
    monkeypatch,
):
    def bad_evaluator(
        **kwargs,
    ):
        return EvaluationResult(
            target_split="validation",
            catalog_size=3,
            users_evaluated=3,
            per_user=pd.DataFrame(
                {
                    "userId": [
                        1,
                        2,
                        3,
                    ],
                }
            ),
            aggregate={
                "ndcg_at_10": np.nan,
            },
        )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_from_config",
        bad_evaluator,
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        run_experiment(
            model=DummyModel(),
            split=synthetic_split,
            config={},
            spec=_spec(),
        )


# ============================================================
# Fail-loud behavior
# ============================================================


def test_fit_exception_propagates(
    synthetic_split,
    monkeypatch,
):
    evaluator_called = False

    def evaluator(
        **kwargs,
    ):
        nonlocal evaluator_called

        evaluator_called = True

        return _evaluation_result()

    monkeypatch.setattr(
        experiment_module,
        "evaluate_from_config",
        evaluator,
    )

    model = FailingFitModel()

    with pytest.raises(
        RuntimeError,
        match="intentional fit failure",
    ):
        run_experiment(
            model=model,
            split=synthetic_split,
            config={},
            spec=_spec(
                model_name=(
                    "FailingFitModel"
                )
            ),
        )

    assert (
        model.fit_calls
        ==
        1
    )

    assert (
        evaluator_called
        is False
    )


def test_evaluation_exception_propagates(
    synthetic_split,
    monkeypatch,
):
    def failing_evaluator(
        **kwargs,
    ):
        raise RuntimeError(
            "intentional evaluation failure"
        )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_from_config",
        failing_evaluator,
    )

    model = DummyModel()

    with pytest.raises(
        RuntimeError,
        match=(
            "intentional evaluation failure"
        ),
    ):
        run_experiment(
            model=model,
            split=synthetic_split,
            config={},
            spec=_spec(),
        )

    # Fit occurred first, but the failure was not swallowed.
    assert (
        model.fit_calls
        ==
        1
    )


# ============================================================
# Defensive result copies
# ============================================================


def test_training_history_is_defensively_copied(
    synthetic_split,
    monkeypatch,
):
    _install_fake_evaluator(
        monkeypatch,
        returned_target="validation",
    )

    model = DummyModel()

    result = run_experiment(
        model=model,
        split=synthetic_split,
        config={},
        spec=_spec(),
    )

    assert (
        result.training_history[
            0
        ][
            "loss"
        ]
        ==
        0.25
    )

    model.training_history[
        0
    ][
        "loss"
    ] = 999.0

    assert (
        result.training_history[
            0
        ][
            "loss"
        ]
        ==
        0.25
    )


def test_experiment_result_to_dict_is_defensive(
    synthetic_split,
    monkeypatch,
):
    _install_fake_evaluator(
        monkeypatch,
        returned_target="validation",
    )

    result = run_experiment(
        model=DummyModel(),
        split=synthetic_split,
        config={},
        spec=_spec(),
    )

    payload = (
        result.to_dict()
    )

    payload[
        "ranking_metrics"
    ][
        "ndcg_at_10"
    ] = 999.0

    payload[
        "training_history"
    ][
        0
    ][
        "loss"
    ] = 999.0

    payload[
        "metadata"
    ][
        "hyperparameters"
    ][
        "n_factors"
    ] = 999

    assert (
        result.ranking_metrics[
            "ndcg_at_10"
        ]
        ==
        0.4
    )

    assert (
        result.training_history[
            0
        ][
            "loss"
        ]
        ==
        0.25
    )

    assert (
        result.metadata[
            "hyperparameters"
        ][
            "n_factors"
        ]
        ==
        2
    )


# ============================================================
# ExperimentResult invariants
# ============================================================


def test_completed_result_cannot_contain_error_message():
    with pytest.raises(
        ValueError,
        match="error_message",
    ):
        ExperimentResult(
            model_name="DummyModel",
            family="synthetic",
            variant="primary",
            seed=2026,
            target_split="validation",
            ranking_metrics={},
            native_metrics={},
            training_history=[],
            runtime_seconds=0.1,
            status="completed",
            error_message="should not exist",
            metadata={},
        )


def test_failed_result_requires_error_message():
    with pytest.raises(
        ValueError,
        match="error_message",
    ):
        ExperimentResult(
            model_name="DummyModel",
            family="synthetic",
            variant="primary",
            seed=2026,
            target_split="validation",
            ranking_metrics={},
            native_metrics={},
            training_history=[],
            runtime_seconds=0.1,
            status="failed",
            error_message=None,
            metadata={},
        )


@pytest.mark.parametrize(
    "runtime",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_result_rejects_invalid_runtime(
    runtime,
):
    with pytest.raises(
        ValueError,
        match="runtime_seconds",
    ):
        ExperimentResult(
            model_name="DummyModel",
            family="synthetic",
            variant="primary",
            seed=2026,
            target_split="validation",
            ranking_metrics={},
            native_metrics={},
            training_history=[],
            runtime_seconds=runtime,
            status="completed",
            error_message=None,
            metadata={},
        )