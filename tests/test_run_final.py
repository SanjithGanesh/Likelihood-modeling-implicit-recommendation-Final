"""Tests for scripts.run_final.

These tests do not evaluate the real MovieLens test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_final as final
from src.evaluation import EvaluationResult
from src.experiment import ExperimentSpec
from src.models.baselines import PopularityBaseline
from src.split import SplitResult


def _canonical_split() -> SplitResult:
    train = pd.DataFrame(
        {
            "userId": [
                1, 1,
                2, 2,
                3, 3,
                4, 4,
            ],
            "movieId": [
                10, 20,
                10, 30,
                20, 40,
                30, 40,
            ],
            "rating": [
                4.0, 3.5,
                5.0, 2.5,
                4.5, 3.0,
                2.0, 4.0,
            ],
        }
    )

    validation = pd.DataFrame(
        {
            "userId": [
                1, 2, 3, 4,
            ],
            "movieId": [
                30, 40, 10, 20,
            ],
            "rating": [
                4.5, 3.0, 5.0, 3.5,
            ],
        }
    )

    test = pd.DataFrame(
        {
            "userId": [
                1, 2, 3, 4,
            ],
            "movieId": [
                40, 20, 30, 10,
            ],
            "rating": [
                5.0, 4.0, 2.5, 4.5,
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


def _frozen_eval_for_synthetic() -> dict:
    return {
        "freeze_id": "paper2-final-evaluation-v1",
        "training_protocol": {
            "train_rows": 8,
            "validation_rows": 4,
            "final_fit_rows": 12,
        },
        "evaluation_protocol": {
            "test_rows": 4,
        },
        "seed_protocol": {
            "final_seeds": [
                2026,
                2027,
                2028,
            ],
        },
    }


def _minimal_frozen_hp() -> dict:
    return {
        "freeze_id": "paper2-final-hyperparameters-v1",
        "tuned_models": {
            "PCA/SVD": {
                "hyperparameters": {
                    "n_components": 16,
                    "n_iter": 7,
                },
            },
            "BernoulliMF": {
                "hyperparameters": {
                    "n_factors": 32,
                    "device": "cpu",
                },
            },
        },
        "untuned_baselines": {
            "Popularity": {},
            "Random": {},
        },
    }


def test_final_fit_split_moves_validation_into_training() -> None:
    canonical = _canonical_split()

    fit_split = (
        final.build_final_fit_split(
            canonical,
            _frozen_eval_for_synthetic(),
        )
    )

    assert len(
        fit_split.train
    ) == 12

    assert len(
        fit_split.validation
    ) == 0

    assert len(
        fit_split.test
    ) == 4

    expected = pd.concat(
        [
            canonical.train,
            canonical.validation,
        ],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(
        fit_split.train.reset_index(
            drop=True
        ),
        expected.reset_index(
            drop=True
        ),
    )

    pd.testing.assert_frame_equal(
        fit_split.test.reset_index(
            drop=True
        ),
        canonical.test.reset_index(
            drop=True
        ),
    )


def test_final_fit_split_keeps_test_as_only_heldout_for_model_builders() -> None:
    canonical = _canonical_split()

    fit_split = (
        final.build_final_fit_split(
            canonical,
            _frozen_eval_for_synthetic(),
        )
    )

    heldout = (
        final.tune._heldout_interactions(
            fit_split
        )
    )

    pd.testing.assert_frame_equal(
        heldout.reset_index(
            drop=True
        ),
        canonical.test.reset_index(
            drop=True
        ),
    )

    all_observed = (
        final.tune._all_observed(
            fit_split
        )
    )

    assert len(
        all_observed
    ) == 16

    assert not all_observed.duplicated(
        subset=[
            "userId",
            "movieId",
        ]
    ).any()


def test_final_fit_catalog_remains_canonical_train_catalog() -> None:
    canonical = _canonical_split()

    fit_split = (
        final.build_final_fit_split(
            canonical,
            _frozen_eval_for_synthetic(),
        )
    )

    canonical_catalog = set(
        canonical.train[
            "movieId"
        ]
    )

    final_catalog = set(
        fit_split.train[
            "movieId"
        ]
    )

    assert (
        final_catalog
        ==
        canonical_catalog
    )


def test_final_seed_replaces_selection_seed_only() -> None:
    frozen_hp = (
        _minimal_frozen_hp()
    )

    hp_2026 = (
        final.materialize_final_hyperparameters(
            model_name="PCA/SVD",
            seed=2026,
            frozen_hyperparameters=(
                frozen_hp
            ),
        )
    )

    hp_2028 = (
        final.materialize_final_hyperparameters(
            model_name="PCA/SVD",
            seed=2028,
            frozen_hyperparameters=(
                frozen_hp
            ),
        )
    )

    assert hp_2026 == {
        "n_components": 16,
        "n_iter": 7,
        "seed": 2026,
    }

    assert hp_2028 == {
        "n_components": 16,
        "n_iter": 7,
        "seed": 2028,
    }

    without_seed_2026 = dict(
        hp_2026
    )

    without_seed_2028 = dict(
        hp_2028
    )

    without_seed_2026.pop(
        "seed"
    )

    without_seed_2028.pop(
        "seed"
    )

    assert (
        without_seed_2026
        ==
        without_seed_2028
    )


def test_popularity_and_random_final_hyperparameters() -> None:
    frozen_hp = (
        _minimal_frozen_hp()
    )

    assert (
        final.materialize_final_hyperparameters(
            model_name="Popularity",
            seed=2027,
            frozen_hyperparameters=(
                frozen_hp
            ),
        )
        ==
        {}
    )

    assert (
        final.materialize_final_hyperparameters(
            model_name="Random",
            seed=2027,
            frozen_hyperparameters=(
                frozen_hp
            ),
        )
        ==
        {
            "seed": 2027,
            "namespace": "random_baseline",
        }
    )


def test_final_execution_gate_fails_before_state_preparation(
    monkeypatch,
) -> None:
    called = False

    def fake_prepare(**kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "must not be reached"
        )

    monkeypatch.setattr(
        final,
        "_prepare_final_state",
        fake_prepare,
    )

    with pytest.raises(
        PermissionError
    ):
        final.execute_final_protocol(
            authorize_test=False
        )

    assert called is False


def test_run_one_fits_final_train_but_evaluates_canonical_split(
    monkeypatch,
) -> None:
    canonical = _canonical_split()

    fit_split = (
        final.build_final_fit_split(
            canonical,
            _frozen_eval_for_synthetic(),
        )
    )

    model = PopularityBaseline()

    spec = ExperimentSpec(
        model_name="Popularity",
        family="baseline",
        variant="primary",
        seed=2026,
        hyperparameters={},
    )

    observed = {}

    def fake_evaluate_from_config(
        *,
        split,
        config,
        score_function,
        target_split,
        allow_test,
    ):
        observed[
            "split"
        ] = split

        observed[
            "target_split"
        ] = target_split

        observed[
            "allow_test"
        ] = allow_test

        return EvaluationResult(
            target_split="test",
            catalog_size=4,
            users_evaluated=4,
            per_user=pd.DataFrame(
                {
                    "userId": [
                        1, 2, 3, 4,
                    ],
                    "ndcg_at_10": [
                        0.1, 0.2, 0.3, 0.4,
                    ],
                }
            ),
            aggregate={
                "ndcg_at_10": 0.25,
            },
        )

    monkeypatch.setattr(
        final,
        "evaluate_from_config",
        fake_evaluate_from_config,
    )

    result = (
        final.run_one_final_experiment(
            model=model,
            spec=spec,
            fit_split=fit_split,
            canonical_split=canonical,
            config={
                "evaluation": {},
            },
            frozen_hyperparameters={
                "freeze_id": (
                    "paper2-final-hyperparameters-v1"
                ),
                "tuned_models": {},
            },
            frozen_evaluation={
                "freeze_id": (
                    "paper2-final-evaluation-v1"
                ),
            },
            authorize_test=True,
        )
    )

    assert model.is_fitted

    # Validation interactions must have contributed to final popularity counts.
    assert model.item_counts[
        30
    ] == 3

    assert (
        observed[
            "split"
        ]
        is
        canonical
    )

    assert (
        observed[
            "target_split"
        ]
        ==
        "test"
    )

    assert (
        observed[
            "allow_test"
        ]
        is True
    )

    assert (
        result.target_split
        ==
        "test"
    )

    assert (
        result.metadata[
            "train_rows"
        ]
        ==
        12
    )

    assert (
        result.metadata[
            "validation_rows"
        ]
        ==
        0
    )

    assert (
        result.metadata[
            "canonical_validation_rows"
        ]
        ==
        4
    )


def test_run_one_rejects_test_without_explicit_authorization() -> None:
    canonical = _canonical_split()

    fit_split = (
        final.build_final_fit_split(
            canonical,
            _frozen_eval_for_synthetic(),
        )
    )

    model = PopularityBaseline()

    spec = ExperimentSpec(
        model_name="Popularity",
        family="baseline",
        variant="primary",
        seed=2026,
        hyperparameters={},
    )

    with pytest.raises(
        PermissionError
    ):
        final.run_one_final_experiment(
            model=model,
            spec=spec,
            fit_split=fit_split,
            canonical_split=canonical,
            config={},
            frozen_hyperparameters={
                "freeze_id": (
                    "paper2-final-hyperparameters-v1"
                ),
                "tuned_models": {},
            },
            frozen_evaluation={
                "freeze_id": (
                    "paper2-final-evaluation-v1"
                ),
            },
            authorize_test=False,
        )

    assert model.is_fitted is False


def test_real_frozen_control_files_match_step_55_and_56a_hashes() -> None:
    (
        _config,
        frozen_hp,
        frozen_eval,
    ) = final.verify_frozen_controls()

    assert (
        frozen_hp[
            "freeze_id"
        ]
        ==
        "paper2-final-hyperparameters-v1"
    )

    assert (
        frozen_eval[
            "freeze_id"
        ]
        ==
        "paper2-final-evaluation-v1"
    )

    assert (
        frozen_eval[
            "training_protocol"
        ][
            "final_fit_rows"
        ]
        ==
        902826
    )

    assert (
        frozen_eval[
            "evaluation_protocol"
        ][
            "test_rows"
        ]
        ==
        97383
    )

    assert (
        frozen_eval[
            "seed_protocol"
        ][
            "final_seeds"
        ]
        ==
        [
            2026,
            2027,
            2028,
        ]
    )
