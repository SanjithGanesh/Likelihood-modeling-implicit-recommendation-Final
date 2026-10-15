"""Tests for src.evaluation.

These tests verify the full-catalog evaluation protocol, deterministic
tie-breaking, candidate exclusions, metric aggregation, and test-set lock.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    deterministic_rank,
    evaluate_from_config,
    evaluate_ranking_split,
)
from src.split import (
    SplitResult,
    split_from_config,
)


def _simple_popularity_scorer(
    train: pd.DataFrame,
):
    """Create a deterministic train-only popularity scorer."""
    popularity = (
        train.groupby("movieId")
        .size()
        .to_dict()
    )

    def scorer(
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [
                popularity[int(item_id)]
                for item_id in candidate_items
            ],
            dtype=np.float64,
        )

    return scorer


def test_deterministic_rank_orders_by_score_then_item_id() -> None:
    """Ties must be broken using ascending item ID."""
    items = np.array(
        [50, 10, 30, 20, 40],
        dtype=np.int64,
    )

    scores = np.array(
        [0.5, 0.9, 0.5, 0.9, 0.5],
        dtype=np.float64,
    )

    ranked = deterministic_rank(
        items,
        scores,
    )

    expected = np.array(
        [10, 20, 30, 40, 50],
        dtype=np.int64,
    )

    assert np.array_equal(
        ranked,
        expected,
    )


def test_deterministic_rank_rejects_duplicate_items() -> None:
    """Candidate item IDs must be unique."""
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        deterministic_rank(
            np.array([1, 1, 2]),
            np.array([0.3, 0.2, 0.1]),
        )


def test_deterministic_rank_rejects_non_finite_scores() -> None:
    """NaN or infinite model scores must fail."""
    with pytest.raises(
        ValueError,
        match="NaN|infinite",
    ):
        deterministic_rank(
            np.array([1, 2, 3]),
            np.array([0.1, np.nan, 0.3]),
        )


def test_validation_full_catalog_evaluation(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Validation evaluation must run over the training-item catalog."""
    split = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    scorer = _simple_popularity_scorer(
        split.train
    )

    result = evaluate_from_config(
        split=split,
        config=synthetic_config,
        score_function=scorer,
        target_split="validation",
    )

    assert (
        result.target_split
        == "validation"
    )

    assert (
        result.catalog_size
        == split.train["movieId"].nunique()
    )

    assert result.users_evaluated == 4
    assert len(result.per_user) == 4

    assert (
        result.per_user["userId"].nunique()
        == 4
    )

    for metric_name, value in (
        result.aggregate.items()
    ):
        assert np.isfinite(value)

        assert 0.0 <= value <= 1.0, (
            metric_name,
            value,
        )


def test_validation_candidates_exclude_training_items(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Validation candidate sets must exclude the user's training history."""
    split = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    captured: dict[int, np.ndarray] = {}

    def scorer(
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        captured[user_id] = (
            candidate_items.copy()
        )

        return np.zeros(
            len(candidate_items),
            dtype=np.float64,
        )

    evaluate_from_config(
        split=split,
        config=synthetic_config,
        score_function=scorer,
        target_split="validation",
    )

    for user_id in range(1, 5):
        training_items = set(
            split.train.loc[
                split.train["userId"]
                == user_id,
                "movieId",
            ]
            .astype(int)
            .tolist()
        )

        candidates = set(
            captured[user_id]
            .astype(int)
            .tolist()
        )

        assert not (
            training_items
            & candidates
        )


def test_validation_relevant_items_remain_candidates(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Validation positives must remain in the user's candidate set."""
    split = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    captured: dict[int, np.ndarray] = {}

    def scorer(
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        captured[user_id] = (
            candidate_items.copy()
        )

        return np.zeros(
            len(candidate_items),
            dtype=np.float64,
        )

    evaluate_from_config(
        split=split,
        config=synthetic_config,
        score_function=scorer,
        target_split="validation",
    )

    for user_id in range(1, 5):
        relevant = set(
            split.validation.loc[
                split.validation["userId"]
                == user_id,
                "movieId",
            ]
            .astype(int)
            .tolist()
        )

        candidates = set(
            captured[user_id]
            .astype(int)
            .tolist()
        )

        assert relevant <= candidates


def test_test_evaluation_is_locked_by_default(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Test evaluation must not be available during tuning."""
    split = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    scorer = _simple_popularity_scorer(
        split.train
    )

    with pytest.raises(
        PermissionError,
        match="Test evaluation is locked",
    ):
        evaluate_from_config(
            split=split,
            config=synthetic_config,
            score_function=scorer,
            target_split="test",
        )


def test_authorized_test_excludes_train_and_validation() -> None:
    """Authorized test evaluation must exclude train and validation history."""
    train = pd.DataFrame(
        [
            [1, 1, 5.0, 1],
            [1, 2, 5.0, 2],
            [2, 3, 5.0, 3],
            [2, 4, 5.0, 4],
            [3, 5, 5.0, 5],
            [3, 6, 5.0, 6],
        ],
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
    )

    validation = pd.DataFrame(
        [
            [1, 3, 5.0, 7],
        ],
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
    )

    test = pd.DataFrame(
        [
            [1, 4, 5.0, 8],
        ],
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
    )

    split = SplitResult(
        train=train,
        validation=validation,
        test=test,
        seed=2026,
        attempt=1,
        enforce_train_item_coverage=True,
    )

    config = {
        "evaluation": {
            "ranking_protocol": "full_catalog",
            "exclude_training_items": True,
            "exclude_validation_items_from_test_candidates": True,
            "k_values": [
                1,
                2,
            ],
            "test_access_during_tuning": False,
        },
    }

    captured: dict[int, np.ndarray] = {}

    def scorer(
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        captured[user_id] = (
            candidate_items.copy()
        )

        return candidate_items.astype(
            np.float64
        )

    result = evaluate_from_config(
        split=split,
        config=config,
        score_function=scorer,
        target_split="test",
        allow_test=True,
    )

    candidates = set(
        captured[1]
        .astype(int)
        .tolist()
    )

    assert candidates == {
        4,
        5,
        6,
    }

    assert 1 not in candidates
    assert 2 not in candidates
    assert 3 not in candidates

    assert 4 in candidates

    assert result.users_evaluated == 1
    assert result.catalog_size == 6


def test_evaluation_rejects_relevant_item_missing_from_catalog() -> None:
    """Held-out relevant items absent from train must fail loudly."""
    train = pd.DataFrame(
        [
            [1, 1, 5.0, 1],
            [2, 2, 5.0, 2],
        ],
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
    )

    validation = pd.DataFrame(
        [
            [1, 99, 5.0, 3],
        ],
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
    )

    test = validation.iloc[0:0].copy()

    def scorer(
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(
            len(candidate_items),
            dtype=np.float64,
        )

    with pytest.raises(
        ValueError,
        match="absent from the training catalog",
    ):
        evaluate_ranking_split(
            train=train,
            validation=validation,
            test=test,
            target_split="validation",
            score_function=scorer,
            k_values=[1],
        )


def test_evaluation_rejects_wrong_score_length(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """A scorer returning the wrong number of scores must fail."""
    split = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    def bad_scorer(
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(
            len(candidate_items) + 1,
            dtype=np.float64,
        )

    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        evaluate_from_config(
            split=split,
            config=synthetic_config,
            score_function=bad_scorer,
            target_split="validation",
        )


def test_evaluation_rejects_non_full_catalog_protocol(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """The primary evaluation wrapper must refuse sampled protocols."""
    split = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    bad_config = copy.deepcopy(
        synthetic_config
    )

    bad_config[
        "evaluation"
    ]["ranking_protocol"] = "sampled"

    scorer = _simple_popularity_scorer(
        split.train
    )

    with pytest.raises(
        ValueError,
        match="full_catalog",
    ):
        evaluate_from_config(
            split=split,
            config=bad_config,
            score_function=scorer,
            target_split="validation",
        )