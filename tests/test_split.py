"""Tests for src.split.

These tests verify the canonical deterministic splitting protocol using the
tiny synthetic ratings fixture defined in tests/conftest.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.split import (
    split_from_config,
    split_ratings,
)


def _canonicalize(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return a deterministic ordering for DataFrame comparisons."""
    return (
        frame.sort_values(
            by=[
                "userId",
                "movieId",
                "timestamp",
                "rating",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def test_split_has_expected_sizes(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Each 10-interaction user should receive an 8/1/1 split."""
    result = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    assert len(result.train) == 32
    assert len(result.validation) == 4
    assert len(result.test) == 4

    assert (
        len(result.train)
        + len(result.validation)
        + len(result.test)
        == 40
    )


def test_split_has_exact_per_user_counts(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Every synthetic user must receive exactly 8/1/1 interactions."""
    result = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    train_counts = (
        result.train.groupby("userId")
        .size()
    )

    validation_counts = (
        result.validation.groupby("userId")
        .size()
    )

    test_counts = (
        result.test.groupby("userId")
        .size()
    )

    assert (train_counts == 8).all()
    assert (validation_counts == 1).all()
    assert (test_counts == 1).all()

    assert set(train_counts.index) == {
        1,
        2,
        3,
        4,
    }

    assert set(validation_counts.index) == {
        1,
        2,
        3,
        4,
    }

    assert set(test_counts.index) == {
        1,
        2,
        3,
        4,
    }


def test_split_is_deterministic(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Repeated splitting with the same seed must be identical."""
    first = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    second = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    pd.testing.assert_frame_equal(
        first.train,
        second.train,
    )

    pd.testing.assert_frame_equal(
        first.validation,
        second.validation,
    )

    pd.testing.assert_frame_equal(
        first.test,
        second.test,
    )

    assert first.seed == second.seed
    assert first.attempt == second.attempt


def test_split_is_independent_of_input_row_order(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Reordering raw rows must not change the canonical split."""
    shuffled_input = (
        synthetic_ratings.sample(
            frac=1.0,
            random_state=999,
        )
        .reset_index(drop=True)
    )

    original_result = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    shuffled_result = split_from_config(
        shuffled_input,
        synthetic_config,
    )

    pd.testing.assert_frame_equal(
        original_result.train,
        shuffled_result.train,
    )

    pd.testing.assert_frame_equal(
        original_result.validation,
        shuffled_result.validation,
    )

    pd.testing.assert_frame_equal(
        original_result.test,
        shuffled_result.test,
    )


def test_split_exactly_reconstructs_original(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Train + validation + test must equal the original interactions."""
    result = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    reconstructed = pd.concat(
        [
            result.train,
            result.validation,
            result.test,
        ],
        ignore_index=True,
    )

    expected = _canonicalize(
        synthetic_ratings
    )

    actual = _canonicalize(
        reconstructed
    )

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=True,
        check_exact=True,
    )


def test_split_preserves_train_item_coverage(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Every validation/test item must also occur in training."""
    result = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    original_items = set(
        synthetic_ratings["movieId"]
        .astype(int)
    )

    train_items = set(
        result.train["movieId"]
        .astype(int)
    )

    validation_items = set(
        result.validation["movieId"]
        .astype(int)
    )

    test_items = set(
        result.test["movieId"]
        .astype(int)
    )

    assert train_items == original_items
    assert validation_items <= train_items
    assert test_items <= train_items


def test_split_partitions_are_disjoint(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """No user-item interaction may appear in multiple partitions."""
    result = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    def pairs(
        frame: pd.DataFrame,
    ) -> set[tuple[int, int]]:
        return set(
            zip(
                frame["userId"].astype(int),
                frame["movieId"].astype(int),
            )
        )

    train_pairs = pairs(result.train)
    validation_pairs = pairs(
        result.validation
    )
    test_pairs = pairs(result.test)

    assert not (
        train_pairs
        & validation_pairs
    )

    assert not (
        train_pairs
        & test_pairs
    )

    assert not (
        validation_pairs
        & test_pairs
    )


def test_split_rejects_invalid_fraction_sum(
    synthetic_ratings: pd.DataFrame,
) -> None:
    """Fractions that do not sum to one must fail."""
    with pytest.raises(
        ValueError,
        match="must sum to 1.0",
    ):
        split_ratings(
            ratings=synthetic_ratings,
            train_fraction=0.80,
            validation_fraction=0.10,
            test_fraction=0.20,
            seed=2026,
        )


def test_split_rejects_empty_ratings() -> None:
    """An empty ratings table must fail loudly."""
    empty = pd.DataFrame(
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        split_ratings(
            ratings=empty,
            train_fraction=0.80,
            validation_fraction=0.10,
            test_fraction=0.10,
            seed=2026,
        )