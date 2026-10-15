"""Validation and leakage checks for the Paper 2 pipeline.

This module independently validates canonical dataset splits.

The splitter is responsible for creating train/validation/test partitions.
This module is responsible for refusing partitions that violate experiment
invariants.

Checks include:
- required schema,
- missing/non-finite values,
- duplicate user-item interactions,
- exact reconstruction of the original ratings table,
- train/validation/test disjointness,
- exact per-user allocation,
- user coverage,
- train-item coverage,
- consistency with configuration metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.split import SplitResult


REQUIRED_COLUMNS = [
    "userId",
    "movieId",
    "rating",
    "timestamp",
]


class SplitValidationError(ValueError):
    """Raised when a canonical split violates an experiment invariant."""


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a successfully validated split."""

    original_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    users: int
    original_items: int
    train_items: int
    validation_items: int
    test_items: int
    cold_validation_items: int
    cold_test_items: int
    seed: int
    constrained_attempt: int


def _validate_frame(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """Validate the basic structure of a ratings DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame"
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing_columns:
        raise SplitValidationError(
            f"{name} is missing required columns: "
            f"{missing_columns}"
        )

    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise SplitValidationError(
            f"{name} contains missing values"
        )

    numeric_columns = [
        "userId",
        "movieId",
        "rating",
        "timestamp",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            raise SplitValidationError(
                f"{name}.{column} must be numeric"
            )

    numeric_values = frame[
        numeric_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_values).all():
        raise SplitValidationError(
            f"{name} contains NaN or infinite numeric values"
        )


def _validate_unique_user_item_pairs(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """Require at most one interaction per user-item pair."""
    duplicate_mask = frame.duplicated(
        subset=["userId", "movieId"],
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            frame.loc[
                duplicate_mask,
                ["userId", "movieId"],
            ]
            .head(5)
            .to_dict("records")
        )

        raise SplitValidationError(
            f"{name} contains duplicate user-item "
            f"interactions. Examples: {examples}"
        )


def _interaction_pairs(
    frame: pd.DataFrame,
) -> set[tuple[int, int]]:
    """Return user-item interaction keys."""
    return set(
        zip(
            frame["userId"].astype(int),
            frame["movieId"].astype(int),
        )
    )


def _canonicalize(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return a canonical ordering for exact comparisons."""
    return (
        frame[REQUIRED_COLUMNS]
        .sort_values(
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


def _validate_exact_reconstruction(
    original: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Ensure the three splits reconstruct the original exactly."""
    combined = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    if len(combined) != len(original):
        raise SplitValidationError(
            "Split row counts do not reconstruct the "
            "original dataset"
        )

    expected = _canonicalize(original)
    actual = _canonicalize(combined)

    try:
        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise SplitValidationError(
            "Train/validation/test do not exactly "
            "reconstruct the original ratings table"
        ) from error


def _validate_disjointness(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Ensure no user-item interaction appears in multiple splits."""
    train_pairs = _interaction_pairs(train)
    validation_pairs = _interaction_pairs(validation)
    test_pairs = _interaction_pairs(test)

    train_validation_overlap = (
        train_pairs & validation_pairs
    )

    train_test_overlap = (
        train_pairs & test_pairs
    )

    validation_test_overlap = (
        validation_pairs & test_pairs
    )

    if train_validation_overlap:
        raise SplitValidationError(
            "Leakage detected: train and validation "
            "contain overlapping user-item interactions"
        )

    if train_test_overlap:
        raise SplitValidationError(
            "Leakage detected: train and test contain "
            "overlapping user-item interactions"
        )

    if validation_test_overlap:
        raise SplitValidationError(
            "Leakage detected: validation and test "
            "contain overlapping user-item interactions"
        )


def _validate_per_user_counts(
    original: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    """Validate the exact expected allocation for every user."""
    original_counts = (
        original.groupby("userId")
        .size()
        .sort_index()
    )

    train_counts = (
        train.groupby("userId")
        .size()
        .reindex(
            original_counts.index,
            fill_value=0,
        )
    )

    validation_counts = (
        validation.groupby("userId")
        .size()
        .reindex(
            original_counts.index,
            fill_value=0,
        )
    )

    test_counts = (
        test.groupby("userId")
        .size()
        .reindex(
            original_counts.index,
            fill_value=0,
        )
    )

    expected_validation = np.floor(
        original_counts.to_numpy()
        * validation_fraction
    ).astype(int)

    expected_test = np.floor(
        original_counts.to_numpy()
        * test_fraction
    ).astype(int)

    expected_train = (
        original_counts.to_numpy()
        - expected_validation
        - expected_test
    )

    if not np.array_equal(
        train_counts.to_numpy(),
        expected_train,
    ):
        raise SplitValidationError(
            "Per-user training counts do not match "
            "the configured split protocol"
        )

    if not np.array_equal(
        validation_counts.to_numpy(),
        expected_validation,
    ):
        raise SplitValidationError(
            "Per-user validation counts do not match "
            "the configured split protocol"
        )

    if not np.array_equal(
        test_counts.to_numpy(),
        expected_test,
    ):
        raise SplitValidationError(
            "Per-user test counts do not match "
            "the configured split protocol"
        )

    if np.any(expected_train <= 0):
        raise SplitValidationError(
            "At least one user has no expected "
            "training interactions"
        )


def _validate_user_coverage(
    original: pd.DataFrame,
    train: pd.DataFrame,
) -> None:
    """Require every original user to appear in training."""
    original_users = set(
        original["userId"].astype(int)
    )

    train_users = set(
        train["userId"].astype(int)
    )

    missing_train_users = (
        original_users - train_users
    )

    if missing_train_users:
        raise SplitValidationError(
            "Users missing from training split: "
            f"{len(missing_train_users)}"
        )


def _validate_train_item_coverage(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[int, int]:
    """Require every validation/test item to occur in training."""
    train_items = set(
        train["movieId"].astype(int)
    )

    validation_items = set(
        validation["movieId"].astype(int)
    )

    test_items = set(
        test["movieId"].astype(int)
    )

    cold_validation = (
        validation_items - train_items
    )

    cold_test = (
        test_items - train_items
    )

    if cold_validation:
        raise SplitValidationError(
            "Cold-start leakage/protocol violation: "
            f"{len(cold_validation)} validation items "
            "are absent from training"
        )

    if cold_test:
        raise SplitValidationError(
            "Cold-start leakage/protocol violation: "
            f"{len(cold_test)} test items "
            "are absent from training"
        )

    return (
        len(cold_validation),
        len(cold_test),
    )


def validate_split(
    original: pd.DataFrame,
    split: SplitResult,
    config: dict[str, Any],
) -> ValidationReport:
    """Run all canonical split validation checks."""
    train = split.train
    validation = split.validation
    test = split.test

    for frame, name in [
        (original, "original"),
        (train, "train"),
        (validation, "validation"),
        (test, "test"),
    ]:
        _validate_frame(
            frame=frame,
            name=name,
        )

        _validate_unique_user_item_pairs(
            frame=frame,
            name=name,
        )

    split_config = config["split"]
    preprocessing_config = config[
        "preprocessing"
    ]

    expected_seed = int(
        split_config["seed"]
    )

    if split.seed != expected_seed:
        raise SplitValidationError(
            "Split seed does not match configuration: "
            f"expected {expected_seed}, got {split.seed}"
        )

    expected_coverage = bool(
        preprocessing_config[
            "enforce_train_item_coverage"
        ]
    )

    if (
        split.enforce_train_item_coverage
        != expected_coverage
    ):
        raise SplitValidationError(
            "Split item-coverage setting does not "
            "match configuration"
        )

    _validate_disjointness(
        train=train,
        validation=validation,
        test=test,
    )

    _validate_exact_reconstruction(
        original=original,
        train=train,
        validation=validation,
        test=test,
    )

    _validate_per_user_counts(
        original=original,
        train=train,
        validation=validation,
        test=test,
        train_fraction=float(
            split_config["train_fraction"]
        ),
        validation_fraction=float(
            split_config["validation_fraction"]
        ),
        test_fraction=float(
            split_config["test_fraction"]
        ),
    )

    _validate_user_coverage(
        original=original,
        train=train,
    )

    cold_validation_items = 0
    cold_test_items = 0

    if expected_coverage:
        (
            cold_validation_items,
            cold_test_items,
        ) = _validate_train_item_coverage(
            train=train,
            validation=validation,
            test=test,
        )

    return ValidationReport(
        original_rows=len(original),
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        users=original["userId"].nunique(),
        original_items=original[
            "movieId"
        ].nunique(),
        train_items=train[
            "movieId"
        ].nunique(),
        validation_items=validation[
            "movieId"
        ].nunique(),
        test_items=test[
            "movieId"
        ].nunique(),
        cold_validation_items=(
            cold_validation_items
        ),
        cold_test_items=(
            cold_test_items
        ),
        seed=split.seed,
        constrained_attempt=split.attempt,
    )