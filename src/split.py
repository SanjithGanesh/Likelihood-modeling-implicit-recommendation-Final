"""Canonical dataset splitting utilities for Paper 2.

This module provides the authoritative train/validation/test splitting logic.

Primary publication protocol
----------------------------
- Per-user random splitting.
- Deterministic from a fixed seed.
- Train/validation/test fractions controlled by configuration.
- Validation and test remain disjoint from training.
- Every item appearing in validation or test must also appear in training
  when train-item coverage enforcement is enabled.
- User-level split counts are preserved exactly whenever the constrained
  split is feasible.

The canonical split produced from this logic will later be saved once and
reused by every model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.reproducibility import make_numpy_rng


@dataclass(frozen=True)
class SplitResult:
    """Container for one canonical train/validation/test split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    seed: int
    attempt: int
    enforce_train_item_coverage: bool


def _validate_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    """Validate train/validation/test fractions."""
    fractions = {
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    }

    for name, value in fractions.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1"
            )

    total = (
        float(train_fraction)
        + float(validation_fraction)
        + float(test_fraction)
    )

    if not np.isclose(total, 1.0):
        raise ValueError(
            "Train/validation/test fractions must sum to 1.0; "
            f"got {total}"
        )

    if train_fraction <= 0:
        raise ValueError(
            "train_fraction must be greater than zero"
        )


def _prepare_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Create a stable canonical working copy of the ratings table."""
    required_columns = [
        "userId",
        "movieId",
        "rating",
        "timestamp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in ratings.columns
    ]

    if missing:
        raise ValueError(
            f"Ratings data is missing required columns: {missing}"
        )

    if len(ratings) == 0:
        raise ValueError("Ratings data is empty")

    working = ratings[required_columns].copy()

    if working.isna().any().any():
        raise ValueError(
            "Ratings data contains missing values"
        )

    # Stable sorting makes the split independent of the input row order.
    working = working.sort_values(
        by=[
            "userId",
            "movieId",
            "timestamp",
            "rating",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return working


def _user_split_counts(
    n_interactions: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    """Calculate deterministic per-user split counts."""
    n_validation = int(
        np.floor(n_interactions * validation_fraction)
    )

    n_test = int(
        np.floor(n_interactions * test_fraction)
    )

    n_train = (
        n_interactions
        - n_validation
        - n_test
    )

    if n_train <= 0:
        raise ValueError(
            "User would have no training interactions. "
            f"Interactions={n_interactions}, "
            f"fractions=({train_fraction}, "
            f"{validation_fraction}, {test_fraction})"
        )

    return n_train, n_validation, n_test


def _split_without_item_constraint(
    ratings: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Perform a deterministic per-user random split."""
    assignment = np.full(
        len(ratings),
        "train",
        dtype=object,
    )

    for user_id, user_rows in ratings.groupby(
        "userId",
        sort=True,
    ):
        indices = user_rows.index.to_numpy()

        _, n_validation, n_test = _user_split_counts(
            n_interactions=len(indices),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )

        rng = make_numpy_rng(
            seed,
            f"split:user:{int(user_id)}",
        )

        shuffled = indices[
            rng.permutation(len(indices))
        ]

        validation_indices = shuffled[
            :n_validation
        ]

        test_indices = shuffled[
            n_validation:n_validation + n_test
        ]

        assignment[validation_indices] = "validation"
        assignment[test_indices] = "test"

    return assignment, 1


def _split_with_item_coverage(
    ratings: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    max_attempts: int,
) -> tuple[np.ndarray, int]:
    """Split while guaranteeing that every evaluated item exists in train.

    All interactions initially belong to training. Interactions are moved
    into validation/test only when doing so leaves at least one training
    interaction for that item.

    Different deterministic user-processing orders are attempted if a
    particular greedy ordering cannot satisfy all requested user holdouts.
    """
    if max_attempts < 1:
        raise ValueError(
            "max_attempts must be at least 1"
        )

    user_groups = {
        int(user_id): group.index.to_numpy()
        for user_id, group in ratings.groupby(
            "userId",
            sort=True,
        )
    }

    user_ids = np.array(
        sorted(user_groups.keys()),
        dtype=np.int64,
    )

    initial_item_counts = (
        ratings["movieId"]
        .value_counts()
        .astype("int64")
        .to_dict()
    )

    last_failure: str | None = None

    for attempt in range(1, max_attempts + 1):
        assignment = np.full(
            len(ratings),
            "train",
            dtype=object,
        )

        remaining_train_counts = dict(
            initial_item_counts
        )

        user_order_rng = make_numpy_rng(
            seed,
            f"split:user_order:attempt:{attempt}",
        )

        user_order = user_ids[
            user_order_rng.permutation(len(user_ids))
        ]

        attempt_failed = False

        for user_id in user_order:
            indices = user_groups[int(user_id)]

            _, n_validation, n_test = _user_split_counts(
                n_interactions=len(indices),
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
            )

            n_holdout = n_validation + n_test

            if n_holdout == 0:
                continue

            user_rng = make_numpy_rng(
                seed,
                (
                    f"split:user:{int(user_id)}:"
                    f"attempt:{attempt}"
                ),
            )

            shuffled = indices[
                user_rng.permutation(len(indices))
            ]

            selected: list[int] = []

            for row_index in shuffled:
                item_id = int(
                    ratings.at[row_index, "movieId"]
                )

                # Never remove the final training occurrence
                # of an item.
                if remaining_train_counts[item_id] <= 1:
                    continue

                selected.append(int(row_index))
                remaining_train_counts[item_id] -= 1

                if len(selected) == n_holdout:
                    break

            if len(selected) != n_holdout:
                last_failure = (
                    f"Could not allocate {n_holdout} "
                    f"holdout interactions for user "
                    f"{int(user_id)} while preserving "
                    "at least one training occurrence "
                    "for every item."
                )

                attempt_failed = True
                break

            selected_array = np.asarray(
                selected,
                dtype=np.int64,
            )

            partition_rng = make_numpy_rng(
                seed,
                (
                    f"split:partition:{int(user_id)}:"
                    f"attempt:{attempt}"
                ),
            )

            selected_array = selected_array[
                partition_rng.permutation(
                    len(selected_array)
                )
            ]

            validation_indices = selected_array[
                :n_validation
            ]

            test_indices = selected_array[
                n_validation:n_validation + n_test
            ]

            assignment[
                validation_indices
            ] = "validation"

            assignment[
                test_indices
            ] = "test"

        if not attempt_failed:
            return assignment, attempt

    raise RuntimeError(
        "Unable to construct a constrained canonical split "
        f"after {max_attempts} deterministic attempts. "
        f"Last failure: {last_failure}"
    )


def _canonicalize_output(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return a stable, clean split DataFrame."""
    return (
        frame[
            [
                "userId",
                "movieId",
                "rating",
                "timestamp",
            ]
        ]
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


def split_ratings(
    ratings: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    enforce_train_item_coverage: bool = True,
    max_attempts: int = 32,
) -> SplitResult:
    """Create the canonical deterministic per-user random split."""
    _validate_fractions(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    working = _prepare_ratings(ratings)

    if enforce_train_item_coverage:
        assignment, attempt = _split_with_item_coverage(
            ratings=working,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            max_attempts=max_attempts,
        )
    else:
        assignment, attempt = _split_without_item_constraint(
            ratings=working,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    train = _canonicalize_output(
        working.loc[assignment == "train"]
    )

    validation = _canonicalize_output(
        working.loc[assignment == "validation"]
    )

    test = _canonicalize_output(
        working.loc[assignment == "test"]
    )

    if len(train) + len(validation) + len(test) != len(
        working
    ):
        raise RuntimeError(
            "Split row counts do not reconstruct the "
            "original ratings table"
        )

    if enforce_train_item_coverage:
        train_items = set(
            train["movieId"].unique()
        )

        validation_items = set(
            validation["movieId"].unique()
        )

        test_items = set(
            test["movieId"].unique()
        )

        unseen_validation = (
            validation_items - train_items
        )

        unseen_test = (
            test_items - train_items
        )

        if unseen_validation or unseen_test:
            raise RuntimeError(
                "Train-item coverage invariant failed"
            )

    return SplitResult(
        train=train,
        validation=validation,
        test=test,
        seed=seed,
        attempt=attempt,
        enforce_train_item_coverage=(
            enforce_train_item_coverage
        ),
    )


def split_from_config(
    ratings: pd.DataFrame,
    config: dict[str, Any],
) -> SplitResult:
    """Create a canonical split using an experiment configuration."""
    split_config = config["split"]
    preprocessing_config = config["preprocessing"]

    strategy = split_config["strategy"]

    if strategy != "per_user_random":
        raise ValueError(
            "Unsupported split strategy: "
            f"{strategy}"
        )

    return split_ratings(
        ratings=ratings,
        train_fraction=float(
            split_config["train_fraction"]
        ),
        validation_fraction=float(
            split_config["validation_fraction"]
        ),
        test_fraction=float(
            split_config["test_fraction"]
        ),
        seed=int(split_config["seed"]),
        enforce_train_item_coverage=bool(
            preprocessing_config[
                "enforce_train_item_coverage"
            ]
        ),
    )