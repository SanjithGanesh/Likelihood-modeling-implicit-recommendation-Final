"""Full-catalog ranking evaluation for the Paper 2 pipeline.

This module provides the authoritative evaluation protocol used across models.

Primary protocol
----------------
- Evaluate against the complete training-item catalog.
- Never use sampled evaluation negatives for primary results.
- Validation candidates exclude the user's training interactions.
- Test candidates exclude the user's training interactions and, when
  configured, validation interactions.
- Validation evaluation does not inspect test interactions.
- Ties are broken deterministically by ascending item ID.
- Test evaluation is locked by default and requires explicit authorization.

A model-facing scorer follows this contract:

    score_function(user_id, candidate_items) -> scores

where ``candidate_items`` is a one-dimensional NumPy array of item IDs and
the returned scores have the same length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from src.metrics import ranking_metrics_at_k
from src.split import SplitResult


TargetSplit = Literal[
    "validation",
    "test",
]

ScoreFunction = Callable[
    [int, np.ndarray],
    np.ndarray,
]


@dataclass(frozen=True)
class EvaluationResult:
    """Results from one full-catalog ranking evaluation."""

    target_split: TargetSplit
    catalog_size: int
    users_evaluated: int
    per_user: pd.DataFrame
    aggregate: dict[str, float]


def _validate_frame(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """Validate the minimum schema required for ranking evaluation."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame"
        )

    required = [
        "userId",
        "movieId",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{missing}"
        )

    if frame[required].isna().any().any():
        raise ValueError(
            f"{name} contains missing user/item IDs"
        )


def _validate_k_values(
    k_values: list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    """Validate and normalize ranking cutoffs."""
    if not isinstance(
        k_values,
        (list, tuple),
    ):
        raise TypeError(
            "k_values must be a list or tuple"
        )

    if len(k_values) == 0:
        raise ValueError(
            "k_values must not be empty"
        )

    normalized: list[int] = []

    for k in k_values:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(
                "Every K value must be an integer"
            )

        if k <= 0:
            raise ValueError(
                "Every K value must be greater than zero"
            )

        normalized.append(k)

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "k_values must not contain duplicates"
        )

    return tuple(
        sorted(normalized)
    )


def _build_user_item_sets(
    frame: pd.DataFrame,
) -> dict[int, frozenset[int]]:
    """Build immutable user-to-item sets."""
    result: dict[int, frozenset[int]] = {}

    for user_id, group in frame.groupby(
        "userId",
        sort=True,
    ):
        result[int(user_id)] = frozenset(
            group["movieId"]
            .astype(int)
            .tolist()
        )

    return result


def deterministic_rank(
    candidate_items: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Rank candidate items by score with deterministic tie-breaking.

    Primary ordering:
        Higher model score first.

    Tie-breaking:
        Lower item ID first.

    The tie-break rule depends only on the candidate item ID and never on
    relevance labels.
    """
    candidate_items = np.asarray(
        candidate_items,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if candidate_items.ndim != 1:
        raise ValueError(
            "candidate_items must be one-dimensional"
        )

    if scores.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional"
        )

    if len(candidate_items) != len(scores):
        raise ValueError(
            "candidate_items and scores must have equal length"
        )

    if len(candidate_items) == 0:
        raise ValueError(
            "candidate_items must not be empty"
        )

    if len(candidate_items) != len(
        np.unique(candidate_items)
    ):
        raise ValueError(
            "candidate_items contains duplicate item IDs"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "scores contains NaN or infinite values"
        )

    # np.lexsort uses the final key as the primary key.
    #
    # Primary: -scores -> descending model score.
    # Secondary: candidate item ID -> ascending ID.
    order = np.lexsort(
        (
            candidate_items,
            -scores,
        )
    )

    return candidate_items[order]


def _candidate_items(
    catalog: np.ndarray,
    excluded_items: set[int],
) -> np.ndarray:
    """Create a deterministic candidate catalog after exclusions."""
    if not excluded_items:
        return catalog.copy()

    mask = np.fromiter(
        (
            int(item_id) not in excluded_items
            for item_id in catalog
        ),
        dtype=bool,
        count=len(catalog),
    )

    return catalog[mask]


def evaluate_ranking_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_split: TargetSplit,
    score_function: ScoreFunction,
    k_values: list[int] | tuple[int, ...],
    exclude_validation_from_test: bool = True,
) -> EvaluationResult:
    """Evaluate one split using full-catalog ranking."""
    for frame, name in [
        (train, "train"),
        (validation, "validation"),
        (test, "test"),
    ]:
        _validate_frame(
            frame=frame,
            name=name,
        )

    if target_split not in (
        "validation",
        "test",
    ):
        raise ValueError(
            "target_split must be 'validation' or 'test'"
        )

    if not callable(score_function):
        raise TypeError(
            "score_function must be callable"
        )

    normalized_k_values = _validate_k_values(
        k_values
    )

    max_k = max(
        normalized_k_values
    )

    catalog = np.sort(
        train["movieId"]
        .astype("int64")
        .unique()
    )

    if len(catalog) == 0:
        raise ValueError(
            "Training catalog is empty"
        )

    train_by_user = _build_user_item_sets(
        train
    )

    validation_by_user = _build_user_item_sets(
        validation
    )

    test_by_user = _build_user_item_sets(
        test
    )

    if target_split == "validation":
        target_by_user = validation_by_user
    else:
        target_by_user = test_by_user

    if not target_by_user:
        raise ValueError(
            f"{target_split} contains no users"
        )

    catalog_set = set(
        catalog.astype(int).tolist()
    )

    rows: list[dict[str, float | int]] = []

    for user_id in sorted(
        target_by_user.keys()
    ):
        relevant = set(
            target_by_user[user_id]
        )

        if not relevant:
            raise ValueError(
                f"User {user_id} has no relevant "
                f"items in {target_split}"
            )

        missing_from_catalog = (
            relevant - catalog_set
        )

        if missing_from_catalog:
            raise ValueError(
                f"User {user_id} has relevant "
                "items absent from the training catalog: "
                f"{sorted(missing_from_catalog)[:5]}"
            )

        excluded = set(
            train_by_user.get(
                user_id,
                frozenset(),
            )
        )

        if (
            target_split == "test"
            and exclude_validation_from_test
        ):
            excluded.update(
                validation_by_user.get(
                    user_id,
                    frozenset(),
                )
            )

        overlap = (
            relevant & excluded
        )

        if overlap:
            raise ValueError(
                f"Relevant {target_split} items were "
                f"incorrectly excluded for user {user_id}: "
                f"{sorted(overlap)[:5]}"
            )

        candidates = _candidate_items(
            catalog=catalog,
            excluded_items=excluded,
        )

        candidate_set = set(
            candidates.astype(int).tolist()
        )

        missing_relevant = (
            relevant - candidate_set
        )

        if missing_relevant:
            raise RuntimeError(
                f"User {user_id} has relevant items "
                "missing from the final candidate set: "
                f"{sorted(missing_relevant)[:5]}"
            )

        if len(candidates) < max_k:
            raise ValueError(
                f"User {user_id} has only "
                f"{len(candidates)} candidates, "
                f"fewer than max K={max_k}"
            )

        raw_scores = score_function(
            int(user_id),
            candidates.copy(),
        )

        scores = np.asarray(
            raw_scores,
            dtype=np.float64,
        )

        ranked = deterministic_rank(
            candidate_items=candidates,
            scores=scores,
        )

        # Metric calculations only need the largest requested prefix.
        top_ranked = ranked[
            :max_k
        ]

        row: dict[str, float | int] = {
            "userId": int(user_id),
            "n_relevant": len(relevant),
            "n_candidates": len(candidates),
        }

        for k in normalized_k_values:
            metrics = ranking_metrics_at_k(
                ranked_items=top_ranked,
                relevant_items=relevant,
                k=k,
            )

            row[f"precision_at_{k}"] = (
                metrics.precision
            )

            row[f"recall_at_{k}"] = (
                metrics.recall
            )

            row[f"hit_rate_at_{k}"] = (
                metrics.hit_rate
            )

            row[f"ndcg_at_{k}"] = (
                metrics.ndcg
            )

            row[f"mrr_at_{k}"] = (
                metrics.mrr
            )

            row[f"map_at_{k}"] = (
                metrics.map
            )

        rows.append(row)

    per_user = pd.DataFrame(rows)

    metric_columns = [
        column
        for column in per_user.columns
        if column not in {
            "userId",
            "n_relevant",
            "n_candidates",
        }
    ]

    aggregate = {
        column: float(
            per_user[column].mean()
        )
        for column in metric_columns
    }

    return EvaluationResult(
        target_split=target_split,
        catalog_size=len(catalog),
        users_evaluated=len(per_user),
        per_user=per_user,
        aggregate=aggregate,
    )


def evaluate_from_config(
    split: SplitResult,
    config: dict[str, Any],
    score_function: ScoreFunction,
    target_split: TargetSplit = "validation",
    allow_test: bool = False,
) -> EvaluationResult:
    """Evaluate a canonical split according to experiment configuration.

    Test evaluation is intentionally blocked unless ``allow_test=True``.
    Later experiment orchestration will only provide that authorization
    after model selection and hyperparameter freezing.
    """
    evaluation_config = config[
        "evaluation"
    ]

    protocol = evaluation_config[
        "ranking_protocol"
    ]

    if protocol != "full_catalog":
        raise ValueError(
            "Primary evaluation protocol must be "
            f"'full_catalog', got '{protocol}'"
        )

    if not bool(
        evaluation_config[
            "exclude_training_items"
        ]
    ):
        raise ValueError(
            "Primary evaluation requires training "
            "items to be excluded from candidates"
        )

    if (
        target_split == "test"
        and not allow_test
    ):
        raise PermissionError(
            "Test evaluation is locked. "
            "Explicit final-evaluation authorization "
            "is required."
        )

    k_values = [
        int(k)
        for k in evaluation_config[
            "k_values"
        ]
    ]

    exclude_validation_from_test = bool(
        evaluation_config[
            "exclude_validation_items_from_test_candidates"
        ]
    )

    return evaluate_ranking_split(
        train=split.train,
        validation=split.validation,
        test=split.test,
        target_split=target_split,
        score_function=score_function,
        k_values=k_values,
        exclude_validation_from_test=(
            exclude_validation_from_test
        ),
    )