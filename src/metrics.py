"""Evaluation metrics for the Paper 2 experiment pipeline.

This module provides one authoritative implementation of common ranking
metrics and continuous-prediction metrics.

Ranking conventions
-------------------
- Relevance is binary for the primary implicit-ranking task.
- ``ranked_items`` must be ordered from highest predicted score to lowest.
- Item identifiers in ``ranked_items`` must be unique.
- ``relevant_items`` represents the user's held-out relevant set.
- Metrics are computed at an explicit cutoff K.
- NDCG uses logarithmic discount with rank positions starting at 1.

Continuous metrics
------------------
RMSE and MAE are provided for Gaussian-family native diagnostics. These
metrics are not used as universal cross-family likelihood comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RankingMetrics:
    """Ranking metrics for one user at one cutoff."""

    k: int
    precision: float
    recall: float
    hit_rate: float
    ndcg: float
    mrr: float
    map: float


def _validate_k(k: int) -> int:
    """Validate a ranking cutoff."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    return k


def _normalize_ranked_items(
    ranked_items: Sequence[int] | np.ndarray,
) -> list[int]:
    """Normalize and validate a ranked item list."""
    items = [int(item) for item in ranked_items]

    if len(items) != len(set(items)):
        raise ValueError(
            "ranked_items contains duplicate item IDs"
        )

    return items


def _normalize_relevant_items(
    relevant_items: Iterable[int],
) -> set[int]:
    """Normalize a relevant-item collection."""
    return {
        int(item)
        for item in relevant_items
    }


def precision_at_k(
    ranked_items: Sequence[int] | np.ndarray,
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """Compute binary Precision@K."""
    k = _validate_k(k)
    ranked = _normalize_ranked_items(ranked_items)
    relevant = _normalize_relevant_items(relevant_items)

    top_k = ranked[:k]

    hits = sum(
        item in relevant
        for item in top_k
    )

    return float(hits / k)


def recall_at_k(
    ranked_items: Sequence[int] | np.ndarray,
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """Compute binary Recall@K."""
    k = _validate_k(k)
    ranked = _normalize_ranked_items(ranked_items)
    relevant = _normalize_relevant_items(relevant_items)

    if not relevant:
        raise ValueError(
            "Recall is undefined when relevant_items is empty"
        )

    top_k = ranked[:k]

    hits = sum(
        item in relevant
        for item in top_k
    )

    return float(
        hits / len(relevant)
    )


def hit_rate_at_k(
    ranked_items: Sequence[int] | np.ndarray,
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """Compute binary HitRate@K for one user."""
    k = _validate_k(k)
    ranked = _normalize_ranked_items(ranked_items)
    relevant = _normalize_relevant_items(relevant_items)

    if not relevant:
        raise ValueError(
            "Hit rate is undefined when relevant_items is empty"
        )

    top_k = ranked[:k]

    return float(
        any(
            item in relevant
            for item in top_k
        )
    )


def ndcg_at_k(
    ranked_items: Sequence[int] | np.ndarray,
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """Compute binary NDCG@K."""
    k = _validate_k(k)
    ranked = _normalize_ranked_items(ranked_items)
    relevant = _normalize_relevant_items(relevant_items)

    if not relevant:
        raise ValueError(
            "NDCG is undefined when relevant_items is empty"
        )

    top_k = ranked[:k]

    dcg = 0.0

    for rank, item in enumerate(
        top_k,
        start=1,
    ):
        if item in relevant:
            dcg += 1.0 / np.log2(
                rank + 1
            )

    ideal_hits = min(
        len(relevant),
        k,
    )

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    if idcg <= 0.0:
        raise RuntimeError(
            "Ideal DCG must be positive"
        )

    return float(
        dcg / idcg
    )


def mrr_at_k(
    ranked_items: Sequence[int] | np.ndarray,
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """Compute reciprocal rank of the first relevant item within K."""
    k = _validate_k(k)
    ranked = _normalize_ranked_items(ranked_items)
    relevant = _normalize_relevant_items(relevant_items)

    if not relevant:
        raise ValueError(
            "MRR is undefined when relevant_items is empty"
        )

    for rank, item in enumerate(
        ranked[:k],
        start=1,
    ):
        if item in relevant:
            return float(
                1.0 / rank
            )

    return 0.0


def average_precision_at_k(
    ranked_items: Sequence[int] | np.ndarray,
    relevant_items: Iterable[int],
    k: int,
) -> float:
    """Compute Average Precision@K for one user."""
    k = _validate_k(k)
    ranked = _normalize_ranked_items(ranked_items)
    relevant = _normalize_relevant_items(relevant_items)

    if not relevant:
        raise ValueError(
            "Average precision is undefined "
            "when relevant_items is empty"
        )

    hits = 0
    precision_sum = 0.0

    for rank, item in enumerate(
        ranked[:k],
        start=1,
    ):
        if item in relevant:
            hits += 1
            precision_sum += (
                hits / rank
            )

    denominator = min(
        len(relevant),
        k,
    )

    return float(
        precision_sum / denominator
    )


def ranking_metrics_at_k(
    ranked_items: Sequence[int] | np.ndarray,
    relevant_items: Iterable[int],
    k: int,
) -> RankingMetrics:
    """Compute all supported ranking metrics for one user."""
    return RankingMetrics(
        k=_validate_k(k),
        precision=precision_at_k(
            ranked_items,
            relevant_items,
            k,
        ),
        recall=recall_at_k(
            ranked_items,
            relevant_items,
            k,
        ),
        hit_rate=hit_rate_at_k(
            ranked_items,
            relevant_items,
            k,
        ),
        ndcg=ndcg_at_k(
            ranked_items,
            relevant_items,
            k,
        ),
        mrr=mrr_at_k(
            ranked_items,
            relevant_items,
            k,
        ),
        map=average_precision_at_k(
            ranked_items,
            relevant_items,
            k,
        ),
    )


def _validate_continuous_inputs(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate continuous prediction vectors."""
    true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError(
            "y_true and y_pred must be one-dimensional"
        )

    if len(true) == 0:
        raise ValueError(
            "y_true and y_pred must not be empty"
        )

    if len(true) != len(pred):
        raise ValueError(
            "y_true and y_pred must have equal length"
        )

    if not np.isfinite(true).all():
        raise ValueError(
            "y_true contains non-finite values"
        )

    if not np.isfinite(pred).all():
        raise ValueError(
            "y_pred contains non-finite values"
        )

    return true, pred


def rmse(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Compute root mean squared error."""
    true, pred = _validate_continuous_inputs(
        y_true,
        y_pred,
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(
                    true - pred
                )
            )
        )
    )


def mae(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Compute mean absolute error."""
    true, pred = _validate_continuous_inputs(
        y_true,
        y_pred,
    )

    return float(
        np.mean(
            np.abs(
                true - pred
            )
        )
    )