"""Tests for src.metrics.

These tests verify ranking metrics and continuous prediction metrics using
hand-calculated examples plus important edge and failure cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    mae,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    ranking_metrics_at_k,
    recall_at_k,
    rmse,
)


def test_hand_calculated_ranking_metrics() -> None:
    """Ranking metrics must match a manually calculated example."""
    ranked = [4, 1, 2, 3, 5]
    relevant = {2, 4, 6}
    k = 5

    expected_precision = 2 / 5
    expected_recall = 2 / 3
    expected_hit_rate = 1.0
    expected_mrr = 1.0

    expected_map = (
        1.0
        + (2.0 / 3.0)
    ) / 3.0

    expected_dcg = (
        1.0 / math.log2(2)
        + 1.0 / math.log2(4)
    )

    expected_idcg = (
        1.0 / math.log2(2)
        + 1.0 / math.log2(3)
        + 1.0 / math.log2(4)
    )

    expected_ndcg = (
        expected_dcg
        / expected_idcg
    )

    assert np.isclose(
        precision_at_k(
            ranked,
            relevant,
            k,
        ),
        expected_precision,
    )

    assert np.isclose(
        recall_at_k(
            ranked,
            relevant,
            k,
        ),
        expected_recall,
    )

    assert np.isclose(
        hit_rate_at_k(
            ranked,
            relevant,
            k,
        ),
        expected_hit_rate,
    )

    assert np.isclose(
        ndcg_at_k(
            ranked,
            relevant,
            k,
        ),
        expected_ndcg,
    )

    assert np.isclose(
        mrr_at_k(
            ranked,
            relevant,
            k,
        ),
        expected_mrr,
    )

    assert np.isclose(
        average_precision_at_k(
            ranked,
            relevant,
            k,
        ),
        expected_map,
    )


def test_ranking_metrics_wrapper() -> None:
    """The aggregate wrapper must return the same component metrics."""
    ranked = [4, 1, 2, 3, 5]
    relevant = {2, 4, 6}

    result = ranking_metrics_at_k(
        ranked,
        relevant,
        5,
    )

    assert result.k == 5

    assert np.isclose(
        result.precision,
        0.4,
    )

    assert np.isclose(
        result.recall,
        2 / 3,
    )

    assert np.isclose(
        result.hit_rate,
        1.0,
    )

    assert np.isclose(
        result.mrr,
        1.0,
    )


def test_perfect_ranking_metrics() -> None:
    """A perfect top-K ranking should score 1 on every metric."""
    result = ranking_metrics_at_k(
        ranked_items=[10, 20, 30],
        relevant_items={10, 20},
        k=2,
    )

    assert np.isclose(
        result.precision,
        1.0,
    )

    assert np.isclose(
        result.recall,
        1.0,
    )

    assert np.isclose(
        result.hit_rate,
        1.0,
    )

    assert np.isclose(
        result.ndcg,
        1.0,
    )

    assert np.isclose(
        result.mrr,
        1.0,
    )

    assert np.isclose(
        result.map,
        1.0,
    )


def test_no_hit_metrics_are_zero() -> None:
    """No relevant items inside K should produce zero ranking metrics."""
    ranked = [1, 2, 3]
    relevant = {10, 20}

    result = ranking_metrics_at_k(
        ranked,
        relevant,
        3,
    )

    assert np.isclose(
        result.precision,
        0.0,
    )

    assert np.isclose(
        result.recall,
        0.0,
    )

    assert np.isclose(
        result.hit_rate,
        0.0,
    )

    assert np.isclose(
        result.ndcg,
        0.0,
    )

    assert np.isclose(
        result.mrr,
        0.0,
    )

    assert np.isclose(
        result.map,
        0.0,
    )


def test_duplicate_ranked_items_are_rejected() -> None:
    """A ranked list may not contain duplicate item IDs."""
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        precision_at_k(
            [1, 1, 2],
            {1, 2},
            3,
        )


@pytest.mark.parametrize(
    "metric_function",
    [
        recall_at_k,
        hit_rate_at_k,
        ndcg_at_k,
        mrr_at_k,
        average_precision_at_k,
    ],
)
def test_empty_relevant_set_is_rejected(
    metric_function,
) -> None:
    """Metrics with an undefined relevance denominator must reject empties."""
    with pytest.raises(
        ValueError,
    ):
        metric_function(
            [1, 2, 3],
            set(),
            3,
        )


def test_precision_with_empty_relevant_set_is_zero() -> None:
    """Precision remains defined as zero when no retrieved item is relevant."""
    assert np.isclose(
        precision_at_k(
            [1, 2, 3],
            set(),
            3,
        ),
        0.0,
    )


@pytest.mark.parametrize(
    "invalid_k",
    [
        0,
        -1,
    ],
)
def test_invalid_numeric_k_is_rejected(
    invalid_k: int,
) -> None:
    """Non-positive cutoffs must fail."""
    with pytest.raises(
        ValueError,
    ):
        ndcg_at_k(
            [1, 2, 3],
            {1},
            invalid_k,
        )


def test_boolean_k_is_rejected() -> None:
    """Boolean values must not silently behave as integer cutoffs."""
    with pytest.raises(
        TypeError,
    ):
        ndcg_at_k(
            [1, 2, 3],
            {1},
            True,
        )


def test_hand_calculated_rmse_and_mae() -> None:
    """Continuous metrics must match manually calculated values."""
    y_true = [
        1.0,
        2.0,
        3.0,
    ]

    y_pred = [
        1.0,
        3.0,
        2.0,
    ]

    expected_rmse = math.sqrt(
        2.0 / 3.0
    )

    expected_mae = (
        2.0 / 3.0
    )

    assert np.isclose(
        rmse(
            y_true,
            y_pred,
        ),
        expected_rmse,
    )

    assert np.isclose(
        mae(
            y_true,
            y_pred,
        ),
        expected_mae,
    )


def test_perfect_continuous_predictions_are_zero_error() -> None:
    """Perfect continuous predictions must have zero RMSE and MAE."""
    values = [
        1.0,
        2.0,
        3.0,
    ]

    assert np.isclose(
        rmse(
            values,
            values,
        ),
        0.0,
    )

    assert np.isclose(
        mae(
            values,
            values,
        ),
        0.0,
    )


def test_rmse_rejects_length_mismatch() -> None:
    """RMSE inputs must have equal length."""
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        rmse(
            [1.0, 2.0],
            [1.0],
        )


def test_mae_rejects_non_finite_input() -> None:
    """Continuous metrics must reject NaN or infinite values."""
    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        mae(
            [1.0, np.nan],
            [1.0, 2.0],
        )