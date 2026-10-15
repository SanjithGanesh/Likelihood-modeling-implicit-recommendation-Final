"""Tests for src.sampling.

These tests verify that negative sampling is deterministic, leakage-safe,
and based only on the training catalog and training-derived popularity
weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sampling import NegativeSampler
from src.split import split_from_config


def _build_sampler(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> tuple[NegativeSampler, object]:
    """Build the canonical synthetic split and negative sampler."""
    split = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    sampler = NegativeSampler(
        train=split.train,
        all_observed=synthetic_ratings,
        seed=synthetic_config[
            "reproducibility"
        ]["global_seed"],
    )

    return sampler, split


def test_available_items_exclude_all_observed_interactions(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Every legal negative pool must exclude the user's full history."""
    sampler, _ = _build_sampler(
        synthetic_ratings,
        synthetic_config,
    )

    full_catalog = set(
        sampler.item_universe
        .astype(int)
        .tolist()
    )

    for user_id in range(1, 5):
        observed = set(
            synthetic_ratings.loc[
                synthetic_ratings[
                    "userId"
                ] == user_id,
                "movieId",
            ]
            .astype(int)
            .tolist()
        )

        available = set(
            sampler.available_items(
                user_id
            )
            .astype(int)
            .tolist()
        )

        expected = (
            full_catalog - observed
        )

        assert available == expected
        assert not (
            available & observed
        )

        # The synthetic fixture was deliberately
        # designed with exactly two legal negatives
        # per user.
        assert len(available) == 2


def test_uniform_sampling_never_returns_observed_items(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Uniform negatives must never overlap any observed interaction."""
    sampler, _ = _build_sampler(
        synthetic_ratings,
        synthetic_config,
    )

    for user_id in range(1, 5):
        observed = set(
            sampler.observed_items(
                user_id
            )
        )

        sample = sampler.sample(
            user_id=user_id,
            n=100,
            strategy="uniform",
            replace=True,
            draw_key="uniform_test",
        )

        assert len(sample.items) == 100

        assert all(
            int(item) not in observed
            for item in sample.items
        )


def test_popularity_sampling_never_returns_observed_items(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Popularity negatives must also exclude the user's full history."""
    sampler, _ = _build_sampler(
        synthetic_ratings,
        synthetic_config,
    )

    for user_id in range(1, 5):
        observed = set(
            sampler.observed_items(
                user_id
            )
        )

        sample = sampler.sample(
            user_id=user_id,
            n=100,
            strategy="popularity",
            replace=True,
            draw_key="popularity_test",
        )

        assert len(sample.items) == 100

        assert all(
            int(item) not in observed
            for item in sample.items
        )


def test_same_draw_key_is_deterministic(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Identical sampling requests must reproduce exactly."""
    sampler, _ = _build_sampler(
        synthetic_ratings,
        synthetic_config,
    )

    first = sampler.sample(
        user_id=1,
        n=50,
        strategy="uniform",
        replace=True,
        draw_key="repeat",
    )

    second = sampler.sample(
        user_id=1,
        n=50,
        strategy="uniform",
        replace=True,
        draw_key="repeat",
    )

    assert np.array_equal(
        first.items,
        second.items,
    )


def test_sampling_without_replacement_returns_unique_legal_items(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Sampling without replacement must return unique legal negatives."""
    sampler, _ = _build_sampler(
        synthetic_ratings,
        synthetic_config,
    )

    available = sampler.available_items(
        1
    )

    sample = sampler.sample(
        user_id=1,
        n=len(available),
        strategy="uniform",
        replace=False,
        draw_key="no_replacement",
    )

    assert len(sample.items) == 2

    assert len(
        np.unique(sample.items)
    ) == 2

    assert set(
        sample.items.astype(int)
    ) == set(
        available.astype(int)
    )


def test_sampling_rejects_too_many_without_replacement(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """A request larger than the legal pool must fail without replacement."""
    sampler, _ = _build_sampler(
        synthetic_ratings,
        synthetic_config,
    )

    with pytest.raises(
        ValueError,
        match="without replacement",
    ):
        sampler.sample(
            user_id=1,
            n=3,
            strategy="uniform",
            replace=False,
            draw_key="too_many",
        )


def test_popularity_counts_come_from_train_only(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Popularity statistics must exactly match training frequencies."""
    sampler, split = _build_sampler(
        synthetic_ratings,
        synthetic_config,
    )

    expected = (
        split.train.groupby("movieId")
        .size()
        .reindex(
            sampler.item_universe,
            fill_value=0,
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    assert np.array_equal(
        sampler.train_item_counts,
        expected,
    )

    assert np.array_equal(
        sampler.popularity_weights,
        expected,
    )


def test_sampler_rejects_train_not_subset_of_all_observed(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Training data must be contained in the full observed matrix."""
    split = split_from_config(
        synthetic_ratings,
        synthetic_config,
    )

    missing_pair = split.train.iloc[0]

    bad_all_observed = (
        synthetic_ratings.loc[
            ~(
                (
                    synthetic_ratings[
                        "userId"
                    ]
                    == missing_pair[
                        "userId"
                    ]
                )
                &
                (
                    synthetic_ratings[
                        "movieId"
                    ]
                    == missing_pair[
                        "movieId"
                    ]
                )
            )
        ]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="not a subset",
    ):
        NegativeSampler(
            train=split.train,
            all_observed=bad_all_observed,
            seed=2026,
        )


def test_sampler_rejects_invalid_strategy(
    synthetic_ratings: pd.DataFrame,
    synthetic_config: dict,
) -> None:
    """Unknown negative-sampling strategies must fail loudly."""
    sampler, _ = _build_sampler(
        synthetic_ratings,
        synthetic_config,
    )

    with pytest.raises(
        ValueError,
        match="strategy",
    ):
        sampler.sample(
            user_id=1,
            n=1,
            strategy="not_a_strategy",  # type: ignore[arg-type]
            draw_key="invalid",
        )