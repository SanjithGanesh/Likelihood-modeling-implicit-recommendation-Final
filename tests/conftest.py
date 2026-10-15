"""Shared pytest fixtures for the Paper 2 test suite.

The synthetic dataset is intentionally tiny but structured to exercise the
same invariants as the real MovieLens pipeline.

Dataset design
--------------
- 4 users.
- 12 items.
- Each user observes exactly 10 items.
- Each user leaves exactly 2 items unobserved.
- Every item is observed by at least 3 users.
- No duplicate user-item interactions.
- 40 total interactions.

With an 80/10/10 per-user split, every user receives:
- 8 training interactions,
- 1 validation interaction,
- 1 test interaction.

The two unobserved items per user also provide legal negatives for sampling
tests.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _build_synthetic_ratings() -> pd.DataFrame:
    """Construct the canonical tiny synthetic ratings dataset."""
    excluded_by_user = {
        1: {11, 12},
        2: {1, 2},
        3: {3, 4},
        4: {5, 6},
    }

    rows: list[dict[str, int | float]] = []

    timestamp = 1_000_000

    for user_id in range(1, 5):
        excluded = excluded_by_user[user_id]

        for movie_id in range(1, 13):
            if movie_id in excluded:
                continue

            rating = float(
                ((user_id + movie_id) % 5) + 1
            )

            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )

            timestamp += 1

    ratings = pd.DataFrame(
        rows,
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
    )

    ratings["userId"] = ratings[
        "userId"
    ].astype("int64")

    ratings["movieId"] = ratings[
        "movieId"
    ].astype("int64")

    ratings["rating"] = ratings[
        "rating"
    ].astype("float64")

    ratings["timestamp"] = ratings[
        "timestamp"
    ].astype("int64")

    return ratings


@pytest.fixture
def synthetic_ratings() -> pd.DataFrame:
    """Return a fresh copy of the tiny synthetic ratings dataset."""
    return _build_synthetic_ratings()


@pytest.fixture
def synthetic_config() -> dict:
    """Return a minimal configuration for synthetic pipeline tests."""
    return {
        "project": {
            "name": "paper2",
            "mode": "test",
        },
        "task": {
            "primary_task": "implicit_ranking",
            "interaction_definition": "all_observed_ratings",
            "positive_rating_threshold": None,
        },
        "preprocessing": {
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "reindex_users": True,
            "reindex_items": True,
            "enforce_train_item_coverage": True,
        },
        "split": {
            "strategy": "per_user_random",
            "train_fraction": 0.80,
            "validation_fraction": 0.10,
            "test_fraction": 0.10,
            "seed": 2026,
            "save_manifest": True,
        },
        "evaluation": {
            "ranking_protocol": "full_catalog",
            "exclude_training_items": True,
            "exclude_validation_items_from_test_candidates": True,
            "k_values": [
                1,
                2,
            ],
            "primary_selection_metric": "ndcg_at_1",
            "test_access_during_tuning": False,
            "deterministic_tie_breaking": True,
        },
        "reproducibility": {
            "global_seed": 2026,
            "deterministic": True,
        },
    }