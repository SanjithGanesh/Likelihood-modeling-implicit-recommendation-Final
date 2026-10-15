"""Negative-sampling utilities for the Paper 2 pipeline.

Primary rule
------------
A sampled training negative for user u must not correspond to any observed
user-item interaction for that user anywhere in the benchmark matrix.

Therefore:
- forbidden user-item pairs come from ALL observed interactions,
  including train, validation, and test;
- popularity weights, when used, are computed from TRAINING data only;
- the candidate item universe is the training-item catalog;
- random-number generation uses deterministic independent streams.

Negative sampling is intended for model training where required. The primary
publication ranking evaluation uses full-catalog evaluation and does not rely
on sampled evaluation negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal

import numpy as np
import pandas as pd

from src.reproducibility import make_numpy_rng


SamplingStrategy = Literal[
    "uniform",
    "popularity",
]


@dataclass(frozen=True)
class NegativeSample:
    """One deterministic negative-sampling result."""

    user_id: int
    items: np.ndarray
    strategy: SamplingStrategy
    replace: bool
    draw_key: str


class NegativeSampler:
    """Leakage-safe deterministic negative sampler."""

    def __init__(
        self,
        train: pd.DataFrame,
        all_observed: pd.DataFrame,
        seed: int,
        namespace: str = "negative_sampling",
        popularity_alpha: float = 1.0,
    ) -> None:
        """Construct a shared training negative sampler.

        Parameters
        ----------
        train:
            Training interactions only.

        all_observed:
            All observed benchmark interactions before splitting.
            This is used only to determine which user-item pairs are
            forbidden as negatives.

        seed:
            Base experiment seed.

        namespace:
            RNG namespace used to derive independent deterministic streams.

        popularity_alpha:
            Exponent applied to TRAINING item frequencies for popularity
            sampling. 1.0 uses raw frequency-proportional sampling.
        """
        self._validate_frame(
            train,
            name="train",
        )

        self._validate_frame(
            all_observed,
            name="all_observed",
        )

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                "seed must be an integer"
            )

        if not isinstance(namespace, str):
            raise TypeError(
                "namespace must be a string"
            )

        if not namespace.strip():
            raise ValueError(
                "namespace must not be empty"
            )

        if not isinstance(
            popularity_alpha,
            (int, float),
        ):
            raise TypeError(
                "popularity_alpha must be numeric"
            )

        if popularity_alpha < 0:
            raise ValueError(
                "popularity_alpha must be non-negative"
            )

        self.seed = seed
        self.namespace = namespace
        self.popularity_alpha = float(
            popularity_alpha
        )

        self.item_universe = np.sort(
            train["movieId"]
            .astype("int64")
            .unique()
        )

        if len(self.item_universe) == 0:
            raise ValueError(
                "Training item universe is empty"
            )

        self._item_to_position = {
            int(item_id): position
            for position, item_id in enumerate(
                self.item_universe
            )
        }

        self._observed_by_user = self._build_observed_sets(
            all_observed
        )

        self._validate_train_subset(
            train=train,
            all_observed=all_observed,
        )

        self._train_item_counts = (
            train.groupby("movieId")
            .size()
            .reindex(
                self.item_universe,
                fill_value=0,
            )
            .to_numpy(dtype=np.float64)
        )

        if np.any(self._train_item_counts <= 0):
            raise RuntimeError(
                "Training item universe contains an item "
                "with zero training frequency"
            )

        self._popularity_weights = (
            self._train_item_counts
            ** self.popularity_alpha
        )

    @staticmethod
    def _validate_frame(
        frame: pd.DataFrame,
        name: str,
    ) -> None:
        """Validate the minimum schema required for sampling."""
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

    @staticmethod
    def _build_observed_sets(
        all_observed: pd.DataFrame,
    ) -> dict[int, frozenset[int]]:
        """Build immutable observed-item sets for each user."""
        observed: dict[int, frozenset[int]] = {}

        for user_id, group in all_observed.groupby(
            "userId",
            sort=True,
        ):
            observed[int(user_id)] = frozenset(
                group["movieId"]
                .astype(int)
                .tolist()
            )

        return observed

    @staticmethod
    def _validate_train_subset(
        train: pd.DataFrame,
        all_observed: pd.DataFrame,
    ) -> None:
        """Require every training interaction to exist in all_observed."""
        train_pairs = set(
            zip(
                train["userId"].astype(int),
                train["movieId"].astype(int),
            )
        )

        observed_pairs = set(
            zip(
                all_observed["userId"].astype(int),
                all_observed["movieId"].astype(int),
            )
        )

        invalid_pairs = (
            train_pairs - observed_pairs
        )

        if invalid_pairs:
            examples = list(
                sorted(invalid_pairs)
            )[:5]

            raise ValueError(
                "Training interactions are not a subset "
                "of all_observed. Examples: "
                f"{examples}"
            )

    @property
    def train_item_counts(self) -> np.ndarray:
        """Return a defensive copy of training item frequencies."""
        return self._train_item_counts.copy()

    @property
    def popularity_weights(self) -> np.ndarray:
        """Return a defensive copy of training-derived popularity weights."""
        return self._popularity_weights.copy()

    def observed_items(
        self,
        user_id: int,
    ) -> frozenset[int]:
        """Return all benchmark-observed items for one user."""
        return self._observed_by_user.get(
            int(user_id),
            frozenset(),
        )

    def available_items(
        self,
        user_id: int,
    ) -> np.ndarray:
        """Return legal training-negative items for one user."""
        forbidden = self.observed_items(
            user_id
        )

        if not forbidden:
            return self.item_universe.copy()

        mask = np.fromiter(
            (
                int(item_id) not in forbidden
                for item_id in self.item_universe
            ),
            dtype=bool,
            count=len(self.item_universe),
        )

        return self.item_universe[mask]

    def _available_positions(
        self,
        available_items: np.ndarray,
    ) -> np.ndarray:
        """Map available item IDs to training-universe positions."""
        return np.fromiter(
            (
                self._item_to_position[
                    int(item_id)
                ]
                for item_id in available_items
            ),
            dtype=np.int64,
            count=len(available_items),
        )

    def sample(
        self,
        user_id: int,
        n: int,
        strategy: SamplingStrategy = "uniform",
        replace: bool = True,
        draw_key: Hashable = 0,
    ) -> NegativeSample:
        """Sample deterministic leakage-safe negatives for one user.

        ``draw_key`` identifies this particular sampling event. For example,
        a training loop may use a positive-row index, batch identifier, or
        ``(epoch, row_index)`` tuple.

        Reusing the same user, seed, namespace, strategy, and draw_key
        reproduces the exact same negative sample.
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(
                "n must be an integer"
            )

        if n <= 0:
            raise ValueError(
                "n must be greater than zero"
            )

        if strategy not in (
            "uniform",
            "popularity",
        ):
            raise ValueError(
                "strategy must be 'uniform' or "
                "'popularity'"
            )

        user_id = int(user_id)

        available = self.available_items(
            user_id
        )

        if len(available) == 0:
            raise ValueError(
                f"User {user_id} has no legal "
                "negative items"
            )

        if (
            not replace
            and n > len(available)
        ):
            raise ValueError(
                f"Requested {n} negatives without "
                f"replacement, but user {user_id} "
                f"has only {len(available)} legal items"
            )

        rng = make_numpy_rng(
            self.seed,
            (
                f"{self.namespace}:"
                f"user:{user_id}:"
                f"strategy:{strategy}:"
                f"draw:{repr(draw_key)}"
            ),
        )

        probabilities = None

        if strategy == "popularity":
            positions = self._available_positions(
                available
            )

            weights = self._popularity_weights[
                positions
            ].astype(
                np.float64,
                copy=True,
            )

            weight_sum = float(
                weights.sum()
            )

            if (
                not np.isfinite(weight_sum)
                or weight_sum <= 0.0
            ):
                raise RuntimeError(
                    "Popularity weights are invalid "
                    "after user-specific masking"
                )

            probabilities = (
                weights / weight_sum
            )

        sampled = rng.choice(
            available,
            size=n,
            replace=replace,
            p=probabilities,
        ).astype(
            np.int64,
            copy=False,
        )

        forbidden = self.observed_items(
            user_id
        )

        leaked = [
            int(item_id)
            for item_id in sampled
            if int(item_id) in forbidden
        ]

        if leaked:
            raise RuntimeError(
                "Negative-sampling leakage detected "
                f"for user {user_id}: {leaked[:5]}"
            )

        return NegativeSample(
            user_id=user_id,
            items=sampled,
            strategy=strategy,
            replace=replace,
            draw_key=str(draw_key),
        )