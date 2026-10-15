"""Simple recommendation baselines for Paper 2.

This module implements the two universal baselines defined in the frozen
model specification:

1. RandomBaseline
2. PopularityBaseline

Both models obey the common scoring contract:

    fit(train_data)
    score(user_id, candidate_items)
    save(path)
    load(path)

The implementations are deterministic and operate only on the canonical
training catalog.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_TRAIN_COLUMNS = {
    "userId",
    "movieId",
}


def _validate_train_frame(
    train_data: pd.DataFrame,
) -> None:
    """Validate the minimum schema required by the baselines."""
    if not isinstance(
        train_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "train_data must be a pandas DataFrame"
        )

    missing = (
        REQUIRED_TRAIN_COLUMNS
        - set(train_data.columns)
    )

    if missing:
        raise ValueError(
            "train_data is missing required columns: "
            f"{sorted(missing)}"
        )

    if len(train_data) == 0:
        raise ValueError(
            "train_data must not be empty"
        )

    if train_data[
        ["userId", "movieId"]
    ].isna().any().any():
        raise ValueError(
            "train_data contains missing userId/movieId values"
        )


def _validate_candidate_items(
    candidate_items: np.ndarray,
) -> np.ndarray:
    """Return candidate items as a validated one-dimensional int64 array."""
    items = np.asarray(
        candidate_items,
        dtype=np.int64,
    )

    if items.ndim != 1:
        raise ValueError(
            "candidate_items must be one-dimensional"
        )

    if len(items) == 0:
        raise ValueError(
            "candidate_items must not be empty"
        )

    if len(
        np.unique(items)
    ) != len(items):
        raise ValueError(
            "candidate_items contains duplicate item IDs"
        )

    return items


def _json_ready(
    value: Any,
) -> Any:
    """Convert NumPy scalar values into ordinary JSON-compatible values."""
    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    return value


class RandomBaseline:
    """Deterministic random-ranking baseline.

    The score for a user-item pair is derived from a cryptographic hash of:

    - model namespace;
    - experiment seed;
    - user ID;
    - item ID.

    Consequently, scoring is independent of candidate ordering and repeated
    calls with the same seed/user/item combination return exactly the same
    value.
    """

    model_name = "Random"

    def __init__(
        self,
        seed: int = 2026,
        namespace: str = "random_baseline",
    ) -> None:
        if not isinstance(
            seed,
            int,
        ) or isinstance(
            seed,
            bool,
        ):
            raise TypeError(
                "seed must be an integer"
            )

        if seed < 0:
            raise ValueError(
                "seed must be non-negative"
            )

        if not isinstance(
            namespace,
            str,
        ) or not namespace.strip():
            raise ValueError(
                "namespace must be a non-empty string"
            )

        self.seed = seed
        self.namespace = namespace

        self._item_catalog: np.ndarray | None = None
        self._item_catalog_set: set[int] | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether fit() has been called successfully."""
        return self._is_fitted

    @property
    def item_catalog(self) -> np.ndarray:
        """Return a defensive copy of the fitted training catalog."""
        self._require_fitted()

        assert self._item_catalog is not None

        return self._item_catalog.copy()

    def _require_fitted(self) -> None:
        """Fail if the model has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "RandomBaseline must be fitted before scoring"
            )

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> RandomBaseline:
        """Register the canonical training catalog.

        RandomBaseline learns no parameters, but fit() records the legal item
        catalog so that downstream scoring cannot silently use unknown items.
        """
        _validate_train_frame(
            train_data
        )

        item_catalog = np.sort(
            train_data["movieId"]
            .astype("int64")
            .unique()
        )

        self._item_catalog = (
            item_catalog.astype(
                np.int64,
                copy=False,
            )
        )

        self._item_catalog_set = set(
            int(item)
            for item in self._item_catalog
        )

        self._is_fitted = True

        return self

    def _pair_score(
        self,
        user_id: int,
        item_id: int,
    ) -> float:
        """Generate one deterministic pseudo-random score in [0, 1)."""
        payload = (
            f"{self.namespace}|"
            f"{self.seed}|"
            f"{int(user_id)}|"
            f"{int(item_id)}"
        ).encode("utf-8")

        digest = hashlib.sha256(
            payload
        ).digest()

        integer_value = int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )

        return (
            integer_value
            / float(2**64)
        )

    def score(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return deterministic random scores for candidate items."""
        self._require_fitted()

        items = _validate_candidate_items(
            candidate_items
        )

        assert self._item_catalog_set is not None

        unknown = [
            int(item)
            for item in items
            if int(item)
            not in self._item_catalog_set
        ]

        if unknown:
            raise ValueError(
                "candidate_items contains items outside "
                "the fitted training catalog: "
                f"{unknown[:10]}"
            )

        scores = np.fromiter(
            (
                self._pair_score(
                    int(user_id),
                    int(item_id),
                )
                for item_id in items
            ),
            dtype=np.float64,
            count=len(items),
        )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                "RandomBaseline produced non-finite scores"
            )

        return scores

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save the fitted baseline as JSON."""
        self._require_fitted()

        assert self._item_catalog is not None

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_name": self.model_name,
            "seed": self.seed,
            "namespace": self.namespace,
            "item_catalog": [
                int(item)
                for item in self._item_catalog
            ],
        }

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> RandomBaseline:
        """Load a previously saved RandomBaseline."""
        path = Path(path)

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if payload.get(
            "model_name"
        ) != cls.model_name:
            raise ValueError(
                "Saved model is not a RandomBaseline"
            )

        model = cls(
            seed=int(
                payload["seed"]
            ),
            namespace=str(
                payload["namespace"]
            ),
        )

        catalog = np.asarray(
            payload["item_catalog"],
            dtype=np.int64,
        )

        if catalog.ndim != 1:
            raise ValueError(
                "Saved item catalog is invalid"
            )

        if len(catalog) == 0:
            raise ValueError(
                "Saved item catalog is empty"
            )

        if len(
            np.unique(catalog)
        ) != len(catalog):
            raise ValueError(
                "Saved item catalog contains duplicates"
            )

        catalog = np.sort(
            catalog
        )

        model._item_catalog = catalog

        model._item_catalog_set = set(
            int(item)
            for item in catalog
        )

        model._is_fitted = True

        return model


class PopularityBaseline:
    """Training-frequency popularity baseline.

    Item scores are equal to the number of canonical training interactions
    involving that item.

    No validation or test interaction contributes to the score.
    """

    model_name = "Popularity"

    def __init__(
        self,
    ) -> None:
        self._item_catalog: np.ndarray | None = None
        self._item_catalog_set: set[int] | None = None
        self._counts: dict[int, int] | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether fit() has been called successfully."""
        return self._is_fitted

    @property
    def item_catalog(self) -> np.ndarray:
        """Return a defensive copy of the fitted training catalog."""
        self._require_fitted()

        assert self._item_catalog is not None

        return self._item_catalog.copy()

    @property
    def item_counts(self) -> dict[int, int]:
        """Return a defensive copy of training-only popularity counts."""
        self._require_fitted()

        assert self._counts is not None

        return dict(
            self._counts
        )

    def _require_fitted(self) -> None:
        """Fail if the model has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "PopularityBaseline must be fitted before scoring"
            )

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> PopularityBaseline:
        """Calculate item interaction counts from training data only."""
        _validate_train_frame(
            train_data
        )

        counts_series = (
            train_data["movieId"]
            .astype("int64")
            .value_counts(
                sort=False
            )
            .sort_index()
        )

        counts = {
            int(item_id): int(count)
            for item_id, count
            in counts_series.items()
        }

        item_catalog = np.asarray(
            sorted(counts),
            dtype=np.int64,
        )

        self._counts = counts
        self._item_catalog = item_catalog

        self._item_catalog_set = set(
            counts
        )

        self._is_fitted = True

        return self

    def score(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return training-frequency scores for candidate items.

        user_id is accepted to satisfy the common model contract, although
        popularity scores are intentionally user-independent.
        """
        self._require_fitted()

        items = _validate_candidate_items(
            candidate_items
        )

        assert self._item_catalog_set is not None
        assert self._counts is not None

        unknown = [
            int(item)
            for item in items
            if int(item)
            not in self._item_catalog_set
        ]

        if unknown:
            raise ValueError(
                "candidate_items contains items outside "
                "the fitted training catalog: "
                f"{unknown[:10]}"
            )

        scores = np.fromiter(
            (
                float(
                    self._counts[
                        int(item_id)
                    ]
                )
                for item_id in items
            ),
            dtype=np.float64,
            count=len(items),
        )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                "PopularityBaseline produced non-finite scores"
            )

        return scores

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save the fitted popularity baseline as JSON."""
        self._require_fitted()

        assert self._counts is not None

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_name": self.model_name,
            "item_counts": {
                str(item_id): int(count)
                for item_id, count
                in sorted(
                    self._counts.items()
                )
            },
        }

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> PopularityBaseline:
        """Load a previously saved PopularityBaseline."""
        path = Path(path)

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if payload.get(
            "model_name"
        ) != cls.model_name:
            raise ValueError(
                "Saved model is not a PopularityBaseline"
            )

        raw_counts = payload.get(
            "item_counts"
        )

        if not isinstance(
            raw_counts,
            dict,
        ) or len(raw_counts) == 0:
            raise ValueError(
                "Saved popularity counts are invalid"
            )

        counts = {
            int(item_id): int(
                _json_ready(
                    count
                )
            )
            for item_id, count
            in raw_counts.items()
        }

        if any(
            count <= 0
            for count in counts.values()
        ):
            raise ValueError(
                "Saved popularity counts must be positive"
            )

        model = cls()

        model._counts = dict(
            sorted(
                counts.items()
            )
        )

        model._item_catalog = np.asarray(
            sorted(counts),
            dtype=np.int64,
        )

        model._item_catalog_set = set(
            counts
        )

        model._is_fitted = True

        return model