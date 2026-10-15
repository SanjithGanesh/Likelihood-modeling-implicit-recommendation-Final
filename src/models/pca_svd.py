"""Mean-centered truncated-SVD baseline for Paper 2.

The model follows the frozen PCA/SVD specification:

1. Compute each user's mean rating using canonical training ratings only.
2. Construct a sparse user-item matrix where observed entries are

       rating - user_mean

   and missing entries are zero.

3. Fit a rank-K truncated singular-value decomposition.
4. Reconstruct candidate scores as

       user_mean + latent_user @ item_components

The implementation is deterministic for a fixed configuration and exposes
the common Paper 2 model contract:

    fit(train_data)
    score(user_id, candidate_items)
    save(path)
    load(path)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


REQUIRED_TRAIN_COLUMNS = {
    "userId",
    "movieId",
    "rating",
}


def _validate_train_frame(
    train_data: pd.DataFrame,
) -> None:
    """Validate training data required by PCA/SVD."""
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

    required = [
        "userId",
        "movieId",
        "rating",
    ]

    if train_data[
        required
    ].isna().any().any():
        raise ValueError(
            "train_data contains missing required values"
        )

    if train_data.duplicated(
        subset=[
            "userId",
            "movieId",
        ]
    ).any():
        raise ValueError(
            "train_data contains duplicate user-item pairs"
        )

    ratings = train_data[
        "rating"
    ].to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(
        ratings
    ).all():
        raise ValueError(
            "train_data contains non-finite ratings"
        )


def _validate_candidate_items(
    candidate_items: np.ndarray,
) -> np.ndarray:
    """Validate and return a one-dimensional int64 candidate array."""
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


class PCASVDBaseline:
    """Mean-centered truncated-SVD recommendation baseline."""

    model_name = "PCA/SVD"

    def __init__(
        self,
        n_components: int = 32,
        seed: int = 2026,
        n_iter: int = 7,
    ) -> None:
        if (
            not isinstance(
                n_components,
                int,
            )
            or isinstance(
                n_components,
                bool,
            )
        ):
            raise TypeError(
                "n_components must be an integer"
            )

        if n_components <= 0:
            raise ValueError(
                "n_components must be positive"
            )

        if (
            not isinstance(
                seed,
                int,
            )
            or isinstance(
                seed,
                bool,
            )
        ):
            raise TypeError(
                "seed must be an integer"
            )

        if seed < 0:
            raise ValueError(
                "seed must be non-negative"
            )

        if (
            not isinstance(
                n_iter,
                int,
            )
            or isinstance(
                n_iter,
                bool,
            )
        ):
            raise TypeError(
                "n_iter must be an integer"
            )

        if n_iter <= 0:
            raise ValueError(
                "n_iter must be positive"
            )

        self.n_components = n_components
        self.seed = seed
        self.n_iter = n_iter

        self._user_ids: np.ndarray | None = None
        self._item_ids: np.ndarray | None = None

        self._user_to_index: dict[int, int] | None = None
        self._item_to_index: dict[int, int] | None = None

        self._user_means: np.ndarray | None = None
        self._user_latent: np.ndarray | None = None
        self._components: np.ndarray | None = None

        self._explained_variance_ratio: np.ndarray | None = None

        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether fit() completed successfully."""
        return self._is_fitted

    @property
    def user_ids(self) -> np.ndarray:
        """Return fitted user IDs."""
        self._require_fitted()

        assert self._user_ids is not None

        return self._user_ids.copy()

    @property
    def item_catalog(self) -> np.ndarray:
        """Return fitted training item catalog."""
        self._require_fitted()

        assert self._item_ids is not None

        return self._item_ids.copy()

    @property
    def user_means(self) -> np.ndarray:
        """Return training-only user means."""
        self._require_fitted()

        assert self._user_means is not None

        return self._user_means.copy()

    @property
    def explained_variance_ratio(
        self,
    ) -> np.ndarray:
        """Return explained-variance ratios from TruncatedSVD."""
        self._require_fitted()

        assert (
            self._explained_variance_ratio
            is not None
        )

        return (
            self._explained_variance_ratio
            .copy()
        )

    def _require_fitted(self) -> None:
        """Fail if fit() has not completed."""
        if not self._is_fitted:
            raise RuntimeError(
                "PCASVDBaseline must be fitted "
                "before scoring"
            )

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> PCASVDBaseline:
        """Fit mean-centered truncated SVD to training ratings only."""
        _validate_train_frame(
            train_data
        )

        frame = train_data[
            [
                "userId",
                "movieId",
                "rating",
            ]
        ].copy()

        frame["userId"] = frame[
            "userId"
        ].astype("int64")

        frame["movieId"] = frame[
            "movieId"
        ].astype("int64")

        frame["rating"] = frame[
            "rating"
        ].astype("float64")

        user_ids = np.sort(
            frame["userId"].unique()
        ).astype(
            np.int64,
            copy=False,
        )

        item_ids = np.sort(
            frame["movieId"].unique()
        ).astype(
            np.int64,
            copy=False,
        )

        n_users = len(
            user_ids
        )

        n_items = len(
            item_ids
        )

        max_rank = min(
            n_users,
            n_items,
        )

        if (
            self.n_components
            >= max_rank
        ):
            raise ValueError(
                "n_components must be smaller than "
                f"min(n_users, n_items)={max_rank}; "
                f"received {self.n_components}"
            )

        user_to_index = {
            int(user_id): index
            for index, user_id
            in enumerate(user_ids)
        }

        item_to_index = {
            int(item_id): index
            for index, item_id
            in enumerate(item_ids)
        }

        user_means_series = (
            frame
            .groupby(
                "userId",
                sort=True,
            )["rating"]
            .mean()
        )

        user_means = np.asarray(
            [
                float(
                    user_means_series.loc[
                        int(user_id)
                    ]
                )
                for user_id in user_ids
            ],
            dtype=np.float64,
        )

        row_indices = np.fromiter(
            (
                user_to_index[
                    int(user_id)
                ]
                for user_id
                in frame["userId"].to_numpy()
            ),
            dtype=np.int64,
            count=len(frame),
        )

        col_indices = np.fromiter(
            (
                item_to_index[
                    int(item_id)
                ]
                for item_id
                in frame["movieId"].to_numpy()
            ),
            dtype=np.int64,
            count=len(frame),
        )

        ratings = frame[
            "rating"
        ].to_numpy(
            dtype=np.float64,
        )

        centered_values = (
            ratings
            -
            user_means[
                row_indices
            ]
        )

        if not np.isfinite(
            centered_values
        ).all():
            raise RuntimeError(
                "Mean-centered training matrix "
                "contains non-finite values"
            )

        matrix = sparse.csr_matrix(
            (
                centered_values,
                (
                    row_indices,
                    col_indices,
                ),
            ),
            shape=(
                n_users,
                n_items,
            ),
            dtype=np.float64,
        )

        model = TruncatedSVD(
            n_components=self.n_components,
            algorithm="randomized",
            n_iter=self.n_iter,
            random_state=self.seed,
        )

        user_latent = (
            model.fit_transform(
                matrix
            )
        )

        components = (
            model.components_
        )

        explained = (
            model.explained_variance_ratio_
        )

        if user_latent.shape != (
            n_users,
            self.n_components,
        ):
            raise RuntimeError(
                "Unexpected user latent matrix shape"
            )

        if components.shape != (
            self.n_components,
            n_items,
        ):
            raise RuntimeError(
                "Unexpected SVD component matrix shape"
            )

        if not np.isfinite(
            user_latent
        ).all():
            raise RuntimeError(
                "SVD produced non-finite user factors"
            )

        if not np.isfinite(
            components
        ).all():
            raise RuntimeError(
                "SVD produced non-finite components"
            )

        if not np.isfinite(
            explained
        ).all():
            raise RuntimeError(
                "SVD produced non-finite "
                "explained-variance ratios"
            )

        self._user_ids = user_ids
        self._item_ids = item_ids

        self._user_to_index = (
            user_to_index
        )

        self._item_to_index = (
            item_to_index
        )

        self._user_means = (
            user_means
        )

        self._user_latent = np.asarray(
            user_latent,
            dtype=np.float64,
        )

        self._components = np.asarray(
            components,
            dtype=np.float64,
        )

        self._explained_variance_ratio = np.asarray(
            explained,
            dtype=np.float64,
        )

        self._is_fitted = True

        return self

    def score(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Predict reconstructed rating scores for candidate items."""
        self._require_fitted()

        items = _validate_candidate_items(
            candidate_items
        )

        assert self._user_to_index is not None
        assert self._item_to_index is not None
        assert self._user_means is not None
        assert self._user_latent is not None
        assert self._components is not None

        user_id = int(
            user_id
        )

        if (
            user_id
            not in self._user_to_index
        ):
            raise ValueError(
                f"Unknown user ID: {user_id}"
            )

        unknown_items = [
            int(item_id)
            for item_id in items
            if int(item_id)
            not in self._item_to_index
        ]

        if unknown_items:
            raise ValueError(
                "candidate_items contains items outside "
                "the fitted training catalog: "
                f"{unknown_items[:10]}"
            )

        user_index = (
            self._user_to_index[
                user_id
            ]
        )

        item_indices = np.fromiter(
            (
                self._item_to_index[
                    int(item_id)
                ]
                for item_id in items
            ),
            dtype=np.int64,
            count=len(items),
        )

        latent_scores = (
            self._user_latent[
                user_index
            ]
            @
            self._components[
                :,
                item_indices,
            ]
        )

        scores = (
            self._user_means[
                user_index
            ]
            +
            latent_scores
        )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        )

        if scores.shape != (
            len(items),
        ):
            raise RuntimeError(
                "PCA/SVD returned an invalid "
                "score shape"
            )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                "PCA/SVD produced non-finite scores"
            )

        return scores

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted PCA/SVD state.

        The supplied path should end in .npz. A small JSON metadata file is
        written beside it using the same stem and a .json extension.
        """
        self._require_fitted()

        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "PCA/SVD save path must end in .npz"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        assert self._user_ids is not None
        assert self._item_ids is not None
        assert self._user_means is not None
        assert self._user_latent is not None
        assert self._components is not None
        assert (
            self._explained_variance_ratio
            is not None
        )

        np.savez_compressed(
            path,
            user_ids=self._user_ids,
            item_ids=self._item_ids,
            user_means=self._user_means,
            user_latent=self._user_latent,
            components=self._components,
            explained_variance_ratio=(
                self._explained_variance_ratio
            ),
        )

        metadata = {
            "model_name": self.model_name,
            "method": (
                "mean_centered_truncated_svd"
            ),
            "n_components": (
                self.n_components
            ),
            "seed": self.seed,
            "n_iter": self.n_iter,
            "n_users": int(
                len(
                    self._user_ids
                )
            ),
            "n_items": int(
                len(
                    self._item_ids
                )
            ),
        }

        metadata_path = (
            path.with_suffix(
                ".json"
            )
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> PCASVDBaseline:
        """Load fitted PCA/SVD state."""
        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "PCA/SVD load path must end in .npz"
            )

        metadata_path = (
            path.with_suffix(
                ".json"
            )
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if metadata.get(
            "model_name"
        ) != cls.model_name:
            raise ValueError(
                "Saved model is not a PCASVDBaseline"
            )

        model = cls(
            n_components=int(
                metadata[
                    "n_components"
                ]
            ),
            seed=int(
                metadata[
                    "seed"
                ]
            ),
            n_iter=int(
                metadata[
                    "n_iter"
                ]
            ),
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as state:
            user_ids = np.asarray(
                state["user_ids"],
                dtype=np.int64,
            )

            item_ids = np.asarray(
                state["item_ids"],
                dtype=np.int64,
            )

            user_means = np.asarray(
                state["user_means"],
                dtype=np.float64,
            )

            user_latent = np.asarray(
                state["user_latent"],
                dtype=np.float64,
            )

            components = np.asarray(
                state["components"],
                dtype=np.float64,
            )

            explained = np.asarray(
                state[
                    "explained_variance_ratio"
                ],
                dtype=np.float64,
            )

        n_users = len(
            user_ids
        )

        n_items = len(
            item_ids
        )

        if user_means.shape != (
            n_users,
        ):
            raise ValueError(
                "Saved PCA/SVD user means "
                "have invalid shape"
            )

        if user_latent.shape != (
            n_users,
            model.n_components,
        ):
            raise ValueError(
                "Saved PCA/SVD user latent "
                "matrix has invalid shape"
            )

        if components.shape != (
            model.n_components,
            n_items,
        ):
            raise ValueError(
                "Saved PCA/SVD component matrix "
                "has invalid shape"
            )

        if explained.shape != (
            model.n_components,
        ):
            raise ValueError(
                "Saved explained-variance vector "
                "has invalid shape"
            )

        arrays = [
            user_means,
            user_latent,
            components,
            explained,
        ]

        if any(
            not np.isfinite(
                array
            ).all()
            for array in arrays
        ):
            raise ValueError(
                "Saved PCA/SVD state contains "
                "non-finite values"
            )

        if len(
            np.unique(
                user_ids
            )
        ) != n_users:
            raise ValueError(
                "Saved PCA/SVD user IDs "
                "contain duplicates"
            )

        if len(
            np.unique(
                item_ids
            )
        ) != n_items:
            raise ValueError(
                "Saved PCA/SVD item IDs "
                "contain duplicates"
            )

        model._user_ids = (
            user_ids
        )

        model._item_ids = (
            item_ids
        )

        model._user_to_index = {
            int(user_id): index
            for index, user_id
            in enumerate(
                user_ids
            )
        }

        model._item_to_index = {
            int(item_id): index
            for index, item_id
            in enumerate(
                item_ids
            )
        }

        model._user_means = (
            user_means
        )

        model._user_latent = (
            user_latent
        )

        model._components = (
            components
        )

        model._explained_variance_ratio = (
            explained
        )

        model._is_fitted = True

        return model