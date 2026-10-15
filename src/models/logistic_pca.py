"""Canonical Logistic PCA baseline for Paper 2.

This module implements the LogisticPCA definition frozen in
docs/model_spec.md revision v1.2.

Model
-----

For user u and item i,

    theta_ui = mu_i + a_u^T b_i

and

    X_ui ~ Bernoulli(sigmoid(theta_ui))

where:

- A contains user / observation coefficients;
- B contains item / dimension basis vectors;
- mu contains item intercepts.

The primary Paper 2 LogisticPCA model:

- uses the binary interaction matrix;
- treats canonical training interactions as X_ui = 1;
- treats genuinely unobserved interactions as X_ui = 0;
- masks validation/test positives rather than treating them as zeros;
- optimizes the complete masked Bernoulli likelihood;
- does NOT use sampled negatives;
- does NOT impose orthogonality or centering constraints on A;
- does NOT apply L2 regularization to A, B, or mu;
- uses theta_ui itself as the ranking score.

The masked complete-matrix likelihood is evaluated efficiently in user
blocks rather than materializing one giant training-example table.

Common Paper 2 contract:

    fit(train_data)
    score(user_id, candidate_items)
    save(path)
    load(path)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.reproducibility import (
    make_numpy_rng,
    make_torch_generator,
    set_seed,
)


REQUIRED_INTERACTION_COLUMNS = {
    "userId",
    "movieId",
}


def _validate_interaction_frame(
    data: pd.DataFrame,
    *,
    name: str,
    allow_empty: bool,
) -> None:
    """Validate a user-item interaction frame."""
    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{name} must be a pandas DataFrame"
        )

    missing = (
        REQUIRED_INTERACTION_COLUMNS
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if (
        not allow_empty
        and len(data) == 0
    ):
        raise ValueError(
            f"{name} must not be empty"
        )

    if data[
        [
            "userId",
            "movieId",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{name} contains missing "
            "userId/movieId values"
        )

    if data.duplicated(
        subset=[
            "userId",
            "movieId",
        ]
    ).any():
        raise ValueError(
            f"{name} contains duplicate user-item pairs"
        )


def _validate_candidate_items(
    candidate_items: np.ndarray,
) -> np.ndarray:
    """Validate a candidate-item array."""
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


class _LogisticPCAModule(
    nn.Module,
):
    """PyTorch representation of theta = 1 mu^T + A B^T."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        initial_item_intercepts: np.ndarray,
        init_generator: torch.Generator,
    ) -> None:
        super().__init__()

        intercepts = np.asarray(
            initial_item_intercepts,
            dtype=np.float32,
        )

        if intercepts.shape != (
            n_items,
        ):
            raise ValueError(
                "initial_item_intercepts has invalid shape"
            )

        if not np.isfinite(
            intercepts
        ).all():
            raise ValueError(
                "initial_item_intercepts contains "
                "non-finite values"
            )

        self.user_scores = nn.Embedding(
            n_users,
            n_factors,
        )

        self.item_basis = nn.Embedding(
            n_items,
            n_factors,
        )

        self.item_intercept = nn.Parameter(
            torch.from_numpy(
                intercepts.copy()
            )
        )

        nn.init.normal_(
            self.user_scores.weight,
            mean=0.0,
            std=0.01,
            generator=init_generator,
        )

        nn.init.normal_(
            self.item_basis.weight,
            mean=0.0,
            std=0.01,
            generator=init_generator,
        )

    def user_block_logits(
        self,
        user_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for a block of users against all items."""
        a = self.user_scores(
            user_indices
        )

        b = (
            self.item_basis
            .weight
        )

        logits = (
            a
            @
            b.transpose(
                0,
                1,
            )
        )

        logits = (
            logits
            +
            self.item_intercept.unsqueeze(
                0
            )
        )

        return logits

    def pair_logits(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return aligned user-item natural parameters."""
        a = self.user_scores(
            user_indices
        )

        b = self.item_basis(
            item_indices
        )

        interaction = (
            a
            *
            b
        ).sum(
            dim=1
        )

        intercept = (
            self.item_intercept[
                item_indices
            ]
        )

        return (
            intercept
            +
            interaction
        )


class LogisticPCA:
    """Low-rank Bernoulli natural-parameter model."""

    model_name = "LogisticPCA"

    def __init__(
        self,
        heldout_interactions: pd.DataFrame | None = None,
        n_factors: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 5,
        user_batch_size: int = 32,
        max_grad_norm: float | None = 5.0,
        intercept_epsilon: float = 1e-6,
        seed: int = 2026,
        device: str = "cpu",
    ) -> None:
        if (
            not isinstance(
                n_factors,
                int,
            )
            or isinstance(
                n_factors,
                bool,
            )
            or n_factors <= 0
        ):
            raise ValueError(
                "n_factors must be a positive integer"
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if (
            not isinstance(
                epochs,
                int,
            )
            or isinstance(
                epochs,
                bool,
            )
            or epochs <= 0
        ):
            raise ValueError(
                "epochs must be a positive integer"
            )

        if (
            not isinstance(
                user_batch_size,
                int,
            )
            or isinstance(
                user_batch_size,
                bool,
            )
            or user_batch_size <= 0
        ):
            raise ValueError(
                "user_batch_size must be "
                "a positive integer"
            )

        if (
            max_grad_norm is not None
            and max_grad_norm <= 0
        ):
            raise ValueError(
                "max_grad_norm must be positive or None"
            )

        if not (
            0.0
            <
            intercept_epsilon
            <
            0.5
        ):
            raise ValueError(
                "intercept_epsilon must lie in (0, 0.5)"
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
            or seed < 0
        ):
            raise ValueError(
                "seed must be a non-negative integer"
            )

        torch_device = torch.device(
            device
        )

        if (
            torch_device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise ValueError(
                "CUDA device requested but CUDA is unavailable"
            )

        if heldout_interactions is not None:
            _validate_interaction_frame(
                heldout_interactions,
                name="heldout_interactions",
                allow_empty=True,
            )

            self.heldout_interactions = (
                heldout_interactions[
                    [
                        "userId",
                        "movieId",
                    ]
                ]
                .copy()
            )
        else:
            self.heldout_interactions = None

        self.n_factors = n_factors

        self.learning_rate = float(
            learning_rate
        )

        self.epochs = epochs

        self.user_batch_size = (
            user_batch_size
        )

        self.max_grad_norm = (
            None
            if max_grad_norm is None
            else float(
                max_grad_norm
            )
        )

        self.intercept_epsilon = float(
            intercept_epsilon
        )

        self.seed = seed

        self.device = (
            torch_device
        )

        self._user_ids: (
            np.ndarray | None
        ) = None

        self._item_ids: (
            np.ndarray | None
        ) = None

        self._user_to_index: (
            dict[int, int] | None
        ) = None

        self._item_to_index: (
            dict[int, int] | None
        ) = None

        self._train_items_by_user: (
            list[np.ndarray] | None
        ) = None

        self._heldout_items_by_user: (
            list[np.ndarray] | None
        ) = None

        self._module: (
            _LogisticPCAModule | None
        ) = None

        self._legal_cell_count: (
            int | None
        ) = None

        self._positive_cell_count: (
            int | None
        ) = None

        self._heldout_cell_count: (
            int | None
        ) = None

        self._history: list[
            dict[str, float | int]
        ] = []

        self._is_fitted = False

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Whether fitting completed."""
        return self._is_fitted

    @property
    def user_ids(
        self,
    ) -> np.ndarray:
        """Return fitted user IDs."""
        self._require_fitted()

        assert self._user_ids is not None

        return self._user_ids.copy()

    @property
    def item_catalog(
        self,
    ) -> np.ndarray:
        """Return fitted training item catalog."""
        self._require_fitted()

        assert self._item_ids is not None

        return self._item_ids.copy()

    @property
    def legal_cell_count(
        self,
    ) -> int:
        """Number of cells included in the masked likelihood."""
        self._require_fitted()

        assert self._legal_cell_count is not None

        return int(
            self._legal_cell_count
        )

    @property
    def positive_cell_count(
        self,
    ) -> int:
        """Number of canonical training positives."""
        self._require_fitted()

        assert self._positive_cell_count is not None

        return int(
            self._positive_cell_count
        )

    @property
    def zero_cell_count(
        self,
    ) -> int:
        """Number of genuine binary-zero cells."""
        self._require_fitted()

        assert self._legal_cell_count is not None
        assert self._positive_cell_count is not None

        return int(
            self._legal_cell_count
            -
            self._positive_cell_count
        )

    @property
    def heldout_cell_count(
        self,
    ) -> int:
        """Number of validation/test interactions masked from fitting."""
        self._require_fitted()

        assert self._heldout_cell_count is not None

        return int(
            self._heldout_cell_count
        )

    @property
    def training_history(
        self,
    ) -> list[
        dict[str, float | int]
    ]:
        """Return epoch-level diagnostics."""
        return [
            dict(row)
            for row in self._history
        ]

    def _require_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "LogisticPCA must be fitted before scoring"
            )

    def _require_heldout_mask(
        self,
    ) -> pd.DataFrame:
        if self.heldout_interactions is None:
            raise RuntimeError(
                "LogisticPCA requires heldout_interactions "
                "during fit() so validation/test positives "
                "can be masked rather than treated as zeros"
            )

        return self.heldout_interactions

    def _prepare_histories(
        self,
        train: pd.DataFrame,
        heldout: pd.DataFrame,
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
    ]:
        """Map train and held-out interactions to canonical item indices."""
        assert self._user_ids is not None
        assert self._item_ids is not None
        assert self._user_to_index is not None
        assert self._item_to_index is not None

        train_pairs = set(
            zip(
                train[
                    "userId"
                ].astype(
                    "int64"
                ),
                train[
                    "movieId"
                ].astype(
                    "int64"
                ),
            )
        )

        heldout_pairs = set(
            zip(
                heldout[
                    "userId"
                ].astype(
                    "int64"
                ),
                heldout[
                    "movieId"
                ].astype(
                    "int64"
                ),
            )
        )

        overlap = (
            train_pairs
            &
            heldout_pairs
        )

        if overlap:
            raise ValueError(
                "heldout_interactions overlaps "
                "canonical training positives"
            )

        unknown_users = sorted(
            {
                int(user_id)
                for user_id
                in heldout[
                    "userId"
                ].unique()
                if int(user_id)
                not in self._user_to_index
            }
        )

        if unknown_users:
            raise ValueError(
                "heldout_interactions contains users "
                "outside the training user set: "
                f"{unknown_users[:10]}"
            )

        unknown_items = sorted(
            {
                int(item_id)
                for item_id
                in heldout[
                    "movieId"
                ].unique()
                if int(item_id)
                not in self._item_to_index
            }
        )

        if unknown_items:
            raise ValueError(
                "heldout_interactions contains items "
                "outside the canonical training catalog: "
                f"{unknown_items[:10]}"
            )

        train_items_by_user: list[
            list[int]
        ] = [
            []
            for _ in range(
                len(
                    self._user_ids
                )
            )
        ]

        heldout_items_by_user: list[
            list[int]
        ] = [
            []
            for _ in range(
                len(
                    self._user_ids
                )
            )
        ]

        for row in train[
            [
                "userId",
                "movieId",
            ]
        ].itertuples(
            index=False
        ):
            user_index = (
                self._user_to_index[
                    int(
                        row.userId
                    )
                ]
            )

            item_index = (
                self._item_to_index[
                    int(
                        row.movieId
                    )
                ]
            )

            train_items_by_user[
                user_index
            ].append(
                item_index
            )

        for row in heldout[
            [
                "userId",
                "movieId",
            ]
        ].itertuples(
            index=False
        ):
            user_index = (
                self._user_to_index[
                    int(
                        row.userId
                    )
                ]
            )

            item_index = (
                self._item_to_index[
                    int(
                        row.movieId
                    )
                ]
            )

            heldout_items_by_user[
                user_index
            ].append(
                item_index
            )

        train_arrays: list[
            np.ndarray
        ] = []

        heldout_arrays: list[
            np.ndarray
        ] = []

        for user_index in range(
            len(
                self._user_ids
            )
        ):
            train_array = np.asarray(
                sorted(
                    train_items_by_user[
                        user_index
                    ]
                ),
                dtype=np.int64,
            )

            heldout_array = np.asarray(
                sorted(
                    heldout_items_by_user[
                        user_index
                    ]
                ),
                dtype=np.int64,
            )

            if len(
                np.unique(
                    train_array
                )
            ) != len(
                train_array
            ):
                raise RuntimeError(
                    "Duplicate training items found "
                    "after canonicalization"
                )

            if len(
                np.unique(
                    heldout_array
                )
            ) != len(
                heldout_array
            ):
                raise RuntimeError(
                    "Duplicate held-out items found "
                    "after canonicalization"
                )

            if np.intersect1d(
                train_array,
                heldout_array,
            ).size != 0:
                raise RuntimeError(
                    "Training and held-out item sets "
                    "overlap for a user"
                )

            train_arrays.append(
                train_array
            )

            heldout_arrays.append(
                heldout_array
            )

        return (
            train_arrays,
            heldout_arrays,
        )

    def _initial_item_intercepts(
        self,
    ) -> np.ndarray:
        """Initialize item intercepts from training-only prevalence.

        For item i,

            p_i =
                train_positive_count_i
                /
                n_training_users

        and

            mu_i = logit(p_i).

        Validation/test interaction identities are NOT used to initialize
        item intercepts. They are used only later in the likelihood mask so
        held-out positives are not incorrectly treated as binary zeros.
        """
        assert self._item_ids is not None
        assert self._user_ids is not None
        assert self._train_items_by_user is not None

        n_users = len(
            self._user_ids
        )

        n_items = len(
            self._item_ids
        )

        train_counts = np.zeros(
            n_items,
            dtype=np.int64,
        )

        for item_indices in (
            self._train_items_by_user
        ):
            np.add.at(
                train_counts,
                item_indices,
                1,
            )

        if np.any(
            train_counts <= 0
        ):
            raise RuntimeError(
                "At least one canonical training item "
                "has zero training positives"
            )

        if np.any(
            train_counts > n_users
        ):
            raise RuntimeError(
                "Invalid LogisticPCA training prevalence counts"
            )

        prevalence = (
            train_counts.astype(
                np.float64
            )
            /
            float(
                n_users
            )
        )

        prevalence = np.clip(
            prevalence,
            self.intercept_epsilon,
            1.0
            -
            self.intercept_epsilon,
        )

        intercepts = np.log(
            prevalence
            /
            (
                1.0
                -
                prevalence
            )
        )

        if not np.isfinite(
            intercepts
        ).all():
            raise RuntimeError(
                "LogisticPCA item-intercept "
                "initialization is non-finite"
            )

        return intercepts.astype(
            np.float32,
            copy=False,
        )   

    def _build_user_batch_targets(
        self,
        user_indices: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        int,
    ]:
        """Build binary targets and legal-cell mask for a user block."""
        assert self._item_ids is not None
        assert self._train_items_by_user is not None
        assert self._heldout_items_by_user is not None

        batch_users = len(
            user_indices
        )

        n_items = len(
            self._item_ids
        )

        targets = torch.zeros(
            (
                batch_users,
                n_items,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        legal_mask = torch.ones(
            (
                batch_users,
                n_items,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        positive_count = 0
        heldout_count = 0

        for local_index, raw_user_index in enumerate(
            user_indices
        ):
            user_index = int(
                raw_user_index
            )

            positive_items = (
                self._train_items_by_user[
                    user_index
                ]
            )

            heldout_items = (
                self._heldout_items_by_user[
                    user_index
                ]
            )

            if len(
                positive_items
            ) > 0:
                positive_tensor = (
                    torch.as_tensor(
                        positive_items,
                        dtype=torch.long,
                        device=self.device,
                    )
                )

                targets[
                    local_index,
                    positive_tensor,
                ] = 1.0

                positive_count += len(
                    positive_items
                )

            if len(
                heldout_items
            ) > 0:
                heldout_tensor = (
                    torch.as_tensor(
                        heldout_items,
                        dtype=torch.long,
                        device=self.device,
                    )
                )

                legal_mask[
                    local_index,
                    heldout_tensor,
                ] = False

                heldout_count += len(
                    heldout_items
                )

        return (
            targets,
            legal_mask,
            int(
                positive_count
            ),
            int(
                heldout_count
            ),
        )

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> LogisticPCA:
        """Fit LogisticPCA using the complete masked binary matrix."""
        _validate_interaction_frame(
            train_data,
            name="train_data",
            allow_empty=False,
        )

        heldout = (
            self._require_heldout_mask()
        )

        set_seed(
            self.seed,
            deterministic=True,
        )

        train = (
            train_data[
                [
                    "userId",
                    "movieId",
                ]
            ]
            .copy()
            .sort_values(
                [
                    "userId",
                    "movieId",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        train["userId"] = (
            train[
                "userId"
            ]
            .astype(
                "int64"
            )
        )

        train["movieId"] = (
            train[
                "movieId"
            ]
            .astype(
                "int64"
            )
        )

        heldout = (
            heldout[
                [
                    "userId",
                    "movieId",
                ]
            ]
            .copy()
            .sort_values(
                [
                    "userId",
                    "movieId",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        heldout["userId"] = (
            heldout[
                "userId"
            ]
            .astype(
                "int64"
            )
        )

        heldout["movieId"] = (
            heldout[
                "movieId"
            ]
            .astype(
                "int64"
            )
        )

        user_ids = np.sort(
            train[
                "userId"
            ].unique()
        ).astype(
            np.int64,
            copy=False,
        )

        item_ids = np.sort(
            train[
                "movieId"
            ].unique()
        ).astype(
            np.int64,
            copy=False,
        )

        user_to_index = {
            int(user_id): index
            for index, user_id
            in enumerate(
                user_ids
            )
        }

        item_to_index = {
            int(item_id): index
            for index, item_id
            in enumerate(
                item_ids
            )
        }

        self._user_ids = (
            user_ids
        )

        self._item_ids = (
            item_ids
        )

        self._user_to_index = (
            user_to_index
        )

        self._item_to_index = (
            item_to_index
        )

        (
            train_items_by_user,
            heldout_items_by_user,
        ) = self._prepare_histories(
            train,
            heldout,
        )

        self._train_items_by_user = (
            train_items_by_user
        )

        self._heldout_items_by_user = (
            heldout_items_by_user
        )

        n_users = len(
            user_ids
        )

        n_items = len(
            item_ids
        )

        total_matrix_cells = (
            n_users
            *
            n_items
        )

        positive_cell_count = int(
            len(
                train
            )
        )

        heldout_cell_count = int(
            len(
                heldout
            )
        )

        legal_cell_count = int(
            total_matrix_cells
            -
            heldout_cell_count
        )

        zero_cell_count = int(
            legal_cell_count
            -
            positive_cell_count
        )

        if legal_cell_count <= 0:
            raise RuntimeError(
                "LogisticPCA has no legal training cells"
            )

        if positive_cell_count <= 0:
            raise RuntimeError(
                "LogisticPCA has no positive training cells"
            )

        if zero_cell_count < 0:
            raise RuntimeError(
                "LogisticPCA legal-cell accounting is invalid"
            )

        self._positive_cell_count = (
            positive_cell_count
        )

        self._heldout_cell_count = (
            heldout_cell_count
        )

        self._legal_cell_count = (
            legal_cell_count
        )

        initial_item_intercepts = (
            self._initial_item_intercepts()
        )

        init_generator = (
            make_torch_generator(
                self.seed,
                "LogisticPCA.parameter_initialization",
                device="cpu",
            )
        )

        module = _LogisticPCAModule(
            n_users=n_users,
            n_items=n_items,
            n_factors=self.n_factors,
            initial_item_intercepts=(
                initial_item_intercepts
            ),
            init_generator=(
                init_generator
            ),
        )

        module = module.to(
            self.device
        )

        self._module = module

        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.learning_rate,
        )

        self._history = []

        n_batches = int(
            math.ceil(
                n_users
                /
                self.user_batch_size
            )
        )

        for epoch in range(
            self.epochs
        ):
            module.train()

            order_rng = make_numpy_rng(
                self.seed,
                (
                    "LogisticPCA."
                    f"user_order.epoch_{epoch}"
                ),
            )

            user_order = (
                order_rng.permutation(
                    n_users
                )
            )

            epoch_bce_sum = 0.0
            epoch_legal_cells = 0
            epoch_positive_cells = 0
            epoch_heldout_cells = 0
            max_abs_logit = 0.0
            batches = 0

            for start in range(
                0,
                n_users,
                self.user_batch_size,
            ):
                end = min(
                    start
                    +
                    self.user_batch_size,
                    n_users,
                )

                batch_user_indices_np = (
                    user_order[
                        start:end
                    ]
                    .astype(
                        np.int64,
                        copy=False,
                    )
                )

                batch_user_indices = (
                    torch.as_tensor(
                        batch_user_indices_np,
                        dtype=torch.long,
                        device=self.device,
                    )
                )

                (
                    targets,
                    legal_mask,
                    batch_positive_count,
                    batch_heldout_count,
                ) = self._build_user_batch_targets(
                    batch_user_indices_np
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = (
                    module.user_block_logits(
                        batch_user_indices
                    )
                )

                if logits.shape != (
                    len(
                        batch_user_indices_np
                    ),
                    n_items,
                ):
                    raise RuntimeError(
                        "LogisticPCA produced invalid "
                        "user-block logit shape"
                    )

                if not torch.isfinite(
                    logits
                ).all():
                    raise RuntimeError(
                        "LogisticPCA produced "
                        "non-finite logits"
                    )

                batch_max_abs_logit = float(
                    logits
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                    .item()
                )

                max_abs_logit = max(
                    max_abs_logit,
                    batch_max_abs_logit,
                )

                elementwise_bce = (
                    F.binary_cross_entropy_with_logits(
                        logits,
                        targets,
                        reduction="none",
                    )
                )

                legal_losses = (
                    elementwise_bce[
                        legal_mask
                    ]
                )

                if legal_losses.numel() == 0:
                    raise RuntimeError(
                        "LogisticPCA encountered "
                        "an empty legal minibatch"
                    )

                if not torch.isfinite(
                    legal_losses
                ).all():
                    raise RuntimeError(
                        "LogisticPCA produced "
                        "non-finite BCE values"
                    )

                batch_loss_sum = (
                    legal_losses.sum()
                )

                batch_legal_count = int(
                    legal_mask
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )

                # Partition-normalized stochastic objective:
                #
                # Averaging this quantity over all user batches,
                # with fixed parameters, reproduces the complete
                # masked likelihood mean exactly.
                loss = (
                    batch_loss_sum
                    /
                    float(
                        legal_cell_count
                    )
                    *
                    float(
                        n_batches
                    )
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "LogisticPCA produced "
                        "a non-finite training loss"
                    )

                loss.backward()

                if (
                    self.max_grad_norm
                    is not None
                ):
                    gradient_norm = (
                        torch.nn.utils.clip_grad_norm_(
                            module.parameters(),
                            max_norm=(
                                self.max_grad_norm
                            ),
                        )
                    )

                    if not torch.isfinite(
                        torch.as_tensor(
                            gradient_norm
                        )
                    ):
                        raise RuntimeError(
                            "LogisticPCA produced "
                            "a non-finite gradient norm"
                        )

                optimizer.step()

                for parameter in (
                    module.parameters()
                ):
                    if not torch.isfinite(
                        parameter
                    ).all():
                        raise RuntimeError(
                            "LogisticPCA produced "
                            "non-finite parameters"
                        )

                epoch_bce_sum += float(
                    batch_loss_sum
                    .detach()
                    .cpu()
                    .item()
                )

                epoch_legal_cells += (
                    batch_legal_count
                )

                epoch_positive_cells += int(
                    batch_positive_count
                )

                epoch_heldout_cells += int(
                    batch_heldout_count
                )

                batches += 1

            if (
                epoch_legal_cells
                !=
                legal_cell_count
            ):
                raise RuntimeError(
                    "LogisticPCA did not process "
                    "exactly the full masked legal matrix"
                )

            if (
                epoch_positive_cells
                !=
                positive_cell_count
            ):
                raise RuntimeError(
                    "LogisticPCA did not process "
                    "every training positive exactly once"
                )

            if (
                epoch_heldout_cells
                !=
                heldout_cell_count
            ):
                raise RuntimeError(
                    "LogisticPCA held-out mask "
                    "accounting is incorrect"
                )

            epoch_bce = (
                epoch_bce_sum
                /
                float(
                    epoch_legal_cells
                )
            )

            epoch_record: dict[
                str,
                float | int,
            ] = {
                "epoch": int(
                    epoch + 1
                ),
                "bce": float(
                    epoch_bce
                ),
                "total_loss": float(
                    epoch_bce
                ),
                "legal_cells": int(
                    epoch_legal_cells
                ),
                "positive_cells": int(
                    epoch_positive_cells
                ),
                "zero_cells": int(
                    epoch_legal_cells
                    -
                    epoch_positive_cells
                ),
                "heldout_masked_cells": int(
                    epoch_heldout_cells
                ),
                "max_abs_logit": float(
                    max_abs_logit
                ),
                "batches": int(
                    batches
                ),
            }

            for key in [
                "bce",
                "total_loss",
                "max_abs_logit",
            ]:
                if not np.isfinite(
                    float(
                        epoch_record[
                            key
                        ]
                    )
                ):
                    raise RuntimeError(
                        "LogisticPCA epoch diagnostics "
                        "contain non-finite values"
                    )

            self._history.append(
                epoch_record
            )

        module.eval()

        self._is_fitted = True

        return self

    def score(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return theta_ui for candidate items."""
        self._require_fitted()

        items = _validate_candidate_items(
            candidate_items
        )

        assert self._module is not None
        assert self._user_to_index is not None
        assert self._item_to_index is not None

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
            for item_id
            in items
            if int(item_id)
            not in self._item_to_index
        ]

        if unknown_items:
            raise ValueError(
                "candidate_items contains items "
                "outside the fitted training catalog: "
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
                for item_id
                in items
            ),
            dtype=np.int64,
            count=len(items),
        )

        user_tensor = torch.full(
            (
                len(items),
            ),
            user_index,
            dtype=torch.long,
            device=self.device,
        )

        item_tensor = torch.as_tensor(
            item_indices,
            dtype=torch.long,
            device=self.device,
        )

        self._module.eval()

        with torch.no_grad():
            logits = (
                self._module
                .pair_logits(
                    user_tensor,
                    item_tensor,
                )
            )

        scores = (
            logits
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=False,
            )
        )

        if scores.shape != (
            len(items),
        ):
            raise RuntimeError(
                "LogisticPCA returned "
                "an invalid score shape"
            )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                "LogisticPCA produced "
                "non-finite scores"
            )

        return scores

    def probability(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return Bernoulli probabilities for diagnostics."""
        scores = self.score(
            user_id,
            candidate_items,
        )

        # Stable sigmoid implemented using Torch.
        tensor = torch.as_tensor(
            scores,
            dtype=torch.float64,
        )

        probabilities = (
            torch.sigmoid(
                tensor
            )
            .numpy()
        )

        if not np.isfinite(
            probabilities
        ).all():
            raise RuntimeError(
                "LogisticPCA produced "
                "non-finite probabilities"
            )

        if np.any(
            probabilities < 0.0
        ) or np.any(
            probabilities > 1.0
        ):
            raise RuntimeError(
                "LogisticPCA probability "
                "outside [0, 1]"
            )

        return probabilities.astype(
            np.float64,
            copy=False,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted LogisticPCA parameters."""
        self._require_fitted()

        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "LogisticPCA save path "
                "must end in .npz"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        assert self._module is not None
        assert self._user_ids is not None
        assert self._item_ids is not None
        assert self._legal_cell_count is not None
        assert self._positive_cell_count is not None
        assert self._heldout_cell_count is not None

        np.savez_compressed(
            path,
            user_ids=(
                self._user_ids
            ),
            item_ids=(
                self._item_ids
            ),
            user_scores=(
                self._module
                .user_scores
                .weight
                .detach()
                .cpu()
                .numpy()
            ),
            item_basis=(
                self._module
                .item_basis
                .weight
                .detach()
                .cpu()
                .numpy()
            ),
            item_intercept=(
                self._module
                .item_intercept
                .detach()
                .cpu()
                .numpy()
            ),
        )

        metadata: dict[
            str,
            Any,
        ] = {
            "model_name": (
                self.model_name
            ),
            "spec_revision": (
                "v1.2"
            ),
            "formulation": (
                "schein_saul_ungar_"
                "low_rank_bernoulli"
            ),
            "objective": (
                "complete_masked_"
                "bernoulli_likelihood"
            ),
            "n_factors": (
                self.n_factors
            ),
            "learning_rate": (
                self.learning_rate
            ),
            "epochs": (
                self.epochs
            ),
            "user_batch_size": (
                self.user_batch_size
            ),
            "max_grad_norm": (
                self.max_grad_norm
            ),
            "intercept_epsilon": (
                self.intercept_epsilon
            ),
            "seed": (
                self.seed
            ),
            "training_device": str(
                self.device
            ),
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
            "legal_cell_count": int(
                self._legal_cell_count
            ),
            "positive_cell_count": int(
                self._positive_cell_count
            ),
            "zero_cell_count": int(
                self._legal_cell_count
                -
                self._positive_cell_count
            ),
            "heldout_cell_count": int(
                self._heldout_cell_count
            ),
            "l2_regularization": (
                False
            ),
            "orthogonality_constraint": (
                False
            ),
            "training_history": (
                self._history
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
    ) -> LogisticPCA:
        """Load LogisticPCA for deterministic scoring."""
        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "LogisticPCA load path "
                "must end in .npz"
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
                "Saved model is not LogisticPCA"
            )

        if metadata.get(
            "spec_revision"
        ) != "v1.2":
            raise ValueError(
                "Saved LogisticPCA does not use "
                "model-spec revision v1.2"
            )

        if metadata.get(
            "l2_regularization"
        ) is not False:
            raise ValueError(
                "Saved LogisticPCA unexpectedly "
                "uses L2 regularization"
            )

        if metadata.get(
            "orthogonality_constraint"
        ) is not False:
            raise ValueError(
                "Saved LogisticPCA unexpectedly "
                "uses an orthogonality constraint"
            )

        model = cls(
            heldout_interactions=None,
            n_factors=int(
                metadata[
                    "n_factors"
                ]
            ),
            learning_rate=float(
                metadata[
                    "learning_rate"
                ]
            ),
            epochs=int(
                metadata[
                    "epochs"
                ]
            ),
            user_batch_size=int(
                metadata[
                    "user_batch_size"
                ]
            ),
            max_grad_norm=(
                None
                if metadata[
                    "max_grad_norm"
                ] is None
                else float(
                    metadata[
                        "max_grad_norm"
                    ]
                )
            ),
            intercept_epsilon=float(
                metadata[
                    "intercept_epsilon"
                ]
            ),
            seed=int(
                metadata[
                    "seed"
                ]
            ),
            device="cpu",
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as state:
            user_ids = np.asarray(
                state[
                    "user_ids"
                ],
                dtype=np.int64,
            )

            item_ids = np.asarray(
                state[
                    "item_ids"
                ],
                dtype=np.int64,
            )

            user_scores = np.asarray(
                state[
                    "user_scores"
                ],
                dtype=np.float32,
            )

            item_basis = np.asarray(
                state[
                    "item_basis"
                ],
                dtype=np.float32,
            )

            item_intercept = np.asarray(
                state[
                    "item_intercept"
                ],
                dtype=np.float32,
            )

        n_users = len(
            user_ids
        )

        n_items = len(
            item_ids
        )

        n_factors = (
            model.n_factors
        )

        if user_scores.shape != (
            n_users,
            n_factors,
        ):
            raise ValueError(
                "Saved LogisticPCA user-score "
                "matrix has invalid shape"
            )

        if item_basis.shape != (
            n_items,
            n_factors,
        ):
            raise ValueError(
                "Saved LogisticPCA item-basis "
                "matrix has invalid shape"
            )

        if item_intercept.shape != (
            n_items,
        ):
            raise ValueError(
                "Saved LogisticPCA item-intercept "
                "vector has invalid shape"
            )

        arrays = [
            user_scores,
            item_basis,
            item_intercept,
        ]

        if any(
            not np.isfinite(
                array
            ).all()
            for array in arrays
        ):
            raise ValueError(
                "Saved LogisticPCA state "
                "contains non-finite values"
            )

        if len(
            np.unique(
                user_ids
            )
        ) != n_users:
            raise ValueError(
                "Saved LogisticPCA user IDs "
                "contain duplicates"
            )

        if len(
            np.unique(
                item_ids
            )
        ) != n_items:
            raise ValueError(
                "Saved LogisticPCA item IDs "
                "contain duplicates"
            )

        init_generator = (
            make_torch_generator(
                model.seed,
                "LogisticPCA.load_initialization",
                device="cpu",
            )
        )

        module = _LogisticPCAModule(
            n_users=n_users,
            n_items=n_items,
            n_factors=n_factors,
            initial_item_intercepts=(
                item_intercept
            ),
            init_generator=(
                init_generator
            ),
        )

        with torch.no_grad():
            module.user_scores.weight.copy_(
                torch.from_numpy(
                    user_scores
                )
            )

            module.item_basis.weight.copy_(
                torch.from_numpy(
                    item_basis
                )
            )

            module.item_intercept.copy_(
                torch.from_numpy(
                    item_intercept
                )
            )

        module.eval()

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

        model._module = (
            module
        )

        model._legal_cell_count = int(
            metadata[
                "legal_cell_count"
            ]
        )

        model._positive_cell_count = int(
            metadata[
                "positive_cell_count"
            ]
        )

        model._heldout_cell_count = int(
            metadata[
                "heldout_cell_count"
            ]
        )

        model._history = [
            dict(row)
            for row
            in metadata.get(
                "training_history",
                [],
            )
        ]

        model._is_fitted = True

        return model