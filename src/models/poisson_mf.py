"""Poisson matrix factorization for Paper 2.

This module implements the corrected primary PoissonMF model defined in
docs/model_spec.md revision v1.1.

Model
-----

    eta_ui = mu + b_u + b_i + p_u^T q_i

    lambda_ui = exp(eta_ui)

The implicit occurrence target is

    y_ui in {0, 1}

where canonical training interactions have y=1 and genuinely unobserved
user-item cells have y=0.

The conceptual data objective is the masked full Poisson negative
log-likelihood over:

- every canonical training positive; and
- every genuinely unobserved zero in the canonical training catalog.

Validation/test positives are masked and never treated as zero counts.

For computational efficiency, legal zeros are sampled with the shared
NegativeSampler. Uniform zero sampling uses the frozen unbiased estimator

    |Z_u| / m_u * sum_s lambda_{u,j_s}

where Z_u is the user's complete legal-zero set and m_u is the number of
sampled zeros for that user.

The implementation constructs stochastic examples whose sample weights make
the minibatch mean an unbiased estimator of the corrected full data objective.

Primary PoissonMF currently supports uniform zero sampling. A future
nonuniform proposal must expose its proposal probabilities and apply explicit
inverse-probability weighting before it can be used by this primary model.

Common Paper 2 contract:

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
import torch
from torch import nn

from src.reproducibility import (
    make_numpy_rng,
    make_torch_generator,
    set_seed,
)
from src.sampling import NegativeSampler


REQUIRED_TRAIN_COLUMNS = {
    "userId",
    "movieId",
}


def _validate_train_frame(
    train_data: pd.DataFrame,
) -> None:
    """Validate the canonical training frame."""
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
        [
            "userId",
            "movieId",
        ]
    ].isna().any().any():
        raise ValueError(
            "train_data contains missing userId/movieId values"
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


def _validate_candidate_items(
    candidate_items: np.ndarray,
) -> np.ndarray:
    """Validate and normalize a candidate-item array."""
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


class _PoissonMFModule(
    nn.Module,
):
    """Internal PyTorch parameterization for PoissonMF."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        initial_intercept: float,
        init_generator: torch.Generator,
    ) -> None:
        super().__init__()

        self.user_factors = nn.Embedding(
            n_users,
            n_factors,
        )

        self.item_factors = nn.Embedding(
            n_items,
            n_factors,
        )

        self.user_bias = nn.Embedding(
            n_users,
            1,
        )

        self.item_bias = nn.Embedding(
            n_items,
            1,
        )

        self.global_intercept = nn.Parameter(
            torch.tensor(
                float(
                    initial_intercept
                ),
                dtype=torch.float32,
            )
        )

        nn.init.normal_(
            self.user_factors.weight,
            mean=0.0,
            std=0.01,
            generator=init_generator,
        )

        nn.init.normal_(
            self.item_factors.weight,
            mean=0.0,
            std=0.01,
            generator=init_generator,
        )

        nn.init.zeros_(
            self.user_bias.weight
        )

        nn.init.zeros_(
            self.item_bias.weight
        )

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw Poisson log-rates eta_ui."""
        user_vectors = self.user_factors(
            user_indices
        )

        item_vectors = self.item_factors(
            item_indices
        )

        interaction = (
            user_vectors
            *
            item_vectors
        ).sum(
            dim=1
        )

        user_bias = (
            self.user_bias(
                user_indices
            )
            .squeeze(1)
        )

        item_bias = (
            self.item_bias(
                item_indices
            )
            .squeeze(1)
        )

        return (
            self.global_intercept
            +
            user_bias
            +
            item_bias
            +
            interaction
        )


class PoissonMF:
    """Corrected implicit-occurrence Poisson matrix factorization."""

    model_name = "PoissonMF"

    def __init__(
        self,
        negative_sampler: NegativeSampler | None = None,
        n_factors: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 5,
        batch_size: int = 1024,
        zero_ratio: int = 5,
        factor_l2: float = 1e-6,
        bias_l2: float = 1e-6,
        zero_strategy: str = "uniform",
        zero_replace: bool = True,
        log_rate_min: float = -20.0,
        log_rate_max: float = 12.0,
        max_grad_norm: float | None = 5.0,
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
                batch_size,
                int,
            )
            or isinstance(
                batch_size,
                bool,
            )
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size must be a positive integer"
            )

        if (
            not isinstance(
                zero_ratio,
                int,
            )
            or isinstance(
                zero_ratio,
                bool,
            )
            or zero_ratio <= 0
        ):
            raise ValueError(
                "zero_ratio must be a positive integer"
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if factor_l2 < 0:
            raise ValueError(
                "factor_l2 must be non-negative"
            )

        if bias_l2 < 0:
            raise ValueError(
                "bias_l2 must be non-negative"
            )

        # B91:
        # Nonuniform proposals require explicit q_u(j) values.
        # The current shared sampler interface does not expose them.
        if zero_strategy != "uniform":
            raise ValueError(
                "Primary PoissonMF currently supports only "
                "zero_strategy='uniform'. Nonuniform zero "
                "sampling requires explicit proposal "
                "probabilities for inverse-probability weighting."
            )

        if not isinstance(
            zero_replace,
            bool,
        ):
            raise TypeError(
                "zero_replace must be bool"
            )

        if not np.isfinite(
            log_rate_min
        ):
            raise ValueError(
                "log_rate_min must be finite"
            )

        if not np.isfinite(
            log_rate_max
        ):
            raise ValueError(
                "log_rate_max must be finite"
            )

        if log_rate_min >= log_rate_max:
            raise ValueError(
                "log_rate_min must be smaller than log_rate_max"
            )

        if (
            max_grad_norm is not None
            and max_grad_norm <= 0
        ):
            raise ValueError(
                "max_grad_norm must be positive or None"
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

        self.negative_sampler = (
            negative_sampler
        )

        self.n_factors = n_factors
        self.learning_rate = float(
            learning_rate
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.zero_ratio = zero_ratio

        self.factor_l2 = float(
            factor_l2
        )

        self.bias_l2 = float(
            bias_l2
        )

        self.zero_strategy = (
            zero_strategy
        )

        self.zero_replace = (
            zero_replace
        )

        self.log_rate_min = float(
            log_rate_min
        )

        self.log_rate_max = float(
            log_rate_max
        )

        self.max_grad_norm = (
            None
            if max_grad_norm is None
            else float(
                max_grad_norm
            )
        )

        self.seed = seed
        self.device = torch_device

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

        self._module: (
            _PoissonMFModule | None
        ) = None

        self._history: list[
            dict[str, float | int]
        ] = []

        self._full_legal_cells: (
            int | None
        ) = None

        self._is_fitted = False

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Whether model fitting completed."""
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
        """Return the fitted training item catalog."""
        self._require_fitted()

        assert self._item_ids is not None

        return self._item_ids.copy()

    @property
    def training_history(
        self,
    ) -> list[
        dict[str, float | int]
    ]:
        """Return defensive copy of epoch diagnostics."""
        return [
            dict(row)
            for row in self._history
        ]

    @property
    def full_legal_cells(
        self,
    ) -> int:
        """Return M from the frozen full Poisson objective."""
        self._require_fitted()

        assert self._full_legal_cells is not None

        return int(
            self._full_legal_cells
        )

    def _require_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "PoissonMF must be fitted before scoring"
            )

    def _require_sampler(
        self,
    ) -> NegativeSampler:
        if self.negative_sampler is None:
            raise RuntimeError(
                "PoissonMF requires the shared "
                "NegativeSampler during fit()"
            )

        return self.negative_sampler

    def _validate_sampler(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Check sampler consistency with training positives/catalog."""
        sampler = self._require_sampler()

        assert self._item_ids is not None

        catalog_set = set(
            int(item_id)
            for item_id
            in self._item_ids
        )

        for user_id, group in frame.groupby(
            "userId",
            sort=True,
        ):
            user_id = int(
                user_id
            )

            train_items = set(
                int(item_id)
                for item_id
                in group[
                    "movieId"
                ].to_numpy()
            )

            observed = set(
                sampler.observed_items(
                    user_id
                )
            )

            if not train_items.issubset(
                observed
            ):
                raise ValueError(
                    "NegativeSampler does not mark all "
                    "training positives as observed for "
                    f"user {user_id}"
                )

            available = np.asarray(
                sampler.available_items(
                    user_id
                ),
                dtype=np.int64,
            )

            available_set = set(
                int(item_id)
                for item_id
                in available
            )

            if (
                train_items
                &
                available_set
            ):
                raise ValueError(
                    "NegativeSampler exposes a training "
                    "positive as a legal zero for "
                    f"user {user_id}"
                )

            unknown = (
                available_set
                -
                catalog_set
            )

            if unknown:
                raise ValueError(
                    "NegativeSampler exposes items outside "
                    "the canonical training catalog"
                )

    def _regularization(
        self,
    ) -> torch.Tensor:
        """Return frozen factor/bias L2 regularizer."""
        assert self._module is not None

        factor_penalty = (
            self._module
            .user_factors
            .weight
            .pow(2)
            .sum()
            +
            self._module
            .item_factors
            .weight
            .pow(2)
            .sum()
        )

        bias_penalty = (
            self._module
            .user_bias
            .weight
            .pow(2)
            .sum()
            +
            self._module
            .item_bias
            .weight
            .pow(2)
            .sum()
        )

        return (
            self.factor_l2
            *
            factor_penalty
            +
            self.bias_l2
            *
            bias_penalty
        )

    def _calculate_legal_zero_sizes(
        self,
        user_ids: np.ndarray,
    ) -> dict[int, int]:
        """Return |Z_u| for each fitted user."""
        sampler = self._require_sampler()

        zero_sizes: dict[
            int,
            int,
        ] = {}

        for raw_user_id in user_ids:
            user_id = int(
                raw_user_id
            )

            available = np.asarray(
                sampler.available_items(
                    user_id
                ),
                dtype=np.int64,
            )

            zero_sizes[
                user_id
            ] = int(
                len(
                    available
                )
            )

        return zero_sizes

    def _build_epoch_examples(
        self,
        positive_user_ids: np.ndarray,
        positive_user_indices: np.ndarray,
        positive_item_indices: np.ndarray,
        zero_sizes: dict[int, int],
        full_legal_cells: int,
        epoch: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, int | float],
    ]:
        """Construct one unbiased stochastic representation of the full objective.

        For positive examples, the raw cell weight is 1.

        For m_u uniformly sampled legal zeros of user u, every sampled zero
        receives raw weight

            |Z_u| / m_u.

        The raw weights therefore sum exactly to M:

            |R_train| + sum_u |Z_u|.

        If N is the number of stochastic examples, multiplying each raw weight
        by N/M makes the ordinary minibatch mean an unbiased estimator of the
        frozen full data objective.
        """
        sampler = self._require_sampler()

        assert self._user_to_index is not None
        assert self._item_to_index is not None

        unique_users, positive_counts = np.unique(
            positive_user_ids,
            return_counts=True,
        )

        negative_user_parts: list[
            np.ndarray
        ] = []

        negative_item_parts: list[
            np.ndarray
        ] = []

        negative_raw_weight_parts: list[
            np.ndarray
        ] = []

        total_zero_draws = 0

        for (
            raw_user_id,
            raw_positive_count,
        ) in zip(
            unique_users,
            positive_counts,
        ):
            user_id = int(
                raw_user_id
            )

            positive_count = int(
                raw_positive_count
            )

            zero_size = int(
                zero_sizes[
                    user_id
                ]
            )

            if zero_size == 0:
                continue

            requested_draws = (
                positive_count
                *
                self.zero_ratio
            )

            if self.zero_replace:
                n_draws = (
                    requested_draws
                )
            else:
                n_draws = min(
                    requested_draws,
                    zero_size,
                )

            if n_draws <= 0:
                raise RuntimeError(
                    "A user with legal zeros received "
                    "zero Poisson zero draws"
                )

            sample = sampler.sample(
                user_id=user_id,
                n=n_draws,
                strategy=self.zero_strategy,
                replace=self.zero_replace,
                draw_key=(
                    self.model_name,
                    self.seed,
                    "epoch",
                    int(epoch),
                    "user",
                    user_id,
                ),
            )

            sampled_items = np.asarray(
                sample.items,
                dtype=np.int64,
            )

            if sampled_items.shape != (
                n_draws,
            ):
                raise RuntimeError(
                    "NegativeSampler returned an "
                    "unexpected number of zero items"
                )

            observed = set(
                sampler.observed_items(
                    user_id
                )
            )

            if any(
                int(item_id)
                in observed
                for item_id
                in sampled_items
            ):
                raise RuntimeError(
                    "PoissonMF sampled an observed "
                    "interaction as a zero"
                )

            try:
                sampled_item_indices = (
                    np.fromiter(
                        (
                            self._item_to_index[
                                int(item_id)
                            ]
                            for item_id
                            in sampled_items
                        ),
                        dtype=np.int64,
                        count=n_draws,
                    )
                )
            except KeyError as exc:
                raise RuntimeError(
                    "PoissonMF zero sampler returned "
                    "an item outside the training catalog"
                ) from exc

            user_index = (
                self._user_to_index[
                    user_id
                ]
            )

            raw_zero_weight = (
                float(
                    zero_size
                )
                /
                float(
                    n_draws
                )
            )

            negative_user_parts.append(
                np.full(
                    n_draws,
                    user_index,
                    dtype=np.int64,
                )
            )

            negative_item_parts.append(
                sampled_item_indices
            )

            negative_raw_weight_parts.append(
                np.full(
                    n_draws,
                    raw_zero_weight,
                    dtype=np.float64,
                )
            )

            total_zero_draws += (
                n_draws
            )

        if negative_user_parts:
            negative_users = (
                np.concatenate(
                    negative_user_parts
                )
            )

            negative_items = (
                np.concatenate(
                    negative_item_parts
                )
            )

            negative_raw_weights = (
                np.concatenate(
                    negative_raw_weight_parts
                )
            )
        else:
            negative_users = np.empty(
                0,
                dtype=np.int64,
            )

            negative_items = np.empty(
                0,
                dtype=np.int64,
            )

            negative_raw_weights = np.empty(
                0,
                dtype=np.float64,
            )

        n_positives = len(
            positive_user_indices
        )

        positive_targets = np.ones(
            n_positives,
            dtype=np.float32,
        )

        negative_targets = np.zeros(
            len(
                negative_users
            ),
            dtype=np.float32,
        )

        positive_raw_weights = np.ones(
            n_positives,
            dtype=np.float64,
        )

        all_users = np.concatenate(
            [
                positive_user_indices,
                negative_users,
            ]
        )

        all_items = np.concatenate(
            [
                positive_item_indices,
                negative_items,
            ]
        )

        all_targets = np.concatenate(
            [
                positive_targets,
                negative_targets,
            ]
        )

        all_raw_weights = np.concatenate(
            [
                positive_raw_weights,
                negative_raw_weights,
            ]
        )

        n_examples = int(
            len(
                all_users
            )
        )

        if n_examples <= 0:
            raise RuntimeError(
                "PoissonMF constructed no training examples"
            )

        # For the corrected estimator, raw weights must sum exactly to M.
        raw_weight_sum = float(
            all_raw_weights.sum()
        )

        if not np.isclose(
            raw_weight_sum,
            float(
                full_legal_cells
            ),
            rtol=1e-10,
            atol=1e-6,
        ):
            raise RuntimeError(
                "PoissonMF importance weights do not "
                "reconstruct the full legal-cell count"
            )

        scale = (
            float(
                n_examples
            )
            /
            float(
                full_legal_cells
            )
        )

        sample_weights = (
            all_raw_weights
            *
            scale
        )

        if not np.isfinite(
            sample_weights
        ).all():
            raise RuntimeError(
                "PoissonMF generated non-finite "
                "importance weights"
            )

        if np.any(
            sample_weights <= 0
        ):
            raise RuntimeError(
                "PoissonMF generated non-positive "
                "importance weights"
            )

        # This identity is useful because it proves that the stochastic
        # minibatch sample-weight mean remains normalized.
        if not np.isclose(
            float(
                sample_weights.mean()
            ),
            1.0,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(
                "PoissonMF stochastic sample weights "
                "do not have mean one"
            )

        stats: dict[
            str,
            int | float,
        ] = {
            "positive_examples": int(
                n_positives
            ),
            "zero_draws": int(
                total_zero_draws
            ),
            "stochastic_examples": int(
                n_examples
            ),
            "full_legal_cells": int(
                full_legal_cells
            ),
            "raw_weight_sum": float(
                raw_weight_sum
            ),
            "sample_weight_mean": float(
                sample_weights.mean()
            ),
        }

        return (
            all_users,
            all_items,
            all_targets,
            sample_weights,
            stats,
        )

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> PoissonMF:
        """Fit corrected PoissonMF."""
        _validate_train_frame(
            train_data
        )

        self._require_sampler()

        set_seed(
            self.seed,
            deterministic=True,
        )

        frame = (
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

        frame["userId"] = (
            frame[
                "userId"
            ]
            .astype(
                "int64"
            )
        )

        frame["movieId"] = (
            frame[
                "movieId"
            ]
            .astype(
                "int64"
            )
        )

        user_ids = np.sort(
            frame[
                "userId"
            ].unique()
        ).astype(
            np.int64,
            copy=False,
        )

        item_ids = np.sort(
            frame[
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

        self._validate_sampler(
            frame
        )

        positive_user_ids = (
            frame[
                "userId"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        positive_item_ids = (
            frame[
                "movieId"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        positive_user_indices = (
            np.fromiter(
                (
                    user_to_index[
                        int(user_id)
                    ]
                    for user_id
                    in positive_user_ids
                ),
                dtype=np.int64,
                count=len(frame),
            )
        )

        positive_item_indices = (
            np.fromiter(
                (
                    item_to_index[
                        int(item_id)
                    ]
                    for item_id
                    in positive_item_ids
                ),
                dtype=np.int64,
                count=len(frame),
            )
        )

        zero_sizes = (
            self._calculate_legal_zero_sizes(
                user_ids
            )
        )

        full_zero_cells = int(
            sum(
                zero_sizes.values()
            )
        )

        n_positives = int(
            len(
                frame
            )
        )

        full_legal_cells = (
            n_positives
            +
            full_zero_cells
        )

        if full_legal_cells <= 0:
            raise RuntimeError(
                "PoissonMF full legal training domain is empty"
            )

        self._full_legal_cells = (
            full_legal_cells
        )

        mean_count = (
            float(
                n_positives
            )
            /
            float(
                full_legal_cells
            )
        )

        if not (
            0.0
            <
            mean_count
            <=
            1.0
        ):
            raise RuntimeError(
                "Invalid PoissonMF initial mean count"
            )

        initial_intercept = float(
            np.log(
                mean_count
            )
        )

        init_generator = (
            make_torch_generator(
                self.seed,
                "PoissonMF.parameter_initialization",
                device="cpu",
            )
        )

        module = _PoissonMFModule(
            n_users=len(
                user_ids
            ),
            n_items=len(
                item_ids
            ),
            n_factors=self.n_factors,
            initial_intercept=(
                initial_intercept
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

        for epoch in range(
            self.epochs
        ):
            module.train()

            (
                epoch_users,
                epoch_items,
                epoch_targets,
                epoch_weights,
                epoch_stats,
            ) = self._build_epoch_examples(
                positive_user_ids=(
                    positive_user_ids
                ),
                positive_user_indices=(
                    positive_user_indices
                ),
                positive_item_indices=(
                    positive_item_indices
                ),
                zero_sizes=(
                    zero_sizes
                ),
                full_legal_cells=(
                    full_legal_cells
                ),
                epoch=(
                    epoch
                ),
            )

            n_examples = len(
                epoch_users
            )

            order_rng = make_numpy_rng(
                self.seed,
                (
                    "PoissonMF."
                    f"training_order.epoch_{epoch}"
                ),
            )

            order = order_rng.permutation(
                n_examples
            )

            weighted_data_sum = 0.0
            total_loss_sum = 0.0

            clamp_events = 0
            examples_seen = 0
            batches = 0

            for start in range(
                0,
                n_examples,
                self.batch_size,
            ):
                end = min(
                    start
                    +
                    self.batch_size,
                    n_examples,
                )

                batch_order = (
                    order[
                        start:end
                    ]
                )

                users = torch.as_tensor(
                    epoch_users[
                        batch_order
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                items = torch.as_tensor(
                    epoch_items[
                        batch_order
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                targets = torch.as_tensor(
                    epoch_targets[
                        batch_order
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

                weights = torch.as_tensor(
                    epoch_weights[
                        batch_order
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                raw_eta = module(
                    users,
                    items,
                )

                if not torch.isfinite(
                    raw_eta
                ).all():
                    raise RuntimeError(
                        "PoissonMF produced non-finite log-rates"
                    )

                clamp_mask = (
                    (raw_eta < self.log_rate_min)
                    |
                    (raw_eta > self.log_rate_max)
                )

                clamp_events += int(
                    clamp_mask
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )

                guarded_eta = torch.clamp(
                    raw_eta,
                    min=self.log_rate_min,
                    max=self.log_rate_max,
                )

                rates = torch.exp(
                    guarded_eta
                )

                if not torch.isfinite(
                    rates
                ).all():
                    raise RuntimeError(
                        "PoissonMF produced non-finite rates"
                    )

                if torch.any(
                    rates <= 0
                ):
                    raise RuntimeError(
                        "PoissonMF produced a non-positive rate"
                    )

                # y! = 1 for y in {0,1}, so log(y!) = 0.
                nll = (
                    rates
                    -
                    targets
                    *
                    guarded_eta
                )

                if not torch.isfinite(
                    nll
                ).all():
                    raise RuntimeError(
                        "PoissonMF produced non-finite NLL values"
                    )

                weighted_nll = (
                    weights
                    *
                    nll
                )

                data_loss = (
                    weighted_nll
                    .mean()
                )

                regularization = (
                    self._regularization()
                )

                loss = (
                    data_loss
                    +
                    regularization
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "PoissonMF produced "
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
                            "PoissonMF produced "
                            "a non-finite gradient norm"
                        )

                optimizer.step()

                for parameter in module.parameters():
                    if not torch.isfinite(
                        parameter
                    ).all():
                        raise RuntimeError(
                            "PoissonMF produced "
                            "non-finite parameters"
                        )

                batch_size_actual = int(
                    len(
                        batch_order
                    )
                )

                weighted_data_sum += float(
                    weighted_nll
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )

                total_loss_sum += (
                    float(
                        loss
                        .detach()
                        .cpu()
                        .item()
                    )
                    *
                    batch_size_actual
                )

                examples_seen += (
                    batch_size_actual
                )

                batches += 1

            if examples_seen != (
                n_examples
            ):
                raise RuntimeError(
                    "PoissonMF did not process every "
                    "stochastic training example"
                )

            epoch_data_loss = (
                weighted_data_sum
                /
                float(
                    n_examples
                )
            )

            epoch_total_loss = (
                total_loss_sum
                /
                float(
                    n_examples
                )
            )

            clamp_fraction = (
                float(
                    clamp_events
                )
                /
                float(
                    n_examples
                )
            )

            epoch_record: dict[
                str,
                float | int,
            ] = {
                "epoch": int(
                    epoch + 1
                ),
                "data_loss": float(
                    epoch_data_loss
                ),
                "total_loss": float(
                    epoch_total_loss
                ),
                "positive_examples": int(
                    epoch_stats[
                        "positive_examples"
                    ]
                ),
                "zero_draws": int(
                    epoch_stats[
                        "zero_draws"
                    ]
                ),
                "stochastic_examples": int(
                    epoch_stats[
                        "stochastic_examples"
                    ]
                ),
                "full_legal_cells": int(
                    epoch_stats[
                        "full_legal_cells"
                    ]
                ),
                "raw_weight_sum": float(
                    epoch_stats[
                        "raw_weight_sum"
                    ]
                ),
                "sample_weight_mean": float(
                    epoch_stats[
                        "sample_weight_mean"
                    ]
                ),
                "clamp_events": int(
                    clamp_events
                ),
                "clamp_fraction": float(
                    clamp_fraction
                ),
                "batches": int(
                    batches
                ),
            }

            finite_fields = [
                "data_loss",
                "total_loss",
                "raw_weight_sum",
                "sample_weight_mean",
                "clamp_fraction",
            ]

            if not all(
                np.isfinite(
                    float(
                        epoch_record[
                            key
                        ]
                    )
                )
                for key
                in finite_fields
            ):
                raise RuntimeError(
                    "PoissonMF epoch diagnostics "
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
        """Return raw Poisson log-rate scores.

        The frozen ranking score is eta_ui itself. The exponential transform
        is monotonic, so ranking by eta_ui is equivalent to ranking by the
        conceptual Poisson rate.
        """
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
            raw_eta = self._module(
                user_tensor,
                item_tensor,
            )

        scores = (
            raw_eta
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
                "PoissonMF returned an invalid score shape"
            )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                "PoissonMF produced non-finite scores"
            )

        return scores

    def rate(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return numerically guarded Poisson rates for diagnostics."""
        scores = self.score(
            user_id,
            candidate_items,
        )

        guarded = np.clip(
            scores,
            self.log_rate_min,
            self.log_rate_max,
        )

        rates = np.exp(
            guarded
        )

        if not np.isfinite(
            rates
        ).all():
            raise RuntimeError(
                "PoissonMF produced non-finite diagnostic rates"
            )

        if np.any(
            rates <= 0
        ):
            raise RuntimeError(
                "PoissonMF produced non-positive diagnostic rates"
            )

        return rates.astype(
            np.float64,
            copy=False,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted PoissonMF state and metadata."""
        self._require_fitted()

        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "PoissonMF save path must end in .npz"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        assert self._module is not None
        assert self._user_ids is not None
        assert self._item_ids is not None
        assert self._full_legal_cells is not None

        module = self._module

        np.savez_compressed(
            path,
            user_ids=(
                self._user_ids
            ),
            item_ids=(
                self._item_ids
            ),
            user_factors=(
                module
                .user_factors
                .weight
                .detach()
                .cpu()
                .numpy()
            ),
            item_factors=(
                module
                .item_factors
                .weight
                .detach()
                .cpu()
                .numpy()
            ),
            user_bias=(
                module
                .user_bias
                .weight
                .detach()
                .cpu()
                .numpy()
            ),
            item_bias=(
                module
                .item_bias
                .weight
                .detach()
                .cpu()
                .numpy()
            ),
            global_intercept=np.asarray(
                float(
                    module
                    .global_intercept
                    .detach()
                    .cpu()
                    .item()
                ),
                dtype=np.float64,
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
                "v1.1"
            ),
            "objective": (
                "masked_full_poisson_with_"
                "unbiased_uniform_zero_subsampling"
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
            "batch_size": (
                self.batch_size
            ),
            "zero_ratio": (
                self.zero_ratio
            ),
            "factor_l2": (
                self.factor_l2
            ),
            "bias_l2": (
                self.bias_l2
            ),
            "zero_strategy": (
                self.zero_strategy
            ),
            "zero_replace": (
                self.zero_replace
            ),
            "log_rate_min": (
                self.log_rate_min
            ),
            "log_rate_max": (
                self.log_rate_max
            ),
            "max_grad_norm": (
                self.max_grad_norm
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
            "full_legal_cells": int(
                self._full_legal_cells
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
    ) -> PoissonMF:
        """Load a fitted PoissonMF for deterministic scoring."""
        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "PoissonMF load path must end in .npz"
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
                "Saved model is not PoissonMF"
            )

        if metadata.get(
            "spec_revision"
        ) != "v1.1":
            raise ValueError(
                "Saved PoissonMF does not use "
                "model-spec revision v1.1"
            )

        model = cls(
            negative_sampler=None,
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
            batch_size=int(
                metadata[
                    "batch_size"
                ]
            ),
            zero_ratio=int(
                metadata[
                    "zero_ratio"
                ]
            ),
            factor_l2=float(
                metadata[
                    "factor_l2"
                ]
            ),
            bias_l2=float(
                metadata[
                    "bias_l2"
                ]
            ),
            zero_strategy=str(
                metadata[
                    "zero_strategy"
                ]
            ),
            zero_replace=bool(
                metadata[
                    "zero_replace"
                ]
            ),
            log_rate_min=float(
                metadata[
                    "log_rate_min"
                ]
            ),
            log_rate_max=float(
                metadata[
                    "log_rate_max"
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

            user_factors = np.asarray(
                state[
                    "user_factors"
                ],
                dtype=np.float32,
            )

            item_factors = np.asarray(
                state[
                    "item_factors"
                ],
                dtype=np.float32,
            )

            user_bias = np.asarray(
                state[
                    "user_bias"
                ],
                dtype=np.float32,
            )

            item_bias = np.asarray(
                state[
                    "item_bias"
                ],
                dtype=np.float32,
            )

            global_intercept = float(
                np.asarray(
                    state[
                        "global_intercept"
                    ]
                ).item()
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

        if user_factors.shape != (
            n_users,
            n_factors,
        ):
            raise ValueError(
                "Saved PoissonMF user-factor "
                "matrix has invalid shape"
            )

        if item_factors.shape != (
            n_items,
            n_factors,
        ):
            raise ValueError(
                "Saved PoissonMF item-factor "
                "matrix has invalid shape"
            )

        if user_bias.shape != (
            n_users,
            1,
        ):
            raise ValueError(
                "Saved PoissonMF user-bias "
                "matrix has invalid shape"
            )

        if item_bias.shape != (
            n_items,
            1,
        ):
            raise ValueError(
                "Saved PoissonMF item-bias "
                "matrix has invalid shape"
            )

        arrays = [
            user_factors,
            item_factors,
            user_bias,
            item_bias,
        ]

        if any(
            not np.isfinite(
                array
            ).all()
            for array in arrays
        ):
            raise ValueError(
                "Saved PoissonMF state contains "
                "non-finite values"
            )

        if not np.isfinite(
            global_intercept
        ):
            raise ValueError(
                "Saved PoissonMF global intercept "
                "is non-finite"
            )

        if len(
            np.unique(
                user_ids
            )
        ) != n_users:
            raise ValueError(
                "Saved PoissonMF user IDs "
                "contain duplicates"
            )

        if len(
            np.unique(
                item_ids
            )
        ) != n_items:
            raise ValueError(
                "Saved PoissonMF item IDs "
                "contain duplicates"
            )

        init_generator = (
            make_torch_generator(
                model.seed,
                "PoissonMF.load_initialization",
                device="cpu",
            )
        )

        module = _PoissonMFModule(
            n_users=n_users,
            n_items=n_items,
            n_factors=n_factors,
            initial_intercept=(
                global_intercept
            ),
            init_generator=(
                init_generator
            ),
        )

        with torch.no_grad():
            module.user_factors.weight.copy_(
                torch.from_numpy(
                    user_factors
                )
            )

            module.item_factors.weight.copy_(
                torch.from_numpy(
                    item_factors
                )
            )

            module.user_bias.weight.copy_(
                torch.from_numpy(
                    user_bias
                )
            )

            module.item_bias.weight.copy_(
                torch.from_numpy(
                    item_bias
                )
            )

            module.global_intercept.copy_(
                torch.tensor(
                    global_intercept,
                    dtype=torch.float32,
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

        model._full_legal_cells = int(
            metadata[
                "full_legal_cells"
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