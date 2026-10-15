"""Bernoulli matrix factorization for Paper 2.

This module implements the frozen BernoulliMF specification.

The model uses:

    eta_ui = mu + b_u + b_i + p_u^T q_i

    p_ui = sigmoid(eta_ui)

Training uses:

- canonical training positives;
- the shared leakage-safe NegativeSampler;
- explicit sampled-negative weighting;
- logits-based binary cross-entropy;
- L2 regularization;
- deterministic initialization;
- deterministic minibatch ordering;
- deterministic negative-sampling streams;
- finite-value checks;
- optional gradient clipping.

The shared NegativeSampler must be created outside this model from:

    train
    all_observed = train + validation + test

This ensures that held-out validation/test positives can never be sampled
as training negatives.

The model exposes the Paper 2 scoring contract:

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
from torch.nn import functional as F

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

    required = [
        "userId",
        "movieId",
    ]

    if train_data[
        required
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
    """Validate candidate items."""
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


class _BernoulliMFModule(
    nn.Module,
):
    """Internal PyTorch Bernoulli MF parameterization."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
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
            torch.zeros(
                (),
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
        """Return Bernoulli logits."""
        user_vectors = (
            self.user_factors(
                user_indices
            )
        )

        item_vectors = (
            self.item_factors(
                item_indices
            )
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


class BernoulliMF:
    """Implicit-feedback Bernoulli matrix factorization."""

    model_name = "BernoulliMF"

    def __init__(
        self,
        negative_sampler: NegativeSampler | None = None,
        n_factors: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 5,
        batch_size: int = 1024,
        negative_ratio: int = 5,
        negative_weight: float = 1.0,
        factor_l2: float = 1e-6,
        bias_l2: float = 1e-6,
        negative_strategy: str = "uniform",
        negative_replace: bool = True,
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
                negative_ratio,
                int,
            )
            or isinstance(
                negative_ratio,
                bool,
            )
            or negative_ratio <= 0
        ):
            raise ValueError(
                "negative_ratio must be a positive integer"
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if negative_weight <= 0:
            raise ValueError(
                "negative_weight must be positive"
            )

        if factor_l2 < 0:
            raise ValueError(
                "factor_l2 must be non-negative"
            )

        if bias_l2 < 0:
            raise ValueError(
                "bias_l2 must be non-negative"
            )

        if negative_strategy not in {
            "uniform",
            "popularity",
        }:
            raise ValueError(
                "negative_strategy must be "
                "'uniform' or 'popularity'"
            )

        if not isinstance(
            negative_replace,
            bool,
        ):
            raise TypeError(
                "negative_replace must be bool"
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
        self.negative_ratio = (
            negative_ratio
        )
        self.negative_weight = float(
            negative_weight
        )
        self.factor_l2 = float(
            factor_l2
        )
        self.bias_l2 = float(
            bias_l2
        )
        self.negative_strategy = (
            negative_strategy
        )
        self.negative_replace = (
            negative_replace
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
            _BernoulliMFModule | None
        ) = None

        self._history: list[
            dict[str, float | int]
        ] = []

        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether model fitting completed."""
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
    def training_history(
        self,
    ) -> list[dict[str, float | int]]:
        """Return epoch-level training diagnostics."""
        return [
            dict(row)
            for row in self._history
        ]

    def _require_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "BernoulliMF must be fitted "
                "before scoring"
            )

    def _require_sampler(
        self,
    ) -> NegativeSampler:
        if self.negative_sampler is None:
            raise RuntimeError(
                "BernoulliMF requires the shared "
                "NegativeSampler during fit()"
            )

        return self.negative_sampler

    def _validate_sampler(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Verify that the shared sampler covers the training observations."""
        sampler = self._require_sampler()

        assert self._item_ids is not None

        catalog_set = set(
            int(item_id)
            for item_id
            in self._item_ids
        )

        grouped = frame.groupby(
            "userId",
            sort=True,
        )

        for user_id, group in grouped:
            user_id = int(
                user_id
            )

            train_items = set(
                int(item)
                for item
                in group[
                    "movieId"
                ].to_numpy()
            )

            observed = (
                sampler.observed_items(
                    user_id
                )
            )

            missing_from_observed = (
                train_items
                - set(observed)
            )

            if missing_from_observed:
                raise ValueError(
                    "NegativeSampler does not mark "
                    "all training interactions as observed "
                    f"for user {user_id}"
                )

            available = (
                sampler.available_items(
                    user_id
                )
            )

            available_set = set(
                int(item)
                for item in available
            )

            if (
                train_items
                & available_set
            ):
                raise ValueError(
                    "NegativeSampler exposes a training "
                    "positive as an available negative "
                    f"for user {user_id}"
                )

            unknown = (
                available_set
                - catalog_set
            )

            if unknown:
                raise ValueError(
                    "NegativeSampler returned items "
                    "outside the canonical training catalog"
                )

    def _regularization(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return factor, bias, and combined regularization terms."""
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

        regularization = (
            self.factor_l2
            *
            factor_penalty
            +
            self.bias_l2
            *
            bias_penalty
        )

        return (
            factor_penalty,
            bias_penalty,
            regularization,
        )

    def _build_epoch_negatives(
        self,
        positive_user_ids: np.ndarray,
        epoch: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """Draw one complete deterministic epoch of legal negatives."""
        sampler = self._require_sampler()

        assert self._user_to_index is not None
        assert self._item_to_index is not None

        unique_users, counts = np.unique(
            positive_user_ids,
            return_counts=True,
        )

        negative_user_parts: list[
            np.ndarray
        ] = []

        negative_item_parts: list[
            np.ndarray
        ] = []

        for raw_user_id, positive_count in zip(
            unique_users,
            counts,
        ):
            user_id = int(
                raw_user_id
            )

            n_draws = int(
                positive_count
                *
                self.negative_ratio
            )

            sample = sampler.sample(
                user_id=user_id,
                n=n_draws,
                strategy=self.negative_strategy,
                replace=self.negative_replace,
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
                    "unexpected number of items"
                )

            observed = (
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
                    "NegativeSampler produced "
                    "an observed interaction"
                )

            try:
                mapped_items = np.fromiter(
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
            except KeyError as exc:
                raise RuntimeError(
                    "NegativeSampler produced an item "
                    "outside the training catalog"
                ) from exc

            user_index = (
                self._user_to_index[
                    user_id
                ]
            )

            negative_user_parts.append(
                np.full(
                    n_draws,
                    user_index,
                    dtype=np.int64,
                )
            )

            negative_item_parts.append(
                mapped_items
            )

        negative_users = np.concatenate(
            negative_user_parts
        )

        negative_items = np.concatenate(
            negative_item_parts
        )

        expected = (
            len(
                positive_user_ids
            )
            *
            self.negative_ratio
        )

        if negative_users.shape != (
            expected,
        ):
            raise RuntimeError(
                "Negative user-index array "
                "has incorrect shape"
            )

        if negative_items.shape != (
            expected,
        ):
            raise RuntimeError(
                "Negative item-index array "
                "has incorrect shape"
            )

        return (
            negative_users,
            negative_items,
        )

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> BernoulliMF:
        """Fit BernoulliMF using canonical positives and shared negatives."""
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

        frame["userId"] = frame[
            "userId"
        ].astype(
            "int64"
        )

        frame["movieId"] = frame[
            "movieId"
        ].astype(
            "int64"
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

        positive_user_ids = frame[
            "userId"
        ].to_numpy(
            dtype=np.int64,
        )

        positive_item_ids = frame[
            "movieId"
        ].to_numpy(
            dtype=np.int64,
        )

        positive_user_indices = np.fromiter(
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

        positive_item_indices = np.fromiter(
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

        init_generator = (
            make_torch_generator(
                self.seed,
                "BernoulliMF.parameter_initialization",
                device="cpu",
            )
        )

        module = _BernoulliMFModule(
            n_users=len(
                user_ids
            ),
            n_items=len(
                item_ids
            ),
            n_factors=self.n_factors,
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

        n_positives = len(
            positive_user_indices
        )

        for epoch in range(
            self.epochs
        ):
            module.train()

            (
                negative_user_indices,
                negative_item_indices,
            ) = self._build_epoch_negatives(
                positive_user_ids=(
                    positive_user_ids
                ),
                epoch=epoch,
            )

            positive_rng = make_numpy_rng(
                self.seed,
                (
                    "BernoulliMF."
                    f"positive_order.epoch_{epoch}"
                ),
            )

            negative_rng = make_numpy_rng(
                self.seed,
                (
                    "BernoulliMF."
                    f"negative_order.epoch_{epoch}"
                ),
            )

            positive_order = (
                positive_rng.permutation(
                    n_positives
                )
            )

            negative_order = (
                negative_rng.permutation(
                    len(
                        negative_user_indices
                    )
                )
            )

            epoch_positive_sum = 0.0
            epoch_negative_sum = 0.0
            epoch_total_sum = 0.0

            positive_seen = 0
            negative_seen = 0

            batch_number = 0

            for start in range(
                0,
                n_positives,
                self.batch_size,
            ):
                end = min(
                    start
                    +
                    self.batch_size,
                    n_positives,
                )

                pos_batch_order = (
                    positive_order[
                        start:end
                    ]
                )

                neg_start = (
                    start
                    *
                    self.negative_ratio
                )

                neg_end = (
                    end
                    *
                    self.negative_ratio
                )

                neg_batch_order = (
                    negative_order[
                        neg_start:neg_end
                    ]
                )

                pos_users = torch.as_tensor(
                    positive_user_indices[
                        pos_batch_order
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                pos_items = torch.as_tensor(
                    positive_item_indices[
                        pos_batch_order
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                neg_users = torch.as_tensor(
                    negative_user_indices[
                        neg_batch_order
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                neg_items = torch.as_tensor(
                    negative_item_indices[
                        neg_batch_order
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                positive_logits = module(
                    pos_users,
                    pos_items,
                )

                negative_logits = module(
                    neg_users,
                    neg_items,
                )

                positive_loss = (
                    F.binary_cross_entropy_with_logits(
                        positive_logits,
                        torch.ones_like(
                            positive_logits
                        ),
                        reduction="mean",
                    )
                )

                negative_loss = (
                    F.binary_cross_entropy_with_logits(
                        negative_logits,
                        torch.zeros_like(
                            negative_logits
                        ),
                        reduction="mean",
                    )
                )

                (
                    _,
                    _,
                    regularization,
                ) = self._regularization()

                loss = (
                    positive_loss
                    +
                    self.negative_weight
                    *
                    negative_loss
                    +
                    regularization
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "BernoulliMF produced "
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
                            "BernoulliMF produced "
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
                            "BernoulliMF produced "
                            "non-finite parameters"
                        )

                pos_count = int(
                    len(
                        pos_batch_order
                    )
                )

                neg_count = int(
                    len(
                        neg_batch_order
                    )
                )

                epoch_positive_sum += (
                    float(
                        positive_loss
                        .detach()
                        .cpu()
                        .item()
                    )
                    *
                    pos_count
                )

                epoch_negative_sum += (
                    float(
                        negative_loss
                        .detach()
                        .cpu()
                        .item()
                    )
                    *
                    neg_count
                )

                epoch_total_sum += (
                    float(
                        loss
                        .detach()
                        .cpu()
                        .item()
                    )
                    *
                    pos_count
                )

                positive_seen += (
                    pos_count
                )

                negative_seen += (
                    neg_count
                )

                batch_number += 1

            if positive_seen != (
                n_positives
            ):
                raise RuntimeError(
                    "BernoulliMF did not process "
                    "all positive examples"
                )

            expected_negatives = (
                n_positives
                *
                self.negative_ratio
            )

            if negative_seen != (
                expected_negatives
            ):
                raise RuntimeError(
                    "BernoulliMF did not process "
                    "the expected number of negatives"
                )

            epoch_record: dict[
                str,
                float | int,
            ] = {
                "epoch": int(
                    epoch + 1
                ),
                "positive_loss": (
                    epoch_positive_sum
                    /
                    positive_seen
                ),
                "negative_loss": (
                    epoch_negative_sum
                    /
                    negative_seen
                ),
                "total_loss": (
                    epoch_total_sum
                    /
                    positive_seen
                ),
                "positive_examples": int(
                    positive_seen
                ),
                "negative_examples": int(
                    negative_seen
                ),
                "batches": int(
                    batch_number
                ),
            }

            if not all(
                np.isfinite(
                    float(value)
                )
                for key, value
                in epoch_record.items()
                if key
                not in {
                    "epoch",
                    "positive_examples",
                    "negative_examples",
                    "batches",
                }
            ):
                raise RuntimeError(
                    "BernoulliMF epoch history "
                    "contains non-finite values"
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
        """Return Bernoulli logits for candidate items."""
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
            for item_id in items
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
            logits = self._module(
                user_tensor,
                item_tensor,
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
                "BernoulliMF returned "
                "an invalid score shape"
            )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                "BernoulliMF produced "
                "non-finite scores"
            )

        return scores

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted BernoulliMF state.

        The main state is stored in NPZ format.
        Metadata is stored beside it as JSON.
        """
        self._require_fitted()

        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "BernoulliMF save path "
                "must end in .npz"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        assert self._module is not None
        assert self._user_ids is not None
        assert self._item_ids is not None

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
            "negative_ratio": (
                self.negative_ratio
            ),
            "negative_weight": (
                self.negative_weight
            ),
            "factor_l2": (
                self.factor_l2
            ),
            "bias_l2": (
                self.bias_l2
            ),
            "negative_strategy": (
                self.negative_strategy
            ),
            "negative_replace": (
                self.negative_replace
            ),
            "max_grad_norm": (
                self.max_grad_norm
            ),
            "seed": (
                self.seed
            ),
            "training_device": (
                str(
                    self.device
                )
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
    ) -> BernoulliMF:
        """Load BernoulliMF for deterministic scoring.

        The shared NegativeSampler is not needed after fitting, so loaded
        models use negative_sampler=None.
        """
        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "BernoulliMF load path "
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
                "Saved model is not BernoulliMF"
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
            negative_ratio=int(
                metadata[
                    "negative_ratio"
                ]
            ),
            negative_weight=float(
                metadata[
                    "negative_weight"
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
            negative_strategy=str(
                metadata[
                    "negative_strategy"
                ]
            ),
            negative_replace=bool(
                metadata[
                    "negative_replace"
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
                "Saved user-factor matrix "
                "has invalid shape"
            )

        if item_factors.shape != (
            n_items,
            n_factors,
        ):
            raise ValueError(
                "Saved item-factor matrix "
                "has invalid shape"
            )

        if user_bias.shape != (
            n_users,
            1,
        ):
            raise ValueError(
                "Saved user-bias matrix "
                "has invalid shape"
            )

        if item_bias.shape != (
            n_items,
            1,
        ):
            raise ValueError(
                "Saved item-bias matrix "
                "has invalid shape"
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
                "Saved BernoulliMF state "
                "contains non-finite values"
            )

        if not np.isfinite(
            global_intercept
        ):
            raise ValueError(
                "Saved global intercept "
                "is non-finite"
            )

        if len(
            np.unique(
                user_ids
            )
        ) != n_users:
            raise ValueError(
                "Saved user IDs contain duplicates"
            )

        if len(
            np.unique(
                item_ids
            )
        ) != n_items:
            raise ValueError(
                "Saved item IDs contain duplicates"
            )

        init_generator = (
            make_torch_generator(
                model.seed,
                "BernoulliMF.load_initialization",
                device="cpu",
            )
        )

        module = _BernoulliMFModule(
            n_users=n_users,
            n_items=n_items,
            n_factors=n_factors,
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

        raw_history = metadata.get(
            "training_history",
            [],
        )

        model._history = [
            dict(row)
            for row in raw_history
        ]

        model._is_fitted = True

        return model