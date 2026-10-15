"""Gaussian matrix factorization for Paper 2.

This module implements the frozen GaussianMF specification.

Model
-----

For observed training rating r_ui,

    r_hat_ui = mu + b_u + b_i + p_u^T q_i

with

    r_ui ~ Normal(r_hat_ui, sigma^2).

For fixed variance, fitting minimizes mean squared error plus L2
regularization.

Important Paper 2 rules:

- only canonical training ratings are used for fitting;
- validation/test ratings never enter training;
- the global intercept is initialized from the training-only mean rating;
- no negative sampling is used;
- ranking score is the predicted rating itself;
- native diagnostics are RMSE and MAE;
- initialization and minibatch ordering are deterministic.

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


REQUIRED_TRAIN_COLUMNS = {
    "userId",
    "movieId",
    "rating",
}


def _validate_train_frame(
    train_data: pd.DataFrame,
) -> None:
    """Validate canonical explicit-rating training data."""
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
            "train_data contains missing "
            "userId/movieId/rating values"
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

    ratings = pd.to_numeric(
        train_data[
            "rating"
        ],
        errors="coerce",
    ).to_numpy(
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
    """Validate candidate-item array."""
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


class _GaussianMFModule(
    nn.Module,
):
    """Internal PyTorch GaussianMF parameterization."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        initial_global_mean: float,
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

        self.global_mean = nn.Parameter(
            torch.tensor(
                float(
                    initial_global_mean
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
        """Return predicted explicit ratings."""
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
            self.global_mean
            +
            user_bias
            +
            item_bias
            +
            interaction
        )


class GaussianMF:
    """Explicit-rating Gaussian matrix factorization."""

    model_name = "GaussianMF"

    def __init__(
        self,
        n_factors: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 5,
        batch_size: int = 1024,
        factor_l2: float = 1e-6,
        bias_l2: float = 1e-6,
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

        self.n_factors = n_factors

        self.learning_rate = float(
            learning_rate
        )

        self.epochs = epochs

        self.batch_size = batch_size

        self.factor_l2 = float(
            factor_l2
        )

        self.bias_l2 = float(
            bias_l2
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
            _GaussianMFModule | None
        ) = None

        self._training_mean: (
            float | None
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
    def training_mean(
        self,
    ) -> float:
        """Return training-only mean used for initialization."""
        self._require_fitted()

        assert self._training_mean is not None

        return float(
            self._training_mean
        )

    @property
    def training_history(
        self,
    ) -> list[
        dict[str, float | int]
    ]:
        """Return epoch diagnostics."""
        return [
            dict(row)
            for row in self._history
        ]

    def _require_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "GaussianMF must be fitted before scoring"
            )

    def _regularization(
        self,
    ) -> torch.Tensor:
        """Return factor/bias L2 regularizer."""
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

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> GaussianMF:
        """Fit GaussianMF on canonical observed training ratings."""
        _validate_train_frame(
            train_data
        )

        set_seed(
            self.seed,
            deterministic=True,
        )

        frame = (
            train_data[
                [
                    "userId",
                    "movieId",
                    "rating",
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

        frame["rating"] = (
            frame[
                "rating"
            ]
            .astype(
                "float64"
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

        training_mean = float(
            frame[
                "rating"
            ].mean()
        )

        if not np.isfinite(
            training_mean
        ):
            raise RuntimeError(
                "GaussianMF training mean is non-finite"
            )

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

        self._training_mean = (
            training_mean
        )

        user_indices = np.fromiter(
            (
                user_to_index[
                    int(user_id)
                ]
                for user_id
                in frame[
                    "userId"
                ].to_numpy(
                    dtype=np.int64,
                )
            ),
            dtype=np.int64,
            count=len(frame),
        )

        item_indices = np.fromiter(
            (
                item_to_index[
                    int(item_id)
                ]
                for item_id
                in frame[
                    "movieId"
                ].to_numpy(
                    dtype=np.int64,
                )
            ),
            dtype=np.int64,
            count=len(frame),
        )

        ratings = frame[
            "rating"
        ].to_numpy(
            dtype=np.float32,
        )

        init_generator = (
            make_torch_generator(
                self.seed,
                "GaussianMF.parameter_initialization",
                device="cpu",
            )
        )

        module = _GaussianMFModule(
            n_users=len(
                user_ids
            ),
            n_items=len(
                item_ids
            ),
            n_factors=self.n_factors,
            initial_global_mean=(
                training_mean
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

        n_examples = len(
            frame
        )

        for epoch in range(
            self.epochs
        ):
            module.train()

            order_rng = make_numpy_rng(
                self.seed,
                (
                    "GaussianMF."
                    f"training_order.epoch_{epoch}"
                ),
            )

            order = order_rng.permutation(
                n_examples
            )

            squared_error_sum = 0.0
            absolute_error_sum = 0.0
            total_loss_sum = 0.0

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
                    user_indices[
                        batch_order
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                items = torch.as_tensor(
                    item_indices[
                        batch_order
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                targets = torch.as_tensor(
                    ratings[
                        batch_order
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                predictions = module(
                    users,
                    items,
                )

                if not torch.isfinite(
                    predictions
                ).all():
                    raise RuntimeError(
                        "GaussianMF produced "
                        "non-finite predictions"
                    )

                errors = (
                    predictions
                    -
                    targets
                )

                mse = (
                    errors
                    .pow(2)
                    .mean()
                )

                regularization = (
                    self._regularization()
                )

                loss = (
                    mse
                    +
                    regularization
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "GaussianMF produced "
                        "a non-finite loss"
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
                            "GaussianMF produced "
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
                            "GaussianMF produced "
                            "non-finite parameters"
                        )

                batch_count = int(
                    len(
                        batch_order
                    )
                )

                squared_error_sum += float(
                    errors
                    .pow(2)
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )

                absolute_error_sum += float(
                    errors
                    .abs()
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
                    batch_count
                )

                examples_seen += (
                    batch_count
                )

                batches += 1

            if examples_seen != (
                n_examples
            ):
                raise RuntimeError(
                    "GaussianMF did not process "
                    "all training examples"
                )

            mse_value = (
                squared_error_sum
                /
                examples_seen
            )

            rmse_value = float(
                np.sqrt(
                    mse_value
                )
            )

            mae_value = (
                absolute_error_sum
                /
                examples_seen
            )

            total_loss_value = (
                total_loss_sum
                /
                examples_seen
            )

            epoch_record: dict[
                str,
                float | int,
            ] = {
                "epoch": int(
                    epoch + 1
                ),
                "mse": float(
                    mse_value
                ),
                "rmse": float(
                    rmse_value
                ),
                "mae": float(
                    mae_value
                ),
                "total_loss": float(
                    total_loss_value
                ),
                "examples": int(
                    examples_seen
                ),
                "batches": int(
                    batches
                ),
            }

            for key in [
                "mse",
                "rmse",
                "mae",
                "total_loss",
            ]:
                if not np.isfinite(
                    float(
                        epoch_record[
                            key
                        ]
                    )
                ):
                    raise RuntimeError(
                        "GaussianMF epoch diagnostics "
                        "contain non-finite values"
                    )

            self._history.append(
                epoch_record
            )

        module.eval()

        self._is_fitted = True

        return self

    def _map_pairs(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """Map raw IDs into fitted tensor indices."""
        self._require_fitted()

        assert self._user_to_index is not None
        assert self._item_to_index is not None

        raw_users = np.asarray(
            user_ids,
            dtype=np.int64,
        )

        raw_items = np.asarray(
            item_ids,
            dtype=np.int64,
        )

        if raw_users.ndim != 1:
            raise ValueError(
                "user_ids must be one-dimensional"
            )

        if raw_items.ndim != 1:
            raise ValueError(
                "item_ids must be one-dimensional"
            )

        if raw_users.shape != (
            raw_items.shape
        ):
            raise ValueError(
                "user_ids and item_ids must have "
                "identical shape"
            )

        unknown_users = [
            int(user_id)
            for user_id
            in np.unique(
                raw_users
            )
            if int(user_id)
            not in self._user_to_index
        ]

        if unknown_users:
            raise ValueError(
                "Unknown user IDs: "
                f"{unknown_users[:10]}"
            )

        unknown_items = [
            int(item_id)
            for item_id
            in np.unique(
                raw_items
            )
            if int(item_id)
            not in self._item_to_index
        ]

        if unknown_items:
            raise ValueError(
                "Unknown item IDs: "
                f"{unknown_items[:10]}"
            )

        mapped_users = np.fromiter(
            (
                self._user_to_index[
                    int(user_id)
                ]
                for user_id
                in raw_users
            ),
            dtype=np.int64,
            count=len(
                raw_users
            ),
        )

        mapped_items = np.fromiter(
            (
                self._item_to_index[
                    int(item_id)
                ]
                for item_id
                in raw_items
            ),
            dtype=np.int64,
            count=len(
                raw_items
            ),
        )

        return (
            mapped_users,
            mapped_items,
        )

    def predict_pairs(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        """Predict ratings for aligned user-item pairs."""
        self._require_fitted()

        assert self._module is not None

        (
            mapped_users,
            mapped_items,
        ) = self._map_pairs(
            user_ids,
            item_ids,
        )

        users = torch.as_tensor(
            mapped_users,
            dtype=torch.long,
            device=self.device,
        )

        items = torch.as_tensor(
            mapped_items,
            dtype=torch.long,
            device=self.device,
        )

        self._module.eval()

        with torch.no_grad():
            predictions = (
                self._module(
                    users,
                    items,
                )
            )

        output = (
            predictions
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=False,
            )
        )

        if output.shape != (
            len(
                mapped_users
            ),
        ):
            raise RuntimeError(
                "GaussianMF returned invalid "
                "pair-prediction shape"
            )

        if not np.isfinite(
            output
        ).all():
            raise RuntimeError(
                "GaussianMF produced "
                "non-finite predictions"
            )

        return output

    def score(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return predicted ratings for ranking candidates."""
        self._require_fitted()

        items = _validate_candidate_items(
            candidate_items
        )

        user_id = int(
            user_id
        )

        user_array = np.full(
            len(items),
            user_id,
            dtype=np.int64,
        )

        scores = self.predict_pairs(
            user_array,
            items,
        )

        if scores.shape != (
            len(items),
        ):
            raise RuntimeError(
                "GaussianMF returned invalid score shape"
            )

        return scores

    def native_metrics(
        self,
        data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate RMSE and MAE on an explicit-rating frame.

        This method does not alter model parameters.
        """
        self._require_fitted()

        required = {
            "userId",
            "movieId",
            "rating",
        }

        missing = (
            required
            -
            set(
                data.columns
            )
        )

        if missing:
            raise ValueError(
                "native_metrics data is missing columns: "
                f"{sorted(missing)}"
            )

        if len(data) == 0:
            raise ValueError(
                "native_metrics data must not be empty"
            )

        ratings = pd.to_numeric(
            data[
                "rating"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=np.float64,
        )

        if not np.isfinite(
            ratings
        ).all():
            raise ValueError(
                "native_metrics data contains "
                "non-finite ratings"
            )

        predictions = self.predict_pairs(
            data[
                "userId"
            ].to_numpy(
                dtype=np.int64,
            ),
            data[
                "movieId"
            ].to_numpy(
                dtype=np.int64,
            ),
        )

        errors = (
            predictions
            -
            ratings
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    errors
                )
            )
        )

        if not (
            np.isfinite(
                rmse
            )
            and
            np.isfinite(
                mae
            )
        ):
            raise RuntimeError(
                "GaussianMF native metrics are non-finite"
            )

        return {
            "rmse": rmse,
            "mae": mae,
        }

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted GaussianMF state."""
        self._require_fitted()

        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "GaussianMF save path must end in .npz"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        assert self._module is not None
        assert self._user_ids is not None
        assert self._item_ids is not None
        assert self._training_mean is not None

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
            global_mean=np.asarray(
                float(
                    module
                    .global_mean
                    .detach()
                    .cpu()
                    .item()
                ),
                dtype=np.float64,
            ),
            training_mean=np.asarray(
                self._training_mean,
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
            "factor_l2": (
                self.factor_l2
            ),
            "bias_l2": (
                self.bias_l2
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
    ) -> GaussianMF:
        """Load GaussianMF for deterministic scoring."""
        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "GaussianMF load path must end in .npz"
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
                "Saved model is not GaussianMF"
            )

        model = cls(
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

            global_mean = float(
                np.asarray(
                    state[
                        "global_mean"
                    ]
                ).item()
            )

            training_mean = float(
                np.asarray(
                    state[
                        "training_mean"
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
                "Saved GaussianMF user-factor "
                "matrix has invalid shape"
            )

        if item_factors.shape != (
            n_items,
            n_factors,
        ):
            raise ValueError(
                "Saved GaussianMF item-factor "
                "matrix has invalid shape"
            )

        if user_bias.shape != (
            n_users,
            1,
        ):
            raise ValueError(
                "Saved GaussianMF user-bias "
                "matrix has invalid shape"
            )

        if item_bias.shape != (
            n_items,
            1,
        ):
            raise ValueError(
                "Saved GaussianMF item-bias "
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
                "Saved GaussianMF state contains "
                "non-finite values"
            )

        if not np.isfinite(
            global_mean
        ):
            raise ValueError(
                "Saved GaussianMF global mean "
                "is non-finite"
            )

        if not np.isfinite(
            training_mean
        ):
            raise ValueError(
                "Saved GaussianMF training mean "
                "is non-finite"
            )

        if len(
            np.unique(
                user_ids
            )
        ) != n_users:
            raise ValueError(
                "Saved GaussianMF user IDs "
                "contain duplicates"
            )

        if len(
            np.unique(
                item_ids
            )
        ) != n_items:
            raise ValueError(
                "Saved GaussianMF item IDs "
                "contain duplicates"
            )

        init_generator = (
            make_torch_generator(
                model.seed,
                "GaussianMF.load_initialization",
                device="cpu",
            )
        )

        module = _GaussianMFModule(
            n_users=n_users,
            n_items=n_items,
            n_factors=n_factors,
            initial_global_mean=(
                global_mean
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

            module.global_mean.copy_(
                torch.tensor(
                    global_mean,
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

        model._training_mean = (
            training_mean
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