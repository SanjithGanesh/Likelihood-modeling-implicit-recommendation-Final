"""Gaussian exponential-family embedding model for Paper 2.

Model name
----------
g-emb

Target
------

g-emb predicts explicit MovieLens star ratings.

For user u and target item i:

    r_hat_ui
        = mu
        + b_i
        + t_i^T h_ui

where:

- mu is the canonical training-rating mean;
- b_i is an item bias;
- t_i is an unconstrained target embedding;
- h_ui is constructed from unconstrained context embeddings.

Training context
----------------

For an observed training rating (u, i, r_ui):

    context = H_u \\ {i}

The target item is excluded from its own context.

Evaluation context
------------------

For validation/test ranking or rating prediction:

    context = H_u

where H_u contains canonical training-history items only.

Objective
---------

With fixed Gaussian variance, minimizing Gaussian negative
log-likelihood is equivalent to minimizing mean squared error:

    MSE
        = (1 / |R_train|)
          sum_(u,i,r) (r_ui - r_hat_ui)^2

plus:

    lambda_t * ||T||^2
    + lambda_c * ||C||^2
    + lambda_b * ||b||^2

No negative sampling is used.

Optimization
------------

The primary EF-EMB optimizer is PyTorch Adagrad.

Ranking
-------

The common recommendation score is the predicted rating r_hat_ui.

Native diagnostics are RMSE and MAE.
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

from src.models.efemb import (
    EFContextStore,
    normalization_exponent,
)
from src.reproducibility import (
    make_numpy_rng,
    make_torch_generator,
    set_seed,
)


REQUIRED_COLUMNS = {
    "userId",
    "movieId",
    "rating",
}


def _validate_rating_frame(
    data: pd.DataFrame,
    *,
    name: str,
    allow_empty: bool,
) -> None:
    """Validate a frame containing explicit ratings."""
    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{name} must be a pandas DataFrame"
        )

    missing = (
        REQUIRED_COLUMNS
        -
        set(
            data.columns
        )
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
            "rating",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{name} contains missing "
            "userId/movieId/rating values"
        )

    if data.duplicated(
        subset=[
            "userId",
            "movieId",
        ]
    ).any():
        raise ValueError(
            f"{name} contains duplicate "
            "user-item pairs"
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
            f"{name} contains non-finite ratings"
        )


def _validate_heldout_frame(
    data: pd.DataFrame,
    *,
    name: str,
) -> None:
    """Validate held-out user/item identities."""
    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{name} must be a pandas DataFrame"
        )

    required = {
        "userId",
        "movieId",
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
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
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
            f"{name} contains duplicate "
            "user-item pairs"
        )


def _validate_candidate_items(
    candidate_items: np.ndarray,
) -> np.ndarray:
    """Validate candidate item IDs."""
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
        np.unique(
            items
        )
    ) != len(items):
        raise ValueError(
            "candidate_items contains duplicate IDs"
        )

    return items


class _GEmbModule(
    nn.Module,
):
    """Trainable parameters for Gaussian EF-EMB."""

    def __init__(
        self,
        *,
        n_items: int,
        n_factors: int,
        init_generator: torch.Generator,
    ) -> None:
        super().__init__()

        self.target_embeddings = nn.Embedding(
            n_items,
            n_factors,
        )

        self.context_embeddings = nn.Embedding(
            n_items,
            n_factors,
        )

        self.item_bias = nn.Embedding(
            n_items,
            1,
        )

        nn.init.normal_(
            self.target_embeddings.weight,
            mean=0.0,
            std=0.01,
            generator=init_generator,
        )

        nn.init.normal_(
            self.context_embeddings.weight,
            mean=0.0,
            std=0.01,
            generator=init_generator,
        )

        nn.init.zeros_(
            self.item_bias.weight
        )

    def predict(
        self,
        item_indices: torch.Tensor,
        context_vectors: torch.Tensor,
        global_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Return mu + b_i + t_i^T h_ui."""
        targets = (
            self.target_embeddings(
                item_indices
            )
        )

        interaction = (
            targets
            *
            context_vectors
        ).sum(
            dim=1
        )

        bias = (
            self.item_bias(
                item_indices
            )
            .squeeze(
                1
            )
        )

        return (
            global_mean
            +
            bias
            +
            interaction
        )


class GaussianEFEmb:
    """Gaussian exponential-family embedding model."""

    model_name = "g-emb"

    def __init__(
        self,
        *,
        heldout_interactions: pd.DataFrame | None = None,
        n_factors: int = 32,
        learning_rate: float = 0.05,
        epochs: int = 5,
        batch_size: int = 2048,
        context_normalization: str = "mean",
        target_l2: float = 1e-6,
        context_l2: float = 1e-6,
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
            not np.isfinite(
                learning_rate
            )
            or learning_rate <= 0.0
        ):
            raise ValueError(
                "learning_rate must be finite and positive"
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

        normalization_exponent(
            context_normalization
        )

        regularizers = {
            "target_l2": target_l2,
            "context_l2": context_l2,
            "bias_l2": bias_l2,
        }

        for name, value in regularizers.items():
            if (
                not np.isfinite(
                    value
                )
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be finite "
                    "and non-negative"
                )

        if (
            max_grad_norm is not None
            and (
                not np.isfinite(
                    max_grad_norm
                )
                or max_grad_norm <= 0.0
            )
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
                "CUDA requested but CUDA is unavailable"
            )

        if heldout_interactions is not None:
            _validate_heldout_frame(
                heldout_interactions,
                name="heldout_interactions",
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

        self.n_factors = (
            n_factors
        )

        self.learning_rate = float(
            learning_rate
        )

        self.epochs = (
            epochs
        )

        self.batch_size = (
            batch_size
        )

        self.context_normalization = (
            context_normalization
        )

        self.target_l2 = float(
            target_l2
        )

        self.context_l2 = float(
            context_l2
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

        self.seed = (
            seed
        )

        self.device = (
            torch_device
        )

        self._context_store: (
            EFContextStore | None
        ) = None

        self._module: (
            _GEmbModule | None
        ) = None

        self._incidence: (
            torch.Tensor | None
        ) = None

        self._evaluation_context_cache: (
            np.ndarray | None
        ) = None

        self._global_mean: (
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
        return self._is_fitted

    @property
    def user_ids(
        self,
    ) -> np.ndarray:
        self._require_fitted()

        assert self._context_store is not None

        return (
            self._context_store
            .user_ids
        )

    @property
    def item_catalog(
        self,
    ) -> np.ndarray:
        self._require_fitted()

        assert self._context_store is not None

        return (
            self._context_store
            .item_catalog
        )

    @property
    def global_mean(
        self,
    ) -> float:
        self._require_fitted()

        assert self._global_mean is not None

        return float(
            self._global_mean
        )

    @property
    def training_history(
        self,
    ) -> list[
        dict[str, float | int]
    ]:
        return [
            dict(row)
            for row
            in self._history
        ]

    def _require_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "g-emb must be fitted before scoring"
            )

    def _regularization(
        self,
    ) -> torch.Tensor:
        assert self._module is not None

        target_penalty = (
            self._module
            .target_embeddings
            .weight
            .pow(
                2
            )
            .sum()
        )

        context_penalty = (
            self._module
            .context_embeddings
            .weight
            .pow(
                2
            )
            .sum()
        )

        bias_penalty = (
            self._module
            .item_bias
            .weight
            .pow(
                2
            )
            .sum()
        )

        return (
            self.target_l2
            *
            target_penalty
            +
            self.context_l2
            *
            context_penalty
            +
            self.bias_l2
            *
            bias_penalty
        )

    def _refresh_evaluation_context_cache(
        self,
    ) -> None:
        """Cache complete train-history contexts."""
        assert self._context_store is not None
        assert self._module is not None
        assert self._incidence is not None

        self._module.eval()

        all_users = torch.arange(
            self._context_store
            .metadata
            .n_users,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            user_sums = (
                self._context_store
                .aggregate_user_sums(
                    self._module
                    .context_embeddings
                    .weight,
                    incidence=(
                        self._incidence
                    ),
                )
            )

            contexts = (
                self._context_store
                .context_vectors(
                    context_embeddings=(
                        self._module
                        .context_embeddings
                        .weight
                    ),
                    user_indices=(
                        all_users
                    ),
                    target_item_indices=None,
                    exclude_target=False,
                    normalization=(
                        self.context_normalization
                    ),
                    incidence=(
                        self._incidence
                    ),
                    precomputed_user_sums=(
                        user_sums
                    ),
                )
            )

        if not torch.isfinite(
            contexts
        ).all():
            raise RuntimeError(
                "g-emb evaluation context cache "
                "contains non-finite values"
            )

        self._evaluation_context_cache = (
            contexts
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> GaussianEFEmb:
        """Fit Gaussian EF-EMB to explicit training ratings."""
        _validate_rating_frame(
            train_data,
            name="train_data",
            allow_empty=False,
        )

        set_seed(
            self.seed,
            deterministic=True,
        )

        canonical_train = (
            train_data[
                [
                    "userId",
                    "movieId",
                    "rating",
                ]
            ]
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

        context_store = EFContextStore(
            canonical_train[
                [
                    "userId",
                    "movieId",
                ]
            ],
            heldout_interactions=(
                self.heldout_interactions
            ),
            max_context_items=None,
        )

        self._context_store = (
            context_store
        )

        n_items = (
            context_store
            .metadata
            .n_items
        )

        n_positive = (
            context_store
            .metadata
            .n_interactions
        )

        if n_positive != len(
            canonical_train
        ):
            raise RuntimeError(
                "g-emb canonical training "
                "interaction count mismatch"
            )

        canonical_user_ids = (
            context_store.user_ids[
                context_store
                .train_user_indices
            ]
        )

        canonical_item_ids = (
            context_store.item_catalog[
                context_store
                .train_item_indices
            ]
        )

        if not np.array_equal(
            canonical_user_ids,
            canonical_train[
                "userId"
            ].to_numpy(
                dtype=np.int64,
            ),
        ):
            raise RuntimeError(
                "g-emb rating rows are not aligned "
                "with canonical user indices"
            )

        if not np.array_equal(
            canonical_item_ids,
            canonical_train[
                "movieId"
            ].to_numpy(
                dtype=np.int64,
            ),
        ):
            raise RuntimeError(
                "g-emb rating rows are not aligned "
                "with canonical item indices"
            )

        ratings = (
            canonical_train[
                "rating"
            ]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

        if not np.isfinite(
            ratings
        ).all():
            raise RuntimeError(
                "g-emb canonical ratings "
                "contain non-finite values"
            )

        global_mean = float(
            np.mean(
                ratings,
                dtype=np.float64,
            )
        )

        if not np.isfinite(
            global_mean
        ):
            raise RuntimeError(
                "g-emb training global mean "
                "is non-finite"
            )

        self._global_mean = (
            global_mean
        )

        init_generator = (
            make_torch_generator(
                self.seed,
                "g-emb.parameter_initialization",
                device="cpu",
            )
        )

        module = _GEmbModule(
            n_items=n_items,
            n_factors=self.n_factors,
            init_generator=(
                init_generator
            ),
        )

        module = module.to(
            self.device
        )

        self._module = (
            module
        )

        incidence = (
            context_store
            .incidence_tensor(
                device=self.device,
                dtype=torch.float32,
            )
        )

        self._incidence = (
            incidence
        )

        optimizer = torch.optim.Adagrad(
            module.parameters(),
            lr=self.learning_rate,
        )

        self._history = []

        train_user_indices = (
            context_store
            .train_user_indices
        )

        train_item_indices = (
            context_store
            .train_item_indices
        )

        n_batches = int(
            math.ceil(
                n_positive
                /
                self.batch_size
            )
        )

        global_mean_tensor = torch.tensor(
            global_mean,
            dtype=torch.float32,
            device=self.device,
        )

        for epoch_index in range(
            self.epochs
        ):
            module.train()

            rng = make_numpy_rng(
                self.seed,
                (
                    "g-emb."
                    f"training_order.epoch_{epoch_index}"
                ),
            )

            order = rng.permutation(
                n_positive
            )

            batches = np.array_split(
                order,
                n_batches,
            )

            epoch_squared_error_sum = 0.0
            epoch_absolute_error_sum = 0.0
            epoch_examples = 0

            max_abs_prediction = 0.0
            max_context_norm = 0.0
            max_gradient_norm_before_clipping = 0.0

            completed_batches = 0

            for positions in batches:
                users_np = (
                    train_user_indices[
                        positions
                    ]
                )

                items_np = (
                    train_item_indices[
                        positions
                    ]
                )

                ratings_np = (
                    ratings[
                        positions
                    ]
                )

                users = torch.as_tensor(
                    users_np,
                    dtype=torch.long,
                    device=self.device,
                )

                items = torch.as_tensor(
                    items_np,
                    dtype=torch.long,
                    device=self.device,
                )

                targets = torch.as_tensor(
                    ratings_np,
                    dtype=torch.float32,
                    device=self.device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                user_sums = (
                    context_store
                    .aggregate_user_sums(
                        module
                        .context_embeddings
                        .weight,
                        incidence=incidence,
                    )
                )

                contexts = (
                    context_store
                    .context_vectors(
                        context_embeddings=(
                            module
                            .context_embeddings
                            .weight
                        ),
                        user_indices=(
                            users
                        ),
                        target_item_indices=(
                            items
                        ),
                        exclude_target=True,
                        normalization=(
                            self.context_normalization
                        ),
                        incidence=incidence,
                        precomputed_user_sums=(
                            user_sums
                        ),
                    )
                )

                predictions = (
                    module.predict(
                        items,
                        contexts,
                        global_mean_tensor,
                    )
                )

                if not torch.isfinite(
                    predictions
                ).all():
                    raise RuntimeError(
                        "g-emb produced "
                        "non-finite predictions"
                    )

                errors = (
                    predictions
                    -
                    targets
                )

                squared_error_sum = (
                    errors
                    .pow(
                        2
                    )
                    .sum()
                )

                # Scale each stochastic batch so that summing the
                # data gradients over one epoch corresponds to the
                # frozen full-training mean-squared-error objective.
                data_loss = (
                    float(
                        n_batches
                    )
                    /
                    float(
                        n_positive
                    )
                ) * (
                    squared_error_sum
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
                        "g-emb produced a non-finite loss"
                    )

                loss.backward()

                gradient_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        module.parameters(),
                        max_norm=(
                            self.max_grad_norm
                            if self.max_grad_norm
                            is not None
                            else float(
                                "inf"
                            )
                        ),
                    )
                )

                gradient_norm_value = float(
                    torch.as_tensor(
                        gradient_norm
                    )
                    .detach()
                    .cpu()
                    .item()
                )

                if not np.isfinite(
                    gradient_norm_value
                ):
                    raise RuntimeError(
                        "g-emb produced a "
                        "non-finite gradient norm"
                    )

                max_gradient_norm_before_clipping = max(
                    max_gradient_norm_before_clipping,
                    gradient_norm_value,
                )

                optimizer.step()

                for parameter in (
                    module.parameters()
                ):
                    if not torch.isfinite(
                        parameter
                    ).all():
                        raise RuntimeError(
                            "g-emb produced "
                            "non-finite parameters"
                        )

                with torch.no_grad():
                    epoch_squared_error_sum += float(
                        squared_error_sum
                        .detach()
                        .cpu()
                        .item()
                    )

                    epoch_absolute_error_sum += float(
                        errors
                        .abs()
                        .sum()
                        .detach()
                        .cpu()
                        .item()
                    )

                    epoch_examples += len(
                        positions
                    )

                    if len(
                        predictions
                    ) > 0:
                        max_abs_prediction = max(
                            max_abs_prediction,
                            float(
                                predictions
                                .abs()
                                .max()
                                .detach()
                                .cpu()
                                .item()
                            ),
                        )

                    if len(
                        contexts
                    ) > 0:
                        context_norms = (
                            torch.linalg.vector_norm(
                                contexts,
                                ord=2,
                                dim=1,
                            )
                        )

                        max_context_norm = max(
                            max_context_norm,
                            float(
                                context_norms
                                .max()
                                .detach()
                                .cpu()
                                .item()
                            ),
                        )

                completed_batches += 1

            if (
                epoch_examples
                !=
                n_positive
            ):
                raise RuntimeError(
                    "g-emb did not process every "
                    "training rating exactly once"
                )

            mse = (
                epoch_squared_error_sum
                /
                float(
                    n_positive
                )
            )

            rmse = math.sqrt(
                mse
            )

            mae = (
                epoch_absolute_error_sum
                /
                float(
                    n_positive
                )
            )

            with torch.no_grad():
                regularization_value = float(
                    self._regularization()
                    .detach()
                    .cpu()
                    .item()
                )

            total_loss = (
                mse
                +
                regularization_value
            )

            record: dict[
                str,
                float | int,
            ] = {
                "epoch": int(
                    epoch_index + 1
                ),
                "mse": float(
                    mse
                ),
                "rmse": float(
                    rmse
                ),
                "mae": float(
                    mae
                ),
                "regularization": float(
                    regularization_value
                ),
                "total_loss": float(
                    total_loss
                ),
                "examples": int(
                    epoch_examples
                ),
                "global_mean": float(
                    global_mean
                ),
                "max_abs_prediction": float(
                    max_abs_prediction
                ),
                "max_context_norm": float(
                    max_context_norm
                ),
                "max_gradient_norm_before_clipping": float(
                    max_gradient_norm_before_clipping
                ),
                "nonfinite_events": 0,
                "batches": int(
                    completed_batches
                ),
            }

            for key in [
                "mse",
                "rmse",
                "mae",
                "regularization",
                "total_loss",
                "global_mean",
                "max_abs_prediction",
                "max_context_norm",
                "max_gradient_norm_before_clipping",
            ]:
                if not np.isfinite(
                    float(
                        record[
                            key
                        ]
                    )
                ):
                    raise RuntimeError(
                        "g-emb training history contains "
                        f"non-finite {key}"
                    )

            self._history.append(
                record
            )

        module.eval()

        self._refresh_evaluation_context_cache()

        self._is_fitted = True

        return self

    def score(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return predicted ratings for ranking."""
        self._require_fitted()

        items = (
            _validate_candidate_items(
                candidate_items
            )
        )

        assert self._context_store is not None
        assert self._module is not None
        assert self._global_mean is not None
        assert (
            self._evaluation_context_cache
            is not None
        )

        user_index = (
            self._context_store
            .user_index(
                int(
                    user_id
                )
            )
        )

        item_indices = np.fromiter(
            (
                self._context_store
                .item_index(
                    int(item_id)
                )
                for item_id
                in items
            ),
            dtype=np.int64,
            count=len(
                items
            ),
        )

        item_tensor = torch.as_tensor(
            item_indices,
            dtype=torch.long,
            device=self.device,
        )

        context_vector = torch.as_tensor(
            self._evaluation_context_cache[
                user_index
            ],
            dtype=torch.float32,
            device=self.device,
        )

        contexts = (
            context_vector
            .unsqueeze(
                0
            )
            .expand(
                len(
                    item_indices
                ),
                -1,
            )
        )

        global_mean_tensor = torch.tensor(
            self._global_mean,
            dtype=torch.float32,
            device=self.device,
        )

        self._module.eval()

        with torch.no_grad():
            predictions = (
                self._module.predict(
                    item_tensor,
                    contexts,
                    global_mean_tensor,
                )
            )

        result = (
            predictions
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=False,
            )
        )

        if result.shape != (
            len(
                items
            ),
        ):
            raise RuntimeError(
                "g-emb returned invalid score shape"
            )

        if not np.isfinite(
            result
        ).all():
            raise RuntimeError(
                "g-emb returned non-finite predictions"
            )

        return result

    def predict_ratings(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """Predict explicit ratings using full training-history contexts."""
        self._require_fitted()

        _validate_rating_frame(
            data,
            name="data",
            allow_empty=True,
        )

        if len(
            data
        ) == 0:
            return np.empty(
                0,
                dtype=np.float64,
            )

        predictions = np.empty(
            len(
                data
            ),
            dtype=np.float64,
        )

        grouped_positions: dict[
            int,
            list[int],
        ] = {}

        user_array = (
            data[
                "userId"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        item_array = (
            data[
                "movieId"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        for position, user_id in enumerate(
            user_array
        ):
            grouped_positions.setdefault(
                int(
                    user_id
                ),
                [],
            ).append(
                position
            )

        for user_id, positions in (
            grouped_positions.items()
        ):
            candidate_items = (
                item_array[
                    np.asarray(
                        positions,
                        dtype=np.int64,
                    )
                ]
            )

            user_predictions = (
                self.score(
                    user_id,
                    candidate_items,
                )
            )

            predictions[
                np.asarray(
                    positions,
                    dtype=np.int64,
                )
            ] = user_predictions

        if not np.isfinite(
            predictions
        ).all():
            raise RuntimeError(
                "g-emb rating predictions "
                "contain non-finite values"
            )

        return predictions

    def native_metrics(
        self,
        data: pd.DataFrame,
    ) -> dict[
        str,
        float,
    ]:
        """Return RMSE and MAE on an explicit-rating frame."""
        _validate_rating_frame(
            data,
            name="data",
            allow_empty=False,
        )

        predictions = (
            self.predict_ratings(
                data
            )
        )

        actual = (
            data[
                "rating"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        errors = (
            predictions
            -
            actual
        )

        mse = float(
            np.mean(
                np.square(
                    errors
                ),
                dtype=np.float64,
            )
        )

        rmse = float(
            math.sqrt(
                mse
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    errors
                ),
                dtype=np.float64,
            )
        )

        if not (
            np.isfinite(
                rmse
            )
            and np.isfinite(
                mae
            )
        ):
            raise RuntimeError(
                "g-emb native metrics are non-finite"
            )

        return {
            "rmse": rmse,
            "mae": mae,
        }

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted g-emb state."""
        self._require_fitted()

        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "g-emb save path must end in .npz"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        assert self._context_store is not None
        assert self._module is not None
        assert self._global_mean is not None

        np.savez_compressed(
            path,
            user_ids=(
                self._context_store
                .user_ids
            ),
            item_ids=(
                self._context_store
                .item_catalog
            ),
            train_user_indices=(
                self._context_store
                .train_user_indices
            ),
            train_item_indices=(
                self._context_store
                .train_item_indices
            ),
            target_embeddings=(
                self._module
                .target_embeddings
                .weight
                .detach()
                .cpu()
                .numpy()
            ),
            context_embeddings=(
                self._module
                .context_embeddings
                .weight
                .detach()
                .cpu()
                .numpy()
            ),
            item_bias=(
                self._module
                .item_bias
                .weight
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
            "family": (
                "Gaussian"
            ),
            "architecture": (
                "EF-EMB"
            ),
            "target": (
                "explicit_rating"
            ),
            "prediction_formula": (
                "mu+b_i+t_i_dot_h_ui"
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
            "context_normalization": (
                self.context_normalization
            ),
            "target_l2": (
                self.target_l2
            ),
            "context_l2": (
                self.context_l2
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
            "global_mean": float(
                self._global_mean
            ),
            "uses_negative_sampling": (
                False
            ),
            "uses_user_embedding_table": (
                False
            ),
            "target_context_embeddings_separate": (
                True
            ),
            "training_target_excluded_from_context": (
                True
            ),
            "evaluation_context": (
                "full_training_history"
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
    ) -> GaussianEFEmb:
        """Load fitted g-emb for deterministic scoring."""
        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "g-emb load path must end in .npz"
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
                "Saved model is not g-emb"
            )

        if metadata.get(
            "spec_revision"
        ) != "v1.2":
            raise ValueError(
                "Saved g-emb does not use "
                "model-spec revision v1.2"
            )

        if metadata.get(
            "family"
        ) != "Gaussian":
            raise ValueError(
                "Saved g-emb has invalid family"
            )

        if metadata.get(
            "target"
        ) != "explicit_rating":
            raise ValueError(
                "Saved g-emb does not use "
                "the explicit-rating target"
            )

        if metadata.get(
            "uses_negative_sampling"
        ) is not False:
            raise ValueError(
                "Saved g-emb unexpectedly uses "
                "negative sampling"
            )

        if metadata.get(
            "uses_user_embedding_table"
        ) is not False:
            raise ValueError(
                "Saved g-emb unexpectedly contains "
                "a learned user embedding table"
            )

        if metadata.get(
            "target_context_embeddings_separate"
        ) is not True:
            raise ValueError(
                "Saved g-emb does not preserve "
                "separate target/context embeddings"
            )

        if metadata.get(
            "training_target_excluded_from_context"
        ) is not True:
            raise ValueError(
                "Saved g-emb does not confirm "
                "training target exclusion"
            )

        if metadata.get(
            "evaluation_context"
        ) != "full_training_history":
            raise ValueError(
                "Saved g-emb has invalid "
                "evaluation-context semantics"
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
            batch_size=int(
                metadata[
                    "batch_size"
                ]
            ),
            context_normalization=str(
                metadata[
                    "context_normalization"
                ]
            ),
            target_l2=float(
                metadata[
                    "target_l2"
                ]
            ),
            context_l2=float(
                metadata[
                    "context_l2"
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

            train_user_indices = np.asarray(
                state[
                    "train_user_indices"
                ],
                dtype=np.int64,
            )

            train_item_indices = np.asarray(
                state[
                    "train_item_indices"
                ],
                dtype=np.int64,
            )

            target_embeddings = np.asarray(
                state[
                    "target_embeddings"
                ],
                dtype=np.float32,
            )

            context_embeddings = np.asarray(
                state[
                    "context_embeddings"
                ],
                dtype=np.float32,
            )

            item_bias = np.asarray(
                state[
                    "item_bias"
                ],
                dtype=np.float32,
            )

        n_users = len(
            user_ids
        )

        n_items = len(
            item_ids
        )

        k = model.n_factors

        if target_embeddings.shape != (
            n_items,
            k,
        ):
            raise ValueError(
                "Saved g-emb target embeddings "
                "have invalid shape"
            )

        if context_embeddings.shape != (
            n_items,
            k,
        ):
            raise ValueError(
                "Saved g-emb context embeddings "
                "have invalid shape"
            )

        if item_bias.shape != (
            n_items,
            1,
        ):
            raise ValueError(
                "Saved g-emb item bias "
                "has invalid shape"
            )

        if train_user_indices.shape != (
            len(
                train_item_indices
            ),
        ):
            raise ValueError(
                "Saved g-emb training-index arrays "
                "have inconsistent shape"
            )

        if np.any(
            train_user_indices < 0
        ) or np.any(
            train_user_indices >= n_users
        ):
            raise ValueError(
                "Saved g-emb train user indices "
                "are out of range"
            )

        if np.any(
            train_item_indices < 0
        ) or np.any(
            train_item_indices >= n_items
        ):
            raise ValueError(
                "Saved g-emb train item indices "
                "are out of range"
            )

        arrays = [
            target_embeddings,
            context_embeddings,
            item_bias,
        ]

        if any(
            not np.isfinite(
                array
            ).all()
            for array in arrays
        ):
            raise ValueError(
                "Saved g-emb state contains "
                "non-finite values"
            )

        global_mean = float(
            metadata[
                "global_mean"
            ]
        )

        if not np.isfinite(
            global_mean
        ):
            raise ValueError(
                "Saved g-emb global mean "
                "is non-finite"
            )

        train_frame = pd.DataFrame(
            {
                "userId": (
                    user_ids[
                        train_user_indices
                    ]
                ),
                "movieId": (
                    item_ids[
                        train_item_indices
                    ]
                ),
            }
        )

        context_store = (
            EFContextStore(
                train_frame,
                heldout_interactions=None,
                max_context_items=None,
            )
        )

        if not np.array_equal(
            context_store.user_ids,
            user_ids,
        ):
            raise ValueError(
                "Saved g-emb user mapping "
                "could not be reconstructed"
            )

        if not np.array_equal(
            context_store.item_catalog,
            item_ids,
        ):
            raise ValueError(
                "Saved g-emb item mapping "
                "could not be reconstructed"
            )

        init_generator = (
            make_torch_generator(
                model.seed,
                "g-emb.load_initialization",
                device="cpu",
            )
        )

        module = _GEmbModule(
            n_items=n_items,
            n_factors=k,
            init_generator=(
                init_generator
            ),
        )

        with torch.no_grad():
            module.target_embeddings.weight.copy_(
                torch.from_numpy(
                    target_embeddings
                )
            )

            module.context_embeddings.weight.copy_(
                torch.from_numpy(
                    context_embeddings
                )
            )

            module.item_bias.weight.copy_(
                torch.from_numpy(
                    item_bias
                )
            )

        module.eval()

        model._context_store = (
            context_store
        )

        model._module = (
            module
        )

        model._incidence = (
            context_store
            .incidence_tensor(
                device="cpu",
                dtype=torch.float32,
            )
        )

        model._global_mean = (
            global_mean
        )

        model._history = [
            dict(row)
            for row
            in metadata.get(
                "training_history",
                [],
            )
        ]

        model._refresh_evaluation_context_cache()

        model._is_fitted = True

        return model