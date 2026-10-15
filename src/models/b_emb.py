"""Bernoulli exponential-family embedding model for Paper 2.

Model name
----------
b-emb

For user u and target item i,

    eta_ui = b_i + t_i^T h_ui

where:

- t_i is the target embedding;
- c_i is the context embedding;
- b_i is an item intercept;
- h_ui is the normalized train-history context.

Positive training target
------------------------

For (u, i) in canonical training data:

    y_ui = 1
    context = H_u \\ {i}

The target item is therefore never permitted to predict itself.

Negative training target
------------------------

For sampled legal negative j:

    (u, j) not in R_all
    y_uj = 0
    context = H_u

Negative examples are produced only through the shared
src.sampling.NegativeSampler.

The model never implements a second negative-sampling system.

Optimization
------------

The primary optimizer family is PyTorch Adagrad.

The objective is the frozen Paper 2 Bernoulli EF-EMB objective:

    mean positive BCE
    + negative_weight * mean negative BCE
    + L2(target embeddings)
    + L2(context embeddings)
    + L2(item intercepts)

Ranking
-------

The ranking score is the Bernoulli natural parameter eta_ui.

Ranking by eta_ui is equivalent to ranking by sigmoid(eta_ui).
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

from src.models.efemb import (
    EFContextStore,
    normalization_exponent,
)
from src.reproducibility import (
    make_numpy_rng,
    make_torch_generator,
    set_seed,
)
from src.sampling import (
    NegativeSampler,
)


REQUIRED_COLUMNS = {
    "userId",
    "movieId",
}


def _validate_interactions(
    data: pd.DataFrame,
    *,
    name: str,
    allow_empty: bool,
) -> None:
    """Validate an interaction DataFrame."""
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


class _BEmbModule(
    nn.Module,
):
    """Trainable parameters for Bernoulli EF-EMB."""

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

    def logits(
        self,
        item_indices: torch.Tensor,
        context_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Return b_i + t_i^T h_ui."""
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
            bias
            +
            interaction
        )


class BernoulliEFEmb:
    """Bernoulli exponential-family embedding recommender."""

    model_name = "b-emb"

    def __init__(
        self,
        *,
        negative_sampler: NegativeSampler | None = None,
        heldout_interactions: pd.DataFrame | None = None,
        n_factors: int = 32,
        learning_rate: float = 0.05,
        epochs: int = 5,
        batch_size: int = 2048,
        negative_ratio: int = 5,
        negative_weight: float = 1.0,
        negative_strategy: str = "uniform",
        negative_replace: bool = True,
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

        if (
            not np.isfinite(
                negative_weight
            )
            or negative_weight <= 0.0
        ):
            raise ValueError(
                "negative_weight must be finite and positive"
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
                "negative_replace must be boolean"
            )

        # Validate the frozen context-normalization mode.
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

        if (
            negative_sampler is not None
            and not isinstance(
                negative_sampler,
                NegativeSampler,
            )
        ):
            raise TypeError(
                "negative_sampler must be "
                "a shared NegativeSampler"
            )

        if heldout_interactions is not None:
            _validate_interactions(
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

        self.negative_strategy = (
            negative_strategy
        )

        self.negative_replace = (
            negative_replace
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

        self.seed = seed

        self.device = (
            torch_device
        )

        self._context_store: (
            EFContextStore | None
        ) = None

        self._module: (
            _BEmbModule | None
        ) = None

        self._incidence: (
            torch.Tensor | None
        ) = None

        self._evaluation_context_cache: (
            np.ndarray | None
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
                "b-emb must be fitted before scoring"
            )

    def _require_sampler(
        self,
    ) -> NegativeSampler:
        if self.negative_sampler is None:
            raise RuntimeError(
                "b-emb requires the shared "
                "NegativeSampler during fit()"
            )

        return self.negative_sampler

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

    def _validate_sampler(
        self,
    ) -> None:
        """Verify sampler compatibility with the canonical context store.

        The shared sampler must treat BOTH:

        - canonical training positives; and
        - held-out validation/test positives

        as observed interactions.

        This prevents any validation/test positive from ever being
        sampled as a Bernoulli training negative.
        """
        sampler = (
            self._require_sampler()
        )

        assert self._context_store is not None

        catalog = set(
            int(item_id)
            for item_id
            in self._context_store.item_catalog
        )

        heldout_by_user: dict[
            int,
            set[int],
        ] = {}

        if self.heldout_interactions is not None:
            for row in (
                self.heldout_interactions[
                    [
                        "userId",
                        "movieId",
                    ]
                ]
                .itertuples(
                    index=False
                )
            ):
                user_id = int(
                    row.userId
                )

                item_id = int(
                    row.movieId
                )

                heldout_by_user.setdefault(
                    user_id,
                    set(),
                ).add(
                    item_id
                )

        for user_id_raw in (
            self._context_store.user_ids
        ):
            user_id = int(
                user_id_raw
            )

            history = set(
                self._context_store
                .history_item_ids(
                    user_id
                )
            )

            heldout_items = (
                heldout_by_user.get(
                    user_id,
                    set(),
                )
            )

            required_observed = (
                history
                |
                heldout_items
            )

            observed = set(
                int(item_id)
                for item_id
                in sampler.observed_items(
                    user_id
                )
            )

            if not required_observed.issubset(
                observed
            ):
                missing = sorted(
                    required_observed
                    -
                    observed
                )

                raise ValueError(
                    "NegativeSampler does not mark "
                    "all canonical train/validation/test "
                    "positives as observed. "
                    f"User {user_id}, missing items: "
                    f"{missing[:10]}"
                )

            available = (
                sampler.available_items(
                    user_id
                )
            )

            if available.ndim != 1:
                raise ValueError(
                    "NegativeSampler available_items "
                    "must be one-dimensional"
                )

            available_set = set(
                int(item_id)
                for item_id
                in available
            )

            if not available_set.issubset(
                catalog
            ):
                raise ValueError(
                    "NegativeSampler contains items "
                    "outside the canonical training catalog"
                )

            if (
                available_set
                &
                observed
            ):
                raise ValueError(
                    "NegativeSampler exposes an observed "
                    "interaction as an available negative"
                )
    def _sample_epoch_negatives(
        self,
        *,
        epoch_index: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """Sample deterministic legal negatives for one epoch."""
        sampler = (
            self._require_sampler()
        )

        assert self._context_store is not None

        negative_user_indices: list[
            np.ndarray
        ] = []

        negative_item_indices: list[
            np.ndarray
        ] = []

        total_expected = (
            self._context_store
            .metadata
            .n_interactions
            *
            self.negative_ratio
        )

        for user_id_raw in (
            self._context_store.user_ids
        ):
            user_id = int(
                user_id_raw
            )

            user_index = (
                self._context_store
                .user_index(
                    user_id
                )
            )

            positive_count = len(
                self._context_store
                .history_indices(
                    user_id
                )
            )

            n_negative = (
                positive_count
                *
                self.negative_ratio
            )

            sample = sampler.sample(
                user_id=user_id,
                n=n_negative,
                strategy=self.negative_strategy,
                replace=self.negative_replace,
                draw_key=(
                    "b-emb",
                    int(
                        epoch_index
                    ),
                    user_id,
                ),
            )

            sampled_items = np.asarray(
                sample.items,
                dtype=np.int64,
            )

            if sampled_items.shape != (
                n_negative,
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
                    "b-emb received an observed pair "
                    "from the negative sampler"
                )

            mapped_items = np.fromiter(
                (
                    self._context_store.item_index(
                        int(item_id)
                    )
                    for item_id
                    in sampled_items
                ),
                dtype=np.int64,
                count=n_negative,
            )

            negative_user_indices.append(
                np.full(
                    n_negative,
                    user_index,
                    dtype=np.int64,
                )
            )

            negative_item_indices.append(
                mapped_items
            )

        users = np.concatenate(
            negative_user_indices
        )

        items = np.concatenate(
            negative_item_indices
        )

        if len(users) != total_expected:
            raise RuntimeError(
                "b-emb negative-user count "
                "does not match the configured ratio"
            )

        if len(items) != total_expected:
            raise RuntimeError(
                "b-emb negative-item count "
                "does not match the configured ratio"
            )

        return (
            users,
            items,
        )

    def _refresh_evaluation_context_cache(
        self,
    ) -> None:
        """Cache mean/raw/sqrt full-history contexts after fitting."""
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
                    incidence=self._incidence,
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
                "b-emb evaluation context cache "
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
    ) -> BernoulliEFEmb:
        """Fit Bernoulli EF-EMB."""
        _validate_interactions(
            train_data,
            name="train_data",
            allow_empty=False,
        )

        self._require_sampler()

        set_seed(
            self.seed,
            deterministic=True,
        )

        context_store = EFContextStore(
            train_data,
            heldout_interactions=(
                self.heldout_interactions
            ),
            max_context_items=None,
        )

        self._context_store = (
            context_store
        )

        self._validate_sampler()

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

        n_negative = (
            n_positive
            *
            self.negative_ratio
        )

        positive_user_indices = (
            context_store
            .train_user_indices
        )

        positive_item_indices = (
            context_store
            .train_item_indices
        )

        init_generator = (
            make_torch_generator(
                self.seed,
                "b-emb.parameter_initialization",
                device="cpu",
            )
        )

        module = _BEmbModule(
            n_items=n_items,
            n_factors=self.n_factors,
            init_generator=(
                init_generator
            ),
        )

        module = module.to(
            self.device
        )

        self._module = module

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

        n_batches = int(
            math.ceil(
                n_positive
                /
                self.batch_size
            )
        )

        for epoch_index in range(
            self.epochs
        ):
            module.train()

            (
                negative_user_indices,
                negative_item_indices,
            ) = self._sample_epoch_negatives(
                epoch_index=epoch_index
            )

            positive_rng = make_numpy_rng(
                self.seed,
                (
                    "b-emb."
                    f"positive_order.epoch_{epoch_index}"
                ),
            )

            negative_rng = make_numpy_rng(
                self.seed,
                (
                    "b-emb."
                    f"negative_order.epoch_{epoch_index}"
                ),
            )

            positive_order = (
                positive_rng.permutation(
                    n_positive
                )
            )

            negative_order = (
                negative_rng.permutation(
                    n_negative
                )
            )

            epoch_positive_loss_sum = 0.0
            epoch_negative_loss_sum = 0.0

            epoch_positive_examples = 0
            epoch_negative_examples = 0

            max_abs_logit = 0.0
            batches = 0

            for batch_number in range(
                n_batches
            ):
                positive_start = (
                    batch_number
                    *
                    self.batch_size
                )

                positive_end = min(
                    positive_start
                    +
                    self.batch_size,
                    n_positive,
                )

                negative_start = (
                    positive_start
                    *
                    self.negative_ratio
                )

                negative_end = (
                    positive_end
                    *
                    self.negative_ratio
                )

                positive_batch_positions = (
                    positive_order[
                        positive_start:
                        positive_end
                    ]
                )

                negative_batch_positions = (
                    negative_order[
                        negative_start:
                        negative_end
                    ]
                )

                pos_users_np = (
                    positive_user_indices[
                        positive_batch_positions
                    ]
                )

                pos_items_np = (
                    positive_item_indices[
                        positive_batch_positions
                    ]
                )

                neg_users_np = (
                    negative_user_indices[
                        negative_batch_positions
                    ]
                )

                neg_items_np = (
                    negative_item_indices[
                        negative_batch_positions
                    ]
                )

                pos_users = torch.as_tensor(
                    pos_users_np,
                    dtype=torch.long,
                    device=self.device,
                )

                pos_items = torch.as_tensor(
                    pos_items_np,
                    dtype=torch.long,
                    device=self.device,
                )

                neg_users = torch.as_tensor(
                    neg_users_np,
                    dtype=torch.long,
                    device=self.device,
                )

                neg_items = torch.as_tensor(
                    neg_items_np,
                    dtype=torch.long,
                    device=self.device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # Calculate the current exact full-history sums once
                # for this optimizer step and reuse them for both
                # positive and negative contexts.
                user_sums = (
                    context_store
                    .aggregate_user_sums(
                        module
                        .context_embeddings
                        .weight,
                        incidence=incidence,
                    )
                )

                positive_contexts = (
                    context_store
                    .context_vectors(
                        context_embeddings=(
                            module
                            .context_embeddings
                            .weight
                        ),
                        user_indices=(
                            pos_users
                        ),
                        target_item_indices=(
                            pos_items
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

                negative_contexts = (
                    context_store
                    .context_vectors(
                        context_embeddings=(
                            module
                            .context_embeddings
                            .weight
                        ),
                        user_indices=(
                            neg_users
                        ),
                        target_item_indices=None,
                        exclude_target=False,
                        normalization=(
                            self.context_normalization
                        ),
                        incidence=incidence,
                        precomputed_user_sums=(
                            user_sums
                        ),
                    )
                )

                positive_logits = (
                    module.logits(
                        pos_items,
                        positive_contexts,
                    )
                )

                negative_logits = (
                    module.logits(
                        neg_items,
                        negative_contexts,
                    )
                )

                if not torch.isfinite(
                    positive_logits
                ).all():
                    raise RuntimeError(
                        "b-emb produced non-finite "
                        "positive logits"
                    )

                if not torch.isfinite(
                    negative_logits
                ).all():
                    raise RuntimeError(
                        "b-emb produced non-finite "
                        "negative logits"
                    )

                positive_loss_sum = (
                    F.binary_cross_entropy_with_logits(
                        positive_logits,
                        torch.ones_like(
                            positive_logits
                        ),
                        reduction="sum",
                    )
                )

                negative_loss_sum = (
                    F.binary_cross_entropy_with_logits(
                        negative_logits,
                        torch.zeros_like(
                            negative_logits
                        ),
                        reduction="sum",
                    )
                )

                # Batch scaling is chosen so each partition is an
                # estimator of its frozen whole-dataset mean objective.
                positive_loss = (
                    positive_loss_sum
                    *
                    (
                        float(
                            n_batches
                        )
                        /
                        float(
                            n_positive
                        )
                    )
                )

                negative_loss = (
                    negative_loss_sum
                    *
                    (
                        float(
                            n_batches
                        )
                        /
                        float(
                            n_negative
                        )
                    )
                )

                regularization = (
                    self._regularization()
                )

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
                        "b-emb produced a non-finite loss"
                    )

                loss.backward()

                if (
                    self.max_grad_norm
                    is not None
                ):
                    gradient_norm = (
                        torch.nn.utils
                        .clip_grad_norm_(
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
                            "b-emb produced a "
                            "non-finite gradient norm"
                        )

                optimizer.step()

                for parameter in (
                    module.parameters()
                ):
                    if not torch.isfinite(
                        parameter
                    ).all():
                        raise RuntimeError(
                            "b-emb produced "
                            "non-finite parameters"
                        )

                epoch_positive_loss_sum += float(
                    positive_loss_sum
                    .detach()
                    .cpu()
                    .item()
                )

                epoch_negative_loss_sum += float(
                    negative_loss_sum
                    .detach()
                    .cpu()
                    .item()
                )

                epoch_positive_examples += len(
                    pos_users_np
                )

                epoch_negative_examples += len(
                    neg_users_np
                )

                batch_max = max(
                    float(
                        positive_logits
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                        .item()
                    ),
                    float(
                        negative_logits
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                        .item()
                    ),
                )

                max_abs_logit = max(
                    max_abs_logit,
                    batch_max,
                )

                batches += 1

            if (
                epoch_positive_examples
                !=
                n_positive
            ):
                raise RuntimeError(
                    "b-emb did not process every "
                    "positive exactly once"
                )

            if (
                epoch_negative_examples
                !=
                n_negative
            ):
                raise RuntimeError(
                    "b-emb did not process the exact "
                    "configured negative count"
                )

            positive_loss_value = (
                epoch_positive_loss_sum
                /
                float(
                    n_positive
                )
            )

            negative_loss_value = (
                epoch_negative_loss_sum
                /
                float(
                    n_negative
                )
            )

            with torch.no_grad():
                regularization_value = float(
                    self._regularization()
                    .detach()
                    .cpu()
                    .item()
                )

            data_loss_value = (
                positive_loss_value
                +
                self.negative_weight
                *
                negative_loss_value
            )

            total_loss_value = (
                data_loss_value
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
                "positive_loss": float(
                    positive_loss_value
                ),
                "negative_loss": float(
                    negative_loss_value
                ),
                "data_loss": float(
                    data_loss_value
                ),
                "regularization": float(
                    regularization_value
                ),
                "total_loss": float(
                    total_loss_value
                ),
                "positive_examples": int(
                    epoch_positive_examples
                ),
                "negative_examples": int(
                    epoch_negative_examples
                ),
                "negative_ratio": int(
                    self.negative_ratio
                ),
                "max_abs_logit": float(
                    max_abs_logit
                ),
                "batches": int(
                    batches
                ),
            }

            for key in [
                "positive_loss",
                "negative_loss",
                "data_loss",
                "regularization",
                "total_loss",
                "max_abs_logit",
            ]:
                if not np.isfinite(
                    float(
                        record[
                            key
                        ]
                    )
                ):
                    raise RuntimeError(
                        "b-emb training history contains "
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
        """Return Bernoulli natural parameters for candidates."""
        self._require_fitted()

        items = (
            _validate_candidate_items(
                candidate_items
            )
        )

        assert self._context_store is not None
        assert self._module is not None
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

        self._module.eval()

        with torch.no_grad():
            logits = (
                self._module.logits(
                    item_tensor,
                    contexts,
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
            len(
                items
            ),
        ):
            raise RuntimeError(
                "b-emb returned invalid score shape"
            )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                "b-emb returned non-finite scores"
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
                "b-emb produced non-finite probabilities"
            )

        return probabilities.astype(
            np.float64,
            copy=False,
        )

    def native_metrics(
        self,
        data: pd.DataFrame,
    ) -> dict[str, float]:
        """Return exact masked predictive Bernoulli NLL diagnostics.

        The evaluation frame supplies held-out positive cells.

        Genuine zero cells are reconstructed from the leakage-safe
        NegativeSampler. Training positives and all held-out positives
        remain excluded from the zero universe.
        """
        self._require_fitted()

        _validate_interactions(
            data,
            name="data",
            allow_empty=False,
        )

        sampler = self._require_sampler()

        assert self._context_store is not None

        if self.heldout_interactions is None:
            raise RuntimeError(
                "b-emb native metrics require heldout_interactions"
            )

        evaluation_pairs = set(
            zip(
                data["userId"].astype(int),
                data["movieId"].astype(int),
            )
        )

        heldout_pairs = set(
            zip(
                self.heldout_interactions["userId"].astype(int),
                self.heldout_interactions["movieId"].astype(int),
            )
        )

        if not evaluation_pairs.issubset(
            heldout_pairs
        ):
            raise ValueError(
                "b-emb native_metrics data must be a subset "
                "of heldout_interactions"
            )

        positives_by_user: dict[
            int,
            np.ndarray,
        ] = {}

        for user_id, group in data.groupby(
            "userId",
            sort=True,
        ):
            positives_by_user[
                int(user_id)
            ] = (
                group["movieId"]
                .to_numpy(
                    dtype=np.int64,
                )
            )

        positive_nll_sum = 0.0
        zero_nll_sum = 0.0

        positive_cells = 0
        zero_cells = 0

        for user_id_raw in self.user_ids:

            user_id = int(
                user_id_raw
            )

            positives = positives_by_user.get(
                user_id,
                np.empty(
                    0,
                    dtype=np.int64,
                ),
            )

            zeros = np.asarray(
                sampler.available_items(
                    user_id
                ),
                dtype=np.int64,
            )

            if len(positives) > 0:

                positive_logits = self.score(
                    user_id,
                    positives,
                )

                positive_nll = np.logaddexp(
                    0.0,
                    -positive_logits,
                )

                positive_nll_sum += float(
                    positive_nll.sum(
                        dtype=np.float64
                    )
                )

                positive_cells += len(
                    positives
                )

            if len(zeros) > 0:

                zero_logits = self.score(
                    user_id,
                    zeros,
                )

                zero_nll = np.logaddexp(
                    0.0,
                    zero_logits,
                )

                zero_nll_sum += float(
                    zero_nll.sum(
                        dtype=np.float64
                    )
                )

                zero_cells += len(
                    zeros
                )

        if positive_cells != len(data):
            raise RuntimeError(
                "b-emb native metric positive-cell accounting mismatch"
            )

        if zero_cells <= 0:
            raise RuntimeError(
                "b-emb native metric zero-cell universe is empty"
            )

        evaluation_cells = (
            positive_cells
            +
            zero_cells
        )

        positive_mean_nll = (
            positive_nll_sum
            /
            float(
                positive_cells
            )
        )

        zero_mean_nll = (
            zero_nll_sum
            /
            float(
                zero_cells
            )
        )

        masked_nll = (
            positive_nll_sum
            +
            zero_nll_sum
        ) / float(
            evaluation_cells
        )

        metrics = {
            "bernoulli_positive_mean_nll": (
                float(
                    positive_mean_nll
                )
            ),
            "bernoulli_zero_mean_nll": (
                float(
                    zero_mean_nll
                )
            ),
            "masked_predictive_bernoulli_nll": (
                float(
                    masked_nll
                )
            ),
        }

        if not all(
            np.isfinite(
                value
            )
            and value >= 0.0
            for value
            in metrics.values()
        ):
            raise RuntimeError(
                "b-emb native metrics are non-finite or negative"
            )

        return metrics    

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted b-emb state."""
        self._require_fitted()

        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "b-emb save path must end in .npz"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        assert self._context_store is not None
        assert self._module is not None

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
                "Bernoulli"
            ),
            "architecture": (
                "EF-EMB"
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
            "negative_strategy": (
                self.negative_strategy
            ),
            "negative_replace": (
                self.negative_replace
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
            "uses_shared_negative_sampler": (
                True
            ),
            "uses_user_embedding_table": (
                False
            ),
            "target_context_embeddings_separate": (
                True
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
    ) -> BernoulliEFEmb:
        """Load fitted b-emb for deterministic scoring."""
        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "b-emb load path must end in .npz"
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
                "Saved model is not b-emb"
            )

        if metadata.get(
            "spec_revision"
        ) != "v1.2":
            raise ValueError(
                "Saved b-emb does not use "
                "model-spec revision v1.2"
            )

        if metadata.get(
            "uses_shared_negative_sampler"
        ) is not True:
            raise ValueError(
                "Saved b-emb metadata does not "
                "confirm shared sampler usage"
            )

        if metadata.get(
            "uses_user_embedding_table"
        ) is not False:
            raise ValueError(
                "Saved b-emb unexpectedly contains "
                "a learned user embedding table"
            )

        if metadata.get(
            "target_context_embeddings_separate"
        ) is not True:
            raise ValueError(
                "Saved b-emb does not preserve "
                "separate target/context embeddings"
            )

        model = cls(
            negative_sampler=None,
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
                "Saved b-emb target embeddings "
                "have invalid shape"
            )

        if context_embeddings.shape != (
            n_items,
            k,
        ):
            raise ValueError(
                "Saved b-emb context embeddings "
                "have invalid shape"
            )

        if item_bias.shape != (
            n_items,
            1,
        ):
            raise ValueError(
                "Saved b-emb item bias "
                "has invalid shape"
            )

        if train_user_indices.shape != (
            len(
                train_item_indices
            ),
        ):
            raise ValueError(
                "Saved b-emb training-index arrays "
                "have inconsistent shape"
            )

        if np.any(
            train_user_indices < 0
        ) or np.any(
            train_user_indices >= n_users
        ):
            raise ValueError(
                "Saved b-emb train user indices "
                "are out of range"
            )

        if np.any(
            train_item_indices < 0
        ) or np.any(
            train_item_indices >= n_items
        ):
            raise ValueError(
                "Saved b-emb train item indices "
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
                "Saved b-emb state contains "
                "non-finite values"
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
                "Saved b-emb user mapping "
                "could not be reconstructed"
            )

        if not np.array_equal(
            context_store.item_catalog,
            item_ids,
        ):
            raise ValueError(
                "Saved b-emb item mapping "
                "could not be reconstructed"
            )

        init_generator = (
            make_torch_generator(
                model.seed,
                "b-emb.load_initialization",
                device="cpu",
            )
        )

        module = _BEmbModule(
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