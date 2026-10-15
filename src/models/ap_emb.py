"""Additive Poisson exponential-family embedding model for Paper 2.

Model name
----------
ap-emb

This model is mathematically distinct from p-emb.

Raw parameters
--------------

Each item i has unconstrained raw target/context parameters:

    raw_t_i in R^K
    raw_c_i in R^K

The actual positive embeddings are:

    t_i^+ = exp(raw_t_i)
    c_i^+ = exp(raw_c_i)

The positive item baseline is:

    delta_i = exp(b_i)

Additive Poisson rate
---------------------

For user u and target item i:

    lambda_ui
        = delta_i
        + (t_i^+)^T h_ui^+

where h_ui^+ is constructed from positive context embeddings.

Positive training target
------------------------

For (u, i) in canonical training data:

    y_ui = 1
    context = H_u \\ {i}

The target item is excluded from its own context.

Legal zero target
-----------------

For a genuinely unobserved cell (u, j):

    y_uj = 0
    context = H_u

Validation/test positives are masked and never treated as zero cells.

Poisson objective
-----------------

Ignoring the factorial constant:

    NLL_ui
        = lambda_ui
        - y_ui * log(lambda_ui)

The conceptual primary objective is the masked full Poisson likelihood
over every legal training cell.

Legal zeros may be subsampled only with the leakage-safe shared sampler
and the unbiased uniform-with-replacement correction:

    |Z_u| / m_u

Regularization
--------------

Regularization acts in raw/log parameter space:

    lambda_t * ||raw_t||^2
    + lambda_c * ||raw_c||^2
    + lambda_b * ||b||^2

Optimization
------------

The primary optimizer family is PyTorch Adagrad.

Ranking
-------

The common ranking score is:

    score_ui = log(lambda_ui)

Because log is strictly increasing, this produces the same ranking as
the positive additive Poisson rate itself.
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
    """Validate an interaction frame."""
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


class _APEmbModule(
    nn.Module,
):
    """Trainable raw parameters for additive Poisson EF-EMB."""

    def __init__(
        self,
        *,
        n_items: int,
        n_factors: int,
        initial_item_biases: np.ndarray,
        initial_interaction_rate: float,
        init_generator: torch.Generator,
    ) -> None:
        super().__init__()

        biases = np.asarray(
            initial_item_biases,
            dtype=np.float32,
        )

        if biases.shape != (
            n_items,
        ):
            raise ValueError(
                "initial_item_biases has invalid shape"
            )

        if not np.isfinite(
            biases
        ).all():
            raise ValueError(
                "initial_item_biases contains "
                "non-finite values"
            )

        if (
            not np.isfinite(
                initial_interaction_rate
            )
            or initial_interaction_rate <= 0.0
        ):
            raise ValueError(
                "initial_interaction_rate must "
                "be finite and positive"
            )

        self.raw_target_embeddings = nn.Embedding(
            n_items,
            n_factors,
        )

        self.raw_context_embeddings = nn.Embedding(
            n_items,
            n_factors,
        )

        self.item_bias = nn.Embedding(
            n_items,
            1,
        )

        # Choose a K-aware positive-embedding scale so that
        #
        #     K * exp(mu)^2 ~= initial_interaction_rate
        #
        # at initialization.
        raw_location = 0.5 * (
            math.log(
                initial_interaction_rate
            )
            -
            math.log(
                float(
                    n_factors
                )
            )
        )

        nn.init.normal_(
            self.raw_target_embeddings.weight,
            mean=raw_location,
            std=0.01,
            generator=init_generator,
        )

        nn.init.normal_(
            self.raw_context_embeddings.weight,
            mean=raw_location,
            std=0.01,
            generator=init_generator,
        )

        with torch.no_grad():
            self.item_bias.weight.copy_(
                torch.from_numpy(
                    biases
                ).unsqueeze(
                    1
                )
            )

    def positive_target_embeddings(
        self,
    ) -> torch.Tensor:
        """Return exp(raw target parameters)."""
        return torch.exp(
            self.raw_target_embeddings.weight
        )

    def positive_context_embeddings(
        self,
    ) -> torch.Tensor:
        """Return exp(raw context parameters)."""
        return torch.exp(
            self.raw_context_embeddings.weight
        )

    def additive_rate(
        self,
        item_indices: torch.Tensor,
        context_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Return exp(b_i) + (t_i^+)^T h_ui^+."""
        raw_targets = (
            self.raw_target_embeddings(
                item_indices
            )
        )

        positive_targets = torch.exp(
            raw_targets
        )

        baseline = torch.exp(
            self.item_bias(
                item_indices
            )
            .squeeze(
                1
            )
        )

        interaction = (
            positive_targets
            *
            context_vectors
        ).sum(
            dim=1
        )

        return (
            baseline
            +
            interaction
        )


class AdditivePoissonEFEmb:
    """Additive positive Poisson exponential-family embedding."""

    model_name = "ap-emb"

    def __init__(
        self,
        *,
        zero_sampler: NegativeSampler | None = None,
        heldout_interactions: pd.DataFrame | None = None,
        n_factors: int = 32,
        learning_rate: float = 0.05,
        epochs: int = 5,
        batch_size: int = 2048,
        zero_ratio: int = 5,
        zero_sampling_strategy: str = "uniform",
        zero_replace: bool = True,
        context_normalization: str = "mean",
        target_l2: float = 1e-6,
        context_l2: float = 1e-6,
        bias_l2: float = 1e-6,
        initial_interaction_rate: float = 0.01,
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

        if zero_sampling_strategy != "uniform":
            raise ValueError(
                "Primary ap-emb supports only uniform "
                "zero sampling because nonuniform sampling "
                "requires inverse-probability weighting"
            )

        if zero_replace is not True:
            raise ValueError(
                "Primary ap-emb requires zero_replace=True "
                "for the frozen |Z_u|/m_u estimator"
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
            not np.isfinite(
                initial_interaction_rate
            )
            or initial_interaction_rate <= 0.0
        ):
            raise ValueError(
                "initial_interaction_rate must "
                "be finite and positive"
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
            zero_sampler is not None
            and not isinstance(
                zero_sampler,
                NegativeSampler,
            )
        ):
            raise TypeError(
                "zero_sampler must be the shared "
                "NegativeSampler"
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

        self.zero_sampler = (
            zero_sampler
        )

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

        self.zero_ratio = (
            zero_ratio
        )

        self.zero_sampling_strategy = (
            zero_sampling_strategy
        )

        self.zero_replace = (
            zero_replace
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

        self.initial_interaction_rate = float(
            initial_interaction_rate
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
            _APEmbModule | None
        ) = None

        self._incidence: (
            torch.Tensor | None
        ) = None

        self._evaluation_context_cache: (
            np.ndarray | None
        ) = None

        self._zero_population_sizes: (
            np.ndarray | None
        ) = None

        self._zero_population_cells: (
            int | None
        ) = None

        self._legal_cells: (
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

    @property
    def zero_population_cells(
        self,
    ) -> int:
        self._require_fitted()

        assert (
            self._zero_population_cells
            is not None
        )

        return int(
            self._zero_population_cells
        )

    @property
    def legal_cells(
        self,
    ) -> int:
        self._require_fitted()

        assert (
            self._legal_cells
            is not None
        )

        return int(
            self._legal_cells
        )

    def _require_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "ap-emb must be fitted before scoring"
            )

    def _require_sampler(
        self,
    ) -> NegativeSampler:
        if self.zero_sampler is None:
            raise RuntimeError(
                "ap-emb requires the shared "
                "NegativeSampler during fit()"
            )

        return self.zero_sampler

    @staticmethod
    def _validate_positive_rate(
        rate: torch.Tensor,
        *,
        name: str,
    ) -> None:
        if not torch.isfinite(
            rate
        ).all():
            raise RuntimeError(
                f"ap-emb produced non-finite {name}"
            )

        if torch.any(
            rate <= 0.0
        ):
            raise RuntimeError(
                f"ap-emb produced non-positive {name}"
            )

    def _regularization(
        self,
    ) -> torch.Tensor:
        """L2 regularization in frozen raw/log parameter space."""
        assert self._module is not None

        target_penalty = (
            self._module
            .raw_target_embeddings
            .weight
            .pow(
                2
            )
            .sum()
        )

        context_penalty = (
            self._module
            .raw_context_embeddings
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

    def _heldout_items_by_user(
        self,
    ) -> dict[
        int,
        set[int],
    ]:
        result: dict[
            int,
            set[int],
        ] = {}

        if self.heldout_interactions is None:
            return result

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
            result.setdefault(
                int(
                    row.userId
                ),
                set(),
            ).add(
                int(
                    row.movieId
                )
            )

        return result

    def _validate_sampler_and_zero_population(
        self,
    ) -> np.ndarray:
        """Verify exact legal-zero support and return |Z_u|."""
        sampler = (
            self._require_sampler()
        )

        assert self._context_store is not None

        catalog = set(
            int(item_id)
            for item_id
            in self._context_store.item_catalog
        )

        heldout_by_user = (
            self._heldout_items_by_user()
        )

        zero_counts = np.zeros(
            self._context_store
            .metadata
            .n_users,
            dtype=np.int64,
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

            train_history = set(
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
                train_history
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

            if observed != required_observed:
                missing = sorted(
                    required_observed
                    -
                    observed
                )

                extra = sorted(
                    observed
                    -
                    required_observed
                )

                raise ValueError(
                    "NegativeSampler observed set does not "
                    "exactly match canonical train+heldout "
                    "observations. "
                    f"User {user_id}; "
                    f"missing={missing[:10]}; "
                    f"extra={extra[:10]}"
                )

            available = np.asarray(
                sampler.available_items(
                    user_id
                ),
                dtype=np.int64,
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

            expected_zero_set = (
                catalog
                -
                required_observed
            )

            if available_set != expected_zero_set:
                raise ValueError(
                    "NegativeSampler available set does not "
                    "equal the exact legal ap-emb zero set"
                )

            if (
                available_set
                &
                required_observed
            ):
                raise ValueError(
                    "NegativeSampler exposes an observed "
                    "interaction as an ap-emb zero"
                )

            zero_counts[
                user_index
            ] = len(
                available
            )

        return zero_counts

    def _initial_item_biases(
        self,
    ) -> np.ndarray:
        """Training-only log baseline-rate initialization."""
        assert self._context_store is not None

        n_items = (
            self._context_store
            .metadata
            .n_items
        )

        n_users = (
            self._context_store
            .metadata
            .n_users
        )

        counts = np.bincount(
            self._context_store
            .train_item_indices,
            minlength=n_items,
        ).astype(
            np.float64,
            copy=False,
        )

        rates = (
            counts
            /
            float(
                n_users
            )
        )

        rates = np.clip(
            rates,
            1e-6,
            None,
        )

        biases = np.log(
            rates
        )

        if not np.isfinite(
            biases
        ).all():
            raise RuntimeError(
                "ap-emb train-only item-bias "
                "initialization is non-finite"
            )

        return biases.astype(
            np.float32,
            copy=False,
        )

    def _sample_epoch_zeros(
        self,
        *,
        epoch_index: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Sample legal zeros with unbiased |Z_u|/m_u weights."""
        sampler = (
            self._require_sampler()
        )

        assert self._context_store is not None
        assert (
            self._zero_population_sizes
            is not None
        )

        sampled_user_indices: list[
            np.ndarray
        ] = []

        sampled_item_indices: list[
            np.ndarray
        ] = []

        sampled_weights: list[
            np.ndarray
        ] = []

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

            zero_population = int(
                self._zero_population_sizes[
                    user_index
                ]
            )

            if zero_population == 0:
                continue

            positive_count = len(
                self._context_store
                .history_indices(
                    user_id
                )
            )

            m_u = (
                positive_count
                *
                self.zero_ratio
            )

            if m_u <= 0:
                raise RuntimeError(
                    "ap-emb zero sample count "
                    "must be positive"
                )

            sample = sampler.sample(
                user_id=user_id,
                n=m_u,
                strategy="uniform",
                replace=True,
                draw_key=(
                    "ap-emb",
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
                m_u,
            ):
                raise RuntimeError(
                    "NegativeSampler returned an "
                    "unexpected ap-emb zero count"
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
                    "ap-emb received an observed "
                    "interaction as a sampled zero"
                )

            mapped_items = np.fromiter(
                (
                    self._context_store
                    .item_index(
                        int(item_id)
                    )
                    for item_id
                    in sampled_items
                ),
                dtype=np.int64,
                count=m_u,
            )

            weight = (
                float(
                    zero_population
                )
                /
                float(
                    m_u
                )
            )

            sampled_user_indices.append(
                np.full(
                    m_u,
                    user_index,
                    dtype=np.int64,
                )
            )

            sampled_item_indices.append(
                mapped_items
            )

            sampled_weights.append(
                np.full(
                    m_u,
                    weight,
                    dtype=np.float64,
                )
            )

        if not sampled_user_indices:
            return (
                np.empty(
                    0,
                    dtype=np.int64,
                ),
                np.empty(
                    0,
                    dtype=np.int64,
                ),
                np.empty(
                    0,
                    dtype=np.float64,
                ),
            )

        users = np.concatenate(
            sampled_user_indices
        )

        items = np.concatenate(
            sampled_item_indices
        )

        weights = np.concatenate(
            sampled_weights
        )

        if not (
            len(users)
            ==
            len(items)
            ==
            len(weights)
        ):
            raise RuntimeError(
                "ap-emb sampled-zero arrays "
                "have inconsistent lengths"
            )

        if not np.isfinite(
            weights
        ).all():
            raise RuntimeError(
                "ap-emb zero weights contain "
                "non-finite values"
            )

        if np.any(
            weights <= 0.0
        ):
            raise RuntimeError(
                "ap-emb zero weights must be positive"
            )

        expected_weight_sum = float(
            self._zero_population_sizes.sum()
        )

        actual_weight_sum = float(
            weights.sum(
                dtype=np.float64
            )
        )

        if not np.isclose(
            actual_weight_sum,
            expected_weight_sum,
            rtol=1e-12,
            atol=1e-6,
        ):
            raise RuntimeError(
                "ap-emb |Z_u|/m_u correction "
                "does not recover total zero population"
            )

        return (
            users,
            items,
            weights,
        )

    def _positive_context_weight(
        self,
    ) -> torch.Tensor:
        """Return strictly positive context embeddings."""
        assert self._module is not None

        weight = (
            self._module
            .positive_context_embeddings()
        )

        if not torch.isfinite(
            weight
        ).all():
            raise RuntimeError(
                "ap-emb positive context embeddings "
                "contain non-finite values"
            )

        if torch.any(
            weight <= 0.0
        ):
            raise RuntimeError(
                "ap-emb positive context embeddings "
                "must be strictly positive"
            )

        return weight

    def _refresh_evaluation_context_cache(
        self,
    ) -> None:
        """Cache full-history positive context vectors for scoring."""
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
            positive_context_weight = (
                self._positive_context_weight()
            )

            user_sums = (
                self._context_store
                .aggregate_user_sums(
                    positive_context_weight,
                    incidence=(
                        self._incidence
                    ),
                )
            )

            contexts = (
                self._context_store
                .context_vectors(
                    context_embeddings=(
                        positive_context_weight
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
                "ap-emb evaluation context cache "
                "contains non-finite values"
            )

        if torch.any(
            contexts < 0.0
        ):
            raise RuntimeError(
                "ap-emb evaluation contexts "
                "must be nonnegative"
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
    ) -> AdditivePoissonEFEmb:
        """Fit additive Poisson EF-EMB."""
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

        zero_population_sizes = (
            self._validate_sampler_and_zero_population()
        )

        self._zero_population_sizes = (
            zero_population_sizes
        )

        n_users = (
            context_store
            .metadata
            .n_users
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

        zero_population_cells = int(
            zero_population_sizes.sum()
        )

        legal_cells = (
            n_positive
            +
            zero_population_cells
        )

        total_matrix_cells = (
            n_users
            *
            n_items
        )

        heldout_count = (
            0
            if self.heldout_interactions is None
            else len(
                self.heldout_interactions
            )
        )

        expected_legal_cells = (
            total_matrix_cells
            -
            heldout_count
        )

        if legal_cells != expected_legal_cells:
            raise RuntimeError(
                "ap-emb legal-cell accounting mismatch"
            )

        self._zero_population_cells = (
            zero_population_cells
        )

        self._legal_cells = (
            legal_cells
        )

        positive_user_indices = (
            context_store
            .train_user_indices
        )

        positive_item_indices = (
            context_store
            .train_item_indices
        )

        initial_biases = (
            self._initial_item_biases()
        )

        init_generator = (
            make_torch_generator(
                self.seed,
                "ap-emb.parameter_initialization",
                device="cpu",
            )
        )

        module = _APEmbModule(
            n_items=n_items,
            n_factors=self.n_factors,
            initial_item_biases=(
                initial_biases
            ),
            initial_interaction_rate=(
                self.initial_interaction_rate
            ),
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
                zero_user_indices,
                zero_item_indices,
                zero_weights,
            ) = self._sample_epoch_zeros(
                epoch_index=epoch_index
            )

            n_zero_samples = len(
                zero_user_indices
            )

            positive_rng = make_numpy_rng(
                self.seed,
                (
                    "ap-emb."
                    f"positive_order.epoch_{epoch_index}"
                ),
            )

            zero_rng = make_numpy_rng(
                self.seed,
                (
                    "ap-emb."
                    f"zero_order.epoch_{epoch_index}"
                ),
            )

            positive_order = (
                positive_rng.permutation(
                    n_positive
                )
            )

            zero_order = (
                zero_rng.permutation(
                    n_zero_samples
                )
            )

            positive_batches = np.array_split(
                positive_order,
                n_batches,
            )

            zero_batches = np.array_split(
                zero_order,
                n_batches,
            )

            epoch_positive_nll_sum = 0.0
            epoch_zero_estimate_sum = 0.0

            epoch_positive_examples = 0
            epoch_zero_samples = 0

            max_rate = 0.0
            max_abs_log_rate = 0.0

            batches = 0

            for batch_number in range(
                n_batches
            ):
                positive_positions = (
                    positive_batches[
                        batch_number
                    ]
                )

                zero_positions = (
                    zero_batches[
                        batch_number
                    ]
                )

                pos_users_np = (
                    positive_user_indices[
                        positive_positions
                    ]
                )

                pos_items_np = (
                    positive_item_indices[
                        positive_positions
                    ]
                )

                zero_users_np = (
                    zero_user_indices[
                        zero_positions
                    ]
                )

                zero_items_np = (
                    zero_item_indices[
                        zero_positions
                    ]
                )

                zero_weights_np = (
                    zero_weights[
                        zero_positions
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

                zero_users = torch.as_tensor(
                    zero_users_np,
                    dtype=torch.long,
                    device=self.device,
                )

                zero_items = torch.as_tensor(
                    zero_items_np,
                    dtype=torch.long,
                    device=self.device,
                )

                zero_weight_tensor = torch.as_tensor(
                    zero_weights_np,
                    dtype=torch.float32,
                    device=self.device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                positive_context_weight = (
                    self._positive_context_weight()
                )

                positive_target_weight = (
                    module
                    .positive_target_embeddings()
                )

                if not torch.isfinite(
                    positive_target_weight
                ).all():
                    raise RuntimeError(
                        "ap-emb positive target embeddings "
                        "contain non-finite values"
                    )

                if torch.any(
                    positive_target_weight <= 0.0
                ):
                    raise RuntimeError(
                        "ap-emb positive target embeddings "
                        "must be strictly positive"
                    )

                user_sums = (
                    context_store
                    .aggregate_user_sums(
                        positive_context_weight,
                        incidence=incidence,
                    )
                )

                positive_contexts = (
                    context_store
                    .context_vectors(
                        context_embeddings=(
                            positive_context_weight
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

                positive_rate = (
                    module.additive_rate(
                        pos_items,
                        positive_contexts,
                    )
                )

                self._validate_positive_rate(
                    positive_rate,
                    name="positive Poisson rate",
                )

                positive_log_rate = torch.log(
                    positive_rate
                )

                positive_nll_sum = (
                    positive_rate
                    -
                    positive_log_rate
                ).sum()

                zero_contexts = (
                    context_store
                    .context_vectors(
                        context_embeddings=(
                            positive_context_weight
                        ),
                        user_indices=(
                            zero_users
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

                zero_rate = (
                    module.additive_rate(
                        zero_items,
                        zero_contexts,
                    )
                )

                self._validate_positive_rate(
                    zero_rate,
                    name="zero-cell Poisson rate",
                )

                zero_log_rate = torch.log(
                    zero_rate
                )

                weighted_zero_nll_sum = (
                    zero_rate
                    *
                    zero_weight_tensor
                ).sum()

                if not torch.isfinite(
                    positive_nll_sum
                ):
                    raise RuntimeError(
                        "ap-emb produced non-finite "
                        "positive Poisson NLL"
                    )

                if not torch.isfinite(
                    weighted_zero_nll_sum
                ):
                    raise RuntimeError(
                        "ap-emb produced non-finite "
                        "weighted zero NLL"
                    )

                data_loss = (
                    float(
                        n_batches
                    )
                    /
                    float(
                        legal_cells
                    )
                ) * (
                    positive_nll_sum
                    +
                    weighted_zero_nll_sum
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
                        "ap-emb produced a non-finite loss"
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
                            "ap-emb produced a "
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
                            "ap-emb produced "
                            "non-finite raw parameters"
                        )

                epoch_positive_nll_sum += float(
                    positive_nll_sum
                    .detach()
                    .cpu()
                    .item()
                )

                epoch_zero_estimate_sum += float(
                    weighted_zero_nll_sum
                    .detach()
                    .cpu()
                    .item()
                )

                epoch_positive_examples += len(
                    pos_users_np
                )

                epoch_zero_samples += len(
                    zero_users_np
                )

                if len(
                    positive_rate
                ) > 0:
                    max_rate = max(
                        max_rate,
                        float(
                            positive_rate
                            .max()
                            .detach()
                            .cpu()
                            .item()
                        ),
                    )

                    max_abs_log_rate = max(
                        max_abs_log_rate,
                        float(
                            positive_log_rate
                            .abs()
                            .max()
                            .detach()
                            .cpu()
                            .item()
                        ),
                    )

                if len(
                    zero_rate
                ) > 0:
                    max_rate = max(
                        max_rate,
                        float(
                            zero_rate
                            .max()
                            .detach()
                            .cpu()
                            .item()
                        ),
                    )

                    max_abs_log_rate = max(
                        max_abs_log_rate,
                        float(
                            zero_log_rate
                            .abs()
                            .max()
                            .detach()
                            .cpu()
                            .item()
                        ),
                    )

                batches += 1

            if (
                epoch_positive_examples
                !=
                n_positive
            ):
                raise RuntimeError(
                    "ap-emb did not process every "
                    "positive exactly once"
                )

            if (
                epoch_zero_samples
                !=
                n_zero_samples
            ):
                raise RuntimeError(
                    "ap-emb did not process every "
                    "sampled zero exactly once"
                )

            zero_weight_sum = float(
                zero_weights.sum(
                    dtype=np.float64
                )
            )

            if not np.isclose(
                zero_weight_sum,
                float(
                    zero_population_cells
                ),
                rtol=1e-12,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "ap-emb sampled-zero weights do not "
                    "sum to the legal zero population"
                )

            positive_mean_nll = (
                epoch_positive_nll_sum
                /
                float(
                    n_positive
                )
            )

            zero_estimated_mean_rate = (
                epoch_zero_estimate_sum
                /
                float(
                    zero_population_cells
                )
            )

            data_nll = (
                epoch_positive_nll_sum
                +
                epoch_zero_estimate_sum
            ) / float(
                legal_cells
            )

            with torch.no_grad():
                regularization_value = float(
                    self._regularization()
                    .detach()
                    .cpu()
                    .item()
                )

            total_loss_value = (
                data_nll
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
                "positive_mean_nll": float(
                    positive_mean_nll
                ),
                "zero_estimated_mean_rate": float(
                    zero_estimated_mean_rate
                ),
                "data_nll": float(
                    data_nll
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
                "zero_samples": int(
                    epoch_zero_samples
                ),
                "zero_population_cells": int(
                    zero_population_cells
                ),
                "zero_weight_sum": float(
                    zero_weight_sum
                ),
                "legal_cells": int(
                    legal_cells
                ),
                "heldout_masked_cells": int(
                    heldout_count
                ),
                "zero_ratio": int(
                    self.zero_ratio
                ),
                "max_rate": float(
                    max_rate
                ),
                "max_abs_log_rate": float(
                    max_abs_log_rate
                ),
                "batches": int(
                    batches
                ),
            }

            for key in [
                "positive_mean_nll",
                "zero_estimated_mean_rate",
                "data_nll",
                "regularization",
                "total_loss",
                "zero_weight_sum",
                "max_rate",
                "max_abs_log_rate",
            ]:
                if not np.isfinite(
                    float(
                        record[
                            key
                        ]
                    )
                ):
                    raise RuntimeError(
                        "ap-emb training history contains "
                        f"non-finite {key}"
                    )

            self._history.append(
                record
            )

        module.eval()

        self._refresh_evaluation_context_cache()

        self._is_fitted = True

        return self

    def _candidate_rates(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return exact additive Poisson rates for candidates."""
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
            rates = (
                self._module
                .additive_rate(
                    item_tensor,
                    contexts,
                )
            )

            self._validate_positive_rate(
                rates,
                name="inference Poisson rate",
            )

        result = (
            rates
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
                "ap-emb returned invalid rate shape"
            )

        if not np.isfinite(
            result
        ).all():
            raise RuntimeError(
                "ap-emb returned non-finite rates"
            )

        if np.any(
            result <= 0.0
        ):
            raise RuntimeError(
                "ap-emb returned non-positive rates"
            )

        return result

    def rate(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return strictly positive additive Poisson rates."""
        return self._candidate_rates(
            user_id,
            candidate_items,
        )

    def native_metrics(
        self,
        data: pd.DataFrame,
    ) -> dict[str, float]:
        """Return exact masked predictive Poisson NLL diagnostics.

        Observed interactions use the implicit occurrence target y=1.
        Genuine unobserved cells use y=0.
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
                "ap-emb native metrics require heldout_interactions"
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
                "ap-emb native_metrics data must be a subset "
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

                positive_rates = self.rate(
                    user_id,
                    positives,
                )

                positive_nll = (
                    positive_rates
                    -
                    np.log(
                        positive_rates
                    )
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

                zero_rates = self.rate(
                    user_id,
                    zeros,
                )

                zero_nll_sum += float(
                    zero_rates.sum(
                        dtype=np.float64
                    )
                )

                zero_cells += len(
                    zeros
                )

        if positive_cells != len(data):
            raise RuntimeError(
                "ap-emb native metric positive-cell accounting mismatch"
            )

        if zero_cells <= 0:
            raise RuntimeError(
                "ap-emb native metric zero-cell universe is empty"
            )

        if (
            self._zero_population_cells
            is not None
            and zero_cells
            != self._zero_population_cells
        ):
            raise RuntimeError(
                "ap-emb native metric zero-cell accounting mismatch"
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
            "poisson_positive_mean_nll": (
                float(
                    positive_mean_nll
                )
            ),
            "poisson_zero_mean_nll": (
                float(
                    zero_mean_nll
                )
            ),
            "masked_predictive_poisson_nll": (
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
                "ap-emb native metrics are non-finite or negative"
            )

        return metrics
    
    def score(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return log additive-Poisson rate for ranking."""
        rates = self._candidate_rates(
            user_id,
            candidate_items,
        )

        scores = np.log(
            rates
        )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                "ap-emb returned non-finite "
                "log-rate ranking scores"
            )

        return scores.astype(
            np.float64,
            copy=False,
        )
    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted ap-emb state."""
        self._require_fitted()

        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "ap-emb save path must end in .npz"
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
            raw_target_embeddings=(
                self._module
                .raw_target_embeddings
                .weight
                .detach()
                .cpu()
                .numpy()
            ),
            raw_context_embeddings=(
                self._module
                .raw_context_embeddings
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
                "Poisson"
            ),
            "architecture": (
                "EF-EMB"
            ),
            "poisson_mean": (
                "additive_positive"
            ),
            "positive_transform": (
                "exp"
            ),
            "regularization_space": (
                "raw_log_parameters"
            ),
            "ranking_score": (
                "log_additive_rate"
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
            "zero_sampling_strategy": (
                self.zero_sampling_strategy
            ),
            "zero_replace": (
                self.zero_replace
            ),
            "zero_sampling_correction": (
                "|Z_u|/m_u"
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
            "initial_interaction_rate": (
                self.initial_interaction_rate
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
            "masked_full_poisson_objective": (
                True
            ),
            "heldout_positive_cells_masked": (
                True
            ),
            "zero_population_cells": (
                self._zero_population_cells
            ),
            "legal_cells": (
                self._legal_cells
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
    ) -> AdditivePoissonEFEmb:
        """Load fitted ap-emb for deterministic scoring."""
        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "ap-emb load path must end in .npz"
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
                "Saved model is not ap-emb"
            )

        if metadata.get(
            "spec_revision"
        ) != "v1.2":
            raise ValueError(
                "Saved ap-emb does not use "
                "model-spec revision v1.2"
            )

        if metadata.get(
            "poisson_mean"
        ) != "additive_positive":
            raise ValueError(
                "Saved ap-emb is not the frozen "
                "additive Poisson model"
            )

        if metadata.get(
            "positive_transform"
        ) != "exp":
            raise ValueError(
                "Saved ap-emb does not use the "
                "frozen exponential positivity transform"
            )

        if metadata.get(
            "regularization_space"
        ) != "raw_log_parameters":
            raise ValueError(
                "Saved ap-emb does not use "
                "log-space regularization"
            )

        if metadata.get(
            "ranking_score"
        ) != "log_additive_rate":
            raise ValueError(
                "Saved ap-emb has an invalid "
                "ranking-score definition"
            )

        if metadata.get(
            "zero_sampling_correction"
        ) != "|Z_u|/m_u":
            raise ValueError(
                "Saved ap-emb lacks the frozen "
                "sampled-zero correction"
            )

        if metadata.get(
            "masked_full_poisson_objective"
        ) is not True:
            raise ValueError(
                "Saved ap-emb does not represent "
                "the masked full Poisson objective"
            )

        if metadata.get(
            "heldout_positive_cells_masked"
        ) is not True:
            raise ValueError(
                "Saved ap-emb does not confirm "
                "held-out-positive masking"
            )

        if metadata.get(
            "uses_user_embedding_table"
        ) is not False:
            raise ValueError(
                "Saved ap-emb unexpectedly contains "
                "a learned user embedding table"
            )

        if metadata.get(
            "target_context_embeddings_separate"
        ) is not True:
            raise ValueError(
                "Saved ap-emb does not preserve "
                "separate target/context embeddings"
            )

        model = cls(
            zero_sampler=None,
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
            zero_ratio=int(
                metadata[
                    "zero_ratio"
                ]
            ),
            zero_sampling_strategy=str(
                metadata[
                    "zero_sampling_strategy"
                ]
            ),
            zero_replace=bool(
                metadata[
                    "zero_replace"
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
            initial_interaction_rate=float(
                metadata[
                    "initial_interaction_rate"
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

            raw_target_embeddings = np.asarray(
                state[
                    "raw_target_embeddings"
                ],
                dtype=np.float32,
            )

            raw_context_embeddings = np.asarray(
                state[
                    "raw_context_embeddings"
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

        if raw_target_embeddings.shape != (
            n_items,
            k,
        ):
            raise ValueError(
                "Saved ap-emb raw target embeddings "
                "have invalid shape"
            )

        if raw_context_embeddings.shape != (
            n_items,
            k,
        ):
            raise ValueError(
                "Saved ap-emb raw context embeddings "
                "have invalid shape"
            )

        if item_bias.shape != (
            n_items,
            1,
        ):
            raise ValueError(
                "Saved ap-emb item bias "
                "has invalid shape"
            )

        if train_user_indices.shape != (
            len(
                train_item_indices
            ),
        ):
            raise ValueError(
                "Saved ap-emb training-index arrays "
                "have inconsistent shape"
            )

        if np.any(
            train_user_indices < 0
        ) or np.any(
            train_user_indices >= n_users
        ):
            raise ValueError(
                "Saved ap-emb train user indices "
                "are out of range"
            )

        if np.any(
            train_item_indices < 0
        ) or np.any(
            train_item_indices >= n_items
        ):
            raise ValueError(
                "Saved ap-emb train item indices "
                "are out of range"
            )

        arrays = [
            raw_target_embeddings,
            raw_context_embeddings,
            item_bias,
        ]

        if any(
            not np.isfinite(
                array
            ).all()
            for array in arrays
        ):
            raise ValueError(
                "Saved ap-emb state contains "
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
                "Saved ap-emb user mapping "
                "could not be reconstructed"
            )

        if not np.array_equal(
            context_store.item_catalog,
            item_ids,
        ):
            raise ValueError(
                "Saved ap-emb item mapping "
                "could not be reconstructed"
            )

        initial_biases = np.zeros(
            n_items,
            dtype=np.float32,
        )

        init_generator = (
            make_torch_generator(
                model.seed,
                "ap-emb.load_initialization",
                device="cpu",
            )
        )

        module = _APEmbModule(
            n_items=n_items,
            n_factors=k,
            initial_item_biases=(
                initial_biases
            ),
            initial_interaction_rate=(
                model.initial_interaction_rate
            ),
            init_generator=(
                init_generator
            ),
        )

        with torch.no_grad():
            module.raw_target_embeddings.weight.copy_(
                torch.from_numpy(
                    raw_target_embeddings
                )
            )

            module.raw_context_embeddings.weight.copy_(
                torch.from_numpy(
                    raw_context_embeddings
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

        model._zero_population_cells = (
            int(
                metadata[
                    "zero_population_cells"
                ]
            )
            if metadata.get(
                "zero_population_cells"
            ) is not None
            else None
        )

        model._legal_cells = (
            int(
                metadata[
                    "legal_cells"
                ]
            )
            if metadata.get(
                "legal_cells"
            ) is not None
            else None
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