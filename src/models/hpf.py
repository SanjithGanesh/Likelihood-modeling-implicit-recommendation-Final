"""Hierarchical Poisson Factorization for Paper 2.

This module implements the genuine hierarchical Gamma-Poisson model
frozen in docs/model_spec.md.

For user u and item i:

    xi_u ~ Gamma(a0, a0 / b0)
    theta_uk ~ Gamma(a, xi_u)

    eta_i ~ Gamma(c0, c0 / d0)
    beta_ik ~ Gamma(c, eta_i)

    y_ui ~ Poisson(theta_u^T beta_i)

The primary Paper 2 implicit target is binary occurrence:

    y_ui = 1 for canonical training interactions
    y_ui = 0 for genuinely unobserved legal cells

Validation/test positives are masked from the likelihood and are never
treated as zero counts.

Inference uses deterministic mean-field variational coordinate updates
with Gamma factors.

HPF uses NO synthetic negative sampler.

The ranking score is the posterior-mean Poisson rate:

    score_ui =
        sum_k E[theta_uk] E[beta_ik]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.special import (
    digamma,
    logsumexp,
)

from src.reproducibility import (
    make_numpy_rng,
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
            f"{name} contains missing userId/movieId values"
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
            "candidate_items contains duplicate item IDs"
        )

    return items


def _require_finite_positive(
    array: np.ndarray,
    *,
    name: str,
) -> None:
    """Require an array to be finite and strictly positive."""
    if not np.isfinite(
        array
    ).all():
        raise RuntimeError(
            f"{name} contains non-finite values"
        )

    if np.any(
        array <= 0.0
    ):
        raise RuntimeError(
            f"{name} contains non-positive values"
        )


class HPF:
    """Hierarchical Poisson Factorization."""

    model_name = "HPF"

    def __init__(
        self,
        heldout_interactions: pd.DataFrame | None = None,
        n_factors: int = 32,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        a0: float = 0.3,
        a: float = 0.3,
        b0: float = 1.0,
        c0: float = 0.3,
        c: float = 0.3,
        d0: float = 1.0,
        initialization_scale: float = 0.1,
        seed: int = 2026,
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
                max_iterations,
                int,
            )
            or isinstance(
                max_iterations,
                bool,
            )
            or max_iterations <= 0
        ):
            raise ValueError(
                "max_iterations must be a positive integer"
            )

        if (
            not np.isfinite(
                tolerance
            )
            or tolerance <= 0.0
        ):
            raise ValueError(
                "tolerance must be finite and positive"
            )

        hyperparameters = {
            "a0": a0,
            "a": a,
            "b0": b0,
            "c0": c0,
            "c": c,
            "d0": d0,
            "initialization_scale": initialization_scale,
        }

        for name, value in hyperparameters.items():
            if (
                not np.isfinite(
                    value
                )
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and positive"
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
        self.max_iterations = max_iterations
        self.tolerance = float(
            tolerance
        )

        self.a0 = float(
            a0
        )

        self.a = float(
            a
        )

        self.b0 = float(
            b0
        )

        self.c0 = float(
            c0
        )

        self.c = float(
            c
        )

        self.d0 = float(
            d0
        )

        self.initialization_scale = float(
            initialization_scale
        )

        self.seed = seed

        self._user_ids: np.ndarray | None = None
        self._item_ids: np.ndarray | None = None

        self._user_to_index: dict[
            int,
            int,
        ] | None = None

        self._item_to_index: dict[
            int,
            int,
        ] | None = None

        self._train_user_indices: np.ndarray | None = None
        self._train_item_indices: np.ndarray | None = None

        self._heldout_user_indices: np.ndarray | None = None
        self._heldout_item_indices: np.ndarray | None = None

        self._gamma_shape: np.ndarray | None = None
        self._gamma_rate: np.ndarray | None = None

        self._lambda_shape: np.ndarray | None = None
        self._lambda_rate: np.ndarray | None = None

        self._kappa_shape: np.ndarray | None = None
        self._kappa_rate: np.ndarray | None = None

        self._tau_shape: np.ndarray | None = None
        self._tau_rate: np.ndarray | None = None

        self._legal_cell_count: int | None = None
        self._positive_cell_count: int | None = None
        self._heldout_cell_count: int | None = None

        self._history: list[
            dict[str, float | int | bool]
        ] = []

        self._converged = False
        self._is_fitted = False

    @property
    def is_fitted(
        self,
    ) -> bool:
        return self._is_fitted

    @property
    def converged(
        self,
    ) -> bool:
        self._require_fitted()
        return self._converged

    @property
    def user_ids(
        self,
    ) -> np.ndarray:
        self._require_fitted()

        assert self._user_ids is not None

        return self._user_ids.copy()

    @property
    def item_catalog(
        self,
    ) -> np.ndarray:
        self._require_fitted()

        assert self._item_ids is not None

        return self._item_ids.copy()

    @property
    def training_history(
        self,
    ) -> list[
        dict[str, float | int | bool]
    ]:
        return [
            dict(row)
            for row
            in self._history
        ]

    @property
    def legal_cell_count(
        self,
    ) -> int:
        self._require_fitted()

        assert self._legal_cell_count is not None

        return int(
            self._legal_cell_count
        )

    @property
    def positive_cell_count(
        self,
    ) -> int:
        self._require_fitted()

        assert self._positive_cell_count is not None

        return int(
            self._positive_cell_count
        )

    @property
    def zero_cell_count(
        self,
    ) -> int:
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
        self._require_fitted()

        assert self._heldout_cell_count is not None

        return int(
            self._heldout_cell_count
        )

    @property
    def expected_user_factors(
        self,
    ) -> np.ndarray:
        self._require_fitted()

        return self._expected_theta().copy()

    @property
    def expected_item_factors(
        self,
    ) -> np.ndarray:
        self._require_fitted()

        return self._expected_beta().copy()

    def _require_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "HPF must be fitted before scoring"
            )

    def _require_heldout_mask(
        self,
    ) -> pd.DataFrame:
        if self.heldout_interactions is None:
            raise RuntimeError(
                "HPF requires heldout_interactions during "
                "fit() so validation/test positives can be "
                "masked rather than treated as zero counts"
            )

        return self.heldout_interactions

    def _canonicalize(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        result = (
            data[
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

        result["userId"] = (
            result[
                "userId"
            ]
            .astype(
                "int64"
            )
        )

        result["movieId"] = (
            result[
                "movieId"
            ]
            .astype(
                "int64"
            )
        )

        return result

    def _prepare_indices(
        self,
        train: pd.DataFrame,
        heldout: pd.DataFrame,
    ) -> None:
        assert self._user_ids is not None
        assert self._item_ids is not None
        assert self._user_to_index is not None
        assert self._item_to_index is not None

        train_pairs = set(
            zip(
                train[
                    "userId"
                ].tolist(),
                train[
                    "movieId"
                ].tolist(),
            )
        )

        heldout_pairs = set(
            zip(
                heldout[
                    "userId"
                ].tolist(),
                heldout[
                    "movieId"
                ].tolist(),
            )
        )

        if (
            train_pairs
            &
            heldout_pairs
        ):
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
                if int(
                    user_id
                )
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
                if int(
                    item_id
                )
                not in self._item_to_index
            }
        )

        if unknown_items:
            raise ValueError(
                "heldout_interactions contains items "
                "outside the training item catalog: "
                f"{unknown_items[:10]}"
            )

        self._train_user_indices = np.fromiter(
            (
                self._user_to_index[
                    int(user_id)
                ]
                for user_id
                in train[
                    "userId"
                ]
            ),
            dtype=np.int64,
            count=len(
                train
            ),
        )

        self._train_item_indices = np.fromiter(
            (
                self._item_to_index[
                    int(item_id)
                ]
                for item_id
                in train[
                    "movieId"
                ]
            ),
            dtype=np.int64,
            count=len(
                train
            ),
        )

        self._heldout_user_indices = np.fromiter(
            (
                self._user_to_index[
                    int(user_id)
                ]
                for user_id
                in heldout[
                    "userId"
                ]
            ),
            dtype=np.int64,
            count=len(
                heldout
            ),
        )

        self._heldout_item_indices = np.fromiter(
            (
                self._item_to_index[
                    int(item_id)
                ]
                for item_id
                in heldout[
                    "movieId"
                ]
            ),
            dtype=np.int64,
            count=len(
                heldout
            ),
        )

    def _initialize_variational_parameters(
        self,
    ) -> None:
        """Deterministic positive initialization.

        Initialization uses only:
        - training dimensions;
        - frozen prior hyperparameters;
        - deterministic RNG.

        Held-out interaction identities do not influence initialization.
        """
        assert self._user_ids is not None
        assert self._item_ids is not None

        n_users = len(
            self._user_ids
        )

        n_items = len(
            self._item_ids
        )

        k = self.n_factors

        rng = make_numpy_rng(
            self.seed,
            "HPF.variational_initialization",
        )

        user_jitter = rng.uniform(
            0.5,
            1.5,
            size=(
                n_users,
                k,
            ),
        )

        item_jitter = rng.uniform(
            0.5,
            1.5,
            size=(
                n_items,
                k,
            ),
        )

        self._gamma_shape = (
            self.a
            +
            self.initialization_scale
            *
            user_jitter
        ).astype(
            np.float64,
            copy=False,
        )

        self._gamma_rate = (
            1.0
            +
            self.initialization_scale
            *
            user_jitter
        ).astype(
            np.float64,
            copy=False,
        )

        self._lambda_shape = (
            self.c
            +
            self.initialization_scale
            *
            item_jitter
        ).astype(
            np.float64,
            copy=False,
        )

        self._lambda_rate = (
            1.0
            +
            self.initialization_scale
            *
            item_jitter
        ).astype(
            np.float64,
            copy=False,
        )

        theta_mean = (
            self._gamma_shape
            /
            self._gamma_rate
        )

        beta_mean = (
            self._lambda_shape
            /
            self._lambda_rate
        )

        self._kappa_shape = np.full(
            n_users,
            self.a0
            +
            k
            *
            self.a,
            dtype=np.float64,
        )

        self._kappa_rate = (
            self.a0
            /
            self.b0
            +
            theta_mean.sum(
                axis=1
            )
        )

        self._tau_shape = np.full(
            n_items,
            self.c0
            +
            k
            *
            self.c,
            dtype=np.float64,
        )

        self._tau_rate = (
            self.c0
            /
            self.d0
            +
            beta_mean.sum(
                axis=1
            )
        )

        self._validate_variational_state()

    def _validate_variational_state(
        self,
    ) -> None:
        arrays = {
            "gamma_shape": self._gamma_shape,
            "gamma_rate": self._gamma_rate,
            "lambda_shape": self._lambda_shape,
            "lambda_rate": self._lambda_rate,
            "kappa_shape": self._kappa_shape,
            "kappa_rate": self._kappa_rate,
            "tau_shape": self._tau_shape,
            "tau_rate": self._tau_rate,
        }

        for name, array in arrays.items():
            if array is None:
                raise RuntimeError(
                    f"{name} is not initialized"
                )

            _require_finite_positive(
                array,
                name=name,
            )

    def _expected_theta(
        self,
    ) -> np.ndarray:
        assert self._gamma_shape is not None
        assert self._gamma_rate is not None

        value = (
            self._gamma_shape
            /
            self._gamma_rate
        )

        _require_finite_positive(
            value,
            name="E[theta]",
        )

        return value

    def _expected_beta(
        self,
    ) -> np.ndarray:
        assert self._lambda_shape is not None
        assert self._lambda_rate is not None

        value = (
            self._lambda_shape
            /
            self._lambda_rate
        )

        _require_finite_positive(
            value,
            name="E[beta]",
        )

        return value

    def _expected_log_theta(
        self,
    ) -> np.ndarray:
        assert self._gamma_shape is not None
        assert self._gamma_rate is not None

        value = (
            digamma(
                self._gamma_shape
            )
            -
            np.log(
                self._gamma_rate
            )
        )

        if not np.isfinite(
            value
        ).all():
            raise RuntimeError(
                "E[log theta] contains non-finite values"
            )

        return value

    def _expected_log_beta(
        self,
    ) -> np.ndarray:
        assert self._lambda_shape is not None
        assert self._lambda_rate is not None

        value = (
            digamma(
                self._lambda_shape
            )
            -
            np.log(
                self._lambda_rate
            )
        )

        if not np.isfinite(
            value
        ).all():
            raise RuntimeError(
                "E[log beta] contains non-finite values"
            )

        return value

    def _expected_xi(
        self,
    ) -> np.ndarray:
        assert self._kappa_shape is not None
        assert self._kappa_rate is not None

        value = (
            self._kappa_shape
            /
            self._kappa_rate
        )

        _require_finite_positive(
            value,
            name="E[xi]",
        )

        return value

    def _expected_eta(
        self,
    ) -> np.ndarray:
        assert self._tau_shape is not None
        assert self._tau_rate is not None

        value = (
            self._tau_shape
            /
            self._tau_rate
        )

        _require_finite_positive(
            value,
            name="E[eta]",
        )

        return value

    def _legal_beta_sum_by_user(
        self,
        expected_beta: np.ndarray,
    ) -> np.ndarray:
        """Compute sum_{i:w_ui=1} E[beta_i] exactly."""
        assert self._user_ids is not None
        assert self._heldout_user_indices is not None
        assert self._heldout_item_indices is not None

        n_users = len(
            self._user_ids
        )

        total_beta = expected_beta.sum(
            axis=0
        )

        masked_beta = np.zeros(
            (
                n_users,
                self.n_factors,
            ),
            dtype=np.float64,
        )

        if len(
            self._heldout_user_indices
        ) > 0:
            np.add.at(
                masked_beta,
                self._heldout_user_indices,
                expected_beta[
                    self._heldout_item_indices
                ],
            )

        legal_sum = (
            total_beta[
                None,
                :
            ]
            -
            masked_beta
        )

        _require_finite_positive(
            legal_sum,
            name="legal beta sums",
        )

        return legal_sum

    def _legal_theta_sum_by_item(
        self,
        expected_theta: np.ndarray,
    ) -> np.ndarray:
        """Compute sum_{u:w_ui=1} E[theta_u] exactly."""
        assert self._item_ids is not None
        assert self._heldout_user_indices is not None
        assert self._heldout_item_indices is not None

        n_items = len(
            self._item_ids
        )

        total_theta = expected_theta.sum(
            axis=0
        )

        masked_theta = np.zeros(
            (
                n_items,
                self.n_factors,
            ),
            dtype=np.float64,
        )

        if len(
            self._heldout_item_indices
        ) > 0:
            np.add.at(
                masked_theta,
                self._heldout_item_indices,
                expected_theta[
                    self._heldout_user_indices
                ],
            )

        legal_sum = (
            total_theta[
                None,
                :
            ]
            -
            masked_theta
        )

        _require_finite_positive(
            legal_sum,
            name="legal theta sums",
        )

        return legal_sum

    def _latent_allocations(
        self,
    ) -> np.ndarray:
        """Calculate phi_uik for canonical positive interactions."""
        assert self._train_user_indices is not None
        assert self._train_item_indices is not None

        expected_log_theta = (
            self._expected_log_theta()
        )

        expected_log_beta = (
            self._expected_log_beta()
        )

        log_phi = (
            expected_log_theta[
                self._train_user_indices
            ]
            +
            expected_log_beta[
                self._train_item_indices
            ]
        )

        log_phi = (
            log_phi
            -
            logsumexp(
                log_phi,
                axis=1,
                keepdims=True,
            )
        )

        phi = np.exp(
            log_phi
        )

        if not np.isfinite(
            phi
        ).all():
            raise RuntimeError(
                "HPF latent allocations contain "
                "non-finite values"
            )

        if np.any(
            phi <= 0.0
        ):
            raise RuntimeError(
                "HPF latent allocations contain "
                "non-positive values"
            )

        row_sums = phi.sum(
            axis=1
        )

        if not np.allclose(
            row_sums,
            1.0,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "HPF latent allocations do not sum to one"
            )

        return phi

    def _predictive_poisson_nll(
        self,
        expected_theta: np.ndarray,
        expected_beta: np.ndarray,
    ) -> float:
        """Posterior-mean masked Poisson NLL diagnostic.

        This is a predictive diagnostic, not the variational ELBO.
        """
        assert self._train_user_indices is not None
        assert self._train_item_indices is not None
        assert self._heldout_user_indices is not None
        assert self._heldout_item_indices is not None
        assert self._legal_cell_count is not None

        theta_sum = expected_theta.sum(
            axis=0
        )

        beta_sum = expected_beta.sum(
            axis=0
        )

        all_rate_sum = float(
            np.dot(
                theta_sum,
                beta_sum,
            )
        )

        if len(
            self._heldout_user_indices
        ) > 0:
            heldout_rates = (
                expected_theta[
                    self._heldout_user_indices
                ]
                *
                expected_beta[
                    self._heldout_item_indices
                ]
            ).sum(
                axis=1
            )

            masked_rate_sum = float(
                heldout_rates.sum()
            )
        else:
            masked_rate_sum = 0.0

        legal_rate_sum = (
            all_rate_sum
            -
            masked_rate_sum
        )

        positive_rates = (
            expected_theta[
                self._train_user_indices
            ]
            *
            expected_beta[
                self._train_item_indices
            ]
        ).sum(
            axis=1
        )

        _require_finite_positive(
            positive_rates,
            name="positive predictive rates",
        )

        positive_log_rate_sum = float(
            np.log(
                positive_rates
            ).sum()
        )

        nll = (
            legal_rate_sum
            -
            positive_log_rate_sum
        ) / float(
            self._legal_cell_count
        )

        if not np.isfinite(
            nll
        ):
            raise RuntimeError(
                "HPF predictive Poisson NLL is non-finite"
            )

        return float(
            nll
        )

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> HPF:
        """Fit HPF using deterministic mean-field variational inference."""
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

        train = self._canonicalize(
            train_data
        )

        heldout = self._canonicalize(
            heldout
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

        self._user_ids = user_ids
        self._item_ids = item_ids

        self._user_to_index = {
            int(user_id): index
            for index, user_id
            in enumerate(
                user_ids
            )
        }

        self._item_to_index = {
            int(item_id): index
            for index, item_id
            in enumerate(
                item_ids
            )
        }

        self._prepare_indices(
            train,
            heldout,
        )

        n_users = len(
            user_ids
        )

        n_items = len(
            item_ids
        )

        total_cells = (
            n_users
            *
            n_items
        )

        positive_cell_count = len(
            train
        )

        heldout_cell_count = len(
            heldout
        )

        legal_cell_count = (
            total_cells
            -
            heldout_cell_count
        )

        zero_cell_count = (
            legal_cell_count
            -
            positive_cell_count
        )

        if legal_cell_count <= 0:
            raise RuntimeError(
                "HPF has no legal likelihood cells"
            )

        if positive_cell_count <= 0:
            raise RuntimeError(
                "HPF has no positive training cells"
            )

        if zero_cell_count < 0:
            raise RuntimeError(
                "HPF legal-cell accounting is invalid"
            )

        self._positive_cell_count = int(
            positive_cell_count
        )

        self._heldout_cell_count = int(
            heldout_cell_count
        )

        self._legal_cell_count = int(
            legal_cell_count
        )

        self._initialize_variational_parameters()

        self._history = []
        self._converged = False
        self._is_fitted = False

        assert self._train_user_indices is not None
        assert self._train_item_indices is not None

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            old_theta = (
                self._expected_theta()
            )

            old_beta = (
                self._expected_beta()
            )

            phi = (
                self._latent_allocations()
            )

            user_allocations = np.zeros(
                (
                    n_users,
                    self.n_factors,
                ),
                dtype=np.float64,
            )

            item_allocations = np.zeros(
                (
                    n_items,
                    self.n_factors,
                ),
                dtype=np.float64,
            )

            np.add.at(
                user_allocations,
                self._train_user_indices,
                phi,
            )

            np.add.at(
                item_allocations,
                self._train_item_indices,
                phi,
            )

            allocation_total = float(
                phi.sum()
            )

            if not np.isclose(
                allocation_total,
                float(
                    positive_cell_count
                ),
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "HPF positive latent-count "
                    "allocation total is incorrect"
                )

            # --------------------------------------
            # User factors q(theta)
            # --------------------------------------

            assert self._kappa_shape is not None
            assert self._kappa_rate is not None

            expected_xi = (
                self._expected_xi()
            )

            current_beta = (
                self._expected_beta()
            )

            legal_beta_sum = (
                self._legal_beta_sum_by_user(
                    current_beta
                )
            )

            gamma_shape_new = (
                self.a
                +
                user_allocations
            )

            gamma_rate_new = (
                expected_xi[
                    :,
                    None,
                ]
                +
                legal_beta_sum
            )

            _require_finite_positive(
                gamma_shape_new,
                name="updated gamma_shape",
            )

            _require_finite_positive(
                gamma_rate_new,
                name="updated gamma_rate",
            )

            theta_new = (
                gamma_shape_new
                /
                gamma_rate_new
            )

            _require_finite_positive(
                theta_new,
                name="updated E[theta]",
            )

            # --------------------------------------
            # User activity q(xi)
            # --------------------------------------

            kappa_shape_new = np.full(
                n_users,
                self.a0
                +
                self.n_factors
                *
                self.a,
                dtype=np.float64,
            )

            kappa_rate_new = (
                self.a0
                /
                self.b0
                +
                theta_new.sum(
                    axis=1
                )
            )

            _require_finite_positive(
                kappa_shape_new,
                name="updated kappa_shape",
            )

            _require_finite_positive(
                kappa_rate_new,
                name="updated kappa_rate",
            )

            # --------------------------------------
            # Item factors q(beta)
            # --------------------------------------

            assert self._tau_shape is not None
            assert self._tau_rate is not None

            expected_eta = (
                self._expected_eta()
            )

            legal_theta_sum = (
                self._legal_theta_sum_by_item(
                    theta_new
                )
            )

            lambda_shape_new = (
                self.c
                +
                item_allocations
            )

            lambda_rate_new = (
                expected_eta[
                    :,
                    None,
                ]
                +
                legal_theta_sum
            )

            _require_finite_positive(
                lambda_shape_new,
                name="updated lambda_shape",
            )

            _require_finite_positive(
                lambda_rate_new,
                name="updated lambda_rate",
            )

            beta_new = (
                lambda_shape_new
                /
                lambda_rate_new
            )

            _require_finite_positive(
                beta_new,
                name="updated E[beta]",
            )

            # --------------------------------------
            # Item activity q(eta)
            # --------------------------------------

            tau_shape_new = np.full(
                n_items,
                self.c0
                +
                self.n_factors
                *
                self.c,
                dtype=np.float64,
            )

            tau_rate_new = (
                self.c0
                /
                self.d0
                +
                beta_new.sum(
                    axis=1
                )
            )

            _require_finite_positive(
                tau_shape_new,
                name="updated tau_shape",
            )

            _require_finite_positive(
                tau_rate_new,
                name="updated tau_rate",
            )

            # --------------------------------------
            # Commit complete coordinate update
            # --------------------------------------

            self._gamma_shape = gamma_shape_new
            self._gamma_rate = gamma_rate_new

            self._kappa_shape = kappa_shape_new
            self._kappa_rate = kappa_rate_new

            self._lambda_shape = lambda_shape_new
            self._lambda_rate = lambda_rate_new

            self._tau_shape = tau_shape_new
            self._tau_rate = tau_rate_new

            self._validate_variational_state()

            new_theta = (
                self._expected_theta()
            )

            new_beta = (
                self._expected_beta()
            )

            theta_relative_change = float(
                np.max(
                    np.abs(
                        new_theta
                        -
                        old_theta
                    )
                    /
                    np.maximum(
                        np.abs(
                            old_theta
                        ),
                        1e-12,
                    )
                )
            )

            beta_relative_change = float(
                np.max(
                    np.abs(
                        new_beta
                        -
                        old_beta
                    )
                    /
                    np.maximum(
                        np.abs(
                            old_beta
                        ),
                        1e-12,
                    )
                )
            )

            max_relative_change = max(
                theta_relative_change,
                beta_relative_change,
            )

            predictive_nll = (
                self._predictive_poisson_nll(
                    new_theta,
                    new_beta,
                )
            )

            record: dict[
                str,
                float | int | bool,
            ] = {
                "iteration": int(
                    iteration
                ),
                "max_relative_change": float(
                    max_relative_change
                ),
                "theta_relative_change": float(
                    theta_relative_change
                ),
                "beta_relative_change": float(
                    beta_relative_change
                ),
                "predictive_poisson_nll": float(
                    predictive_nll
                ),
                "mean_theta": float(
                    new_theta.mean()
                ),
                "mean_beta": float(
                    new_beta.mean()
                ),
                "positive_allocations": float(
                    allocation_total
                ),
                "legal_cells": int(
                    legal_cell_count
                ),
                "positive_cells": int(
                    positive_cell_count
                ),
                "zero_cells": int(
                    zero_cell_count
                ),
                "heldout_masked_cells": int(
                    heldout_cell_count
                ),
                "converged": bool(
                    max_relative_change
                    <
                    self.tolerance
                ),
            }

            numeric_keys = [
                "max_relative_change",
                "theta_relative_change",
                "beta_relative_change",
                "predictive_poisson_nll",
                "mean_theta",
                "mean_beta",
                "positive_allocations",
            ]

            for key in numeric_keys:
                if not np.isfinite(
                    float(
                        record[
                            key
                        ]
                    )
                ):
                    raise RuntimeError(
                        "HPF training diagnostics contain "
                        f"non-finite value for {key}"
                    )

            self._history.append(
                record
            )

            if (
                max_relative_change
                <
                self.tolerance
            ):
                self._converged = True
                break

        self._validate_variational_state()

        self._is_fitted = True

        return self

    def score(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Return posterior-mean Poisson rates."""
        self._require_fitted()

        items = (
            _validate_candidate_items(
                candidate_items
            )
        )

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
            if int(
                item_id
            )
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
            count=len(
                items
            ),
        )

        expected_theta = (
            self._expected_theta()
        )

        expected_beta = (
            self._expected_beta()
        )

        scores = (
            expected_beta[
                item_indices
            ]
            @
            expected_theta[
                user_index
            ]
        )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        )

        if scores.shape != (
            len(
                items
            ),
        ):
            raise RuntimeError(
                "HPF returned an invalid score shape"
            )

        _require_finite_positive(
            scores,
            name="HPF scores",
        )

        return scores

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted HPF variational parameters."""
        self._require_fitted()

        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "HPF save path must end in .npz"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        assert self._user_ids is not None
        assert self._item_ids is not None

        assert self._gamma_shape is not None
        assert self._gamma_rate is not None

        assert self._lambda_shape is not None
        assert self._lambda_rate is not None

        assert self._kappa_shape is not None
        assert self._kappa_rate is not None

        assert self._tau_shape is not None
        assert self._tau_rate is not None

        np.savez_compressed(
            path,
            user_ids=self._user_ids,
            item_ids=self._item_ids,
            gamma_shape=self._gamma_shape,
            gamma_rate=self._gamma_rate,
            lambda_shape=self._lambda_shape,
            lambda_rate=self._lambda_rate,
            kappa_shape=self._kappa_shape,
            kappa_rate=self._kappa_rate,
            tau_shape=self._tau_shape,
            tau_rate=self._tau_rate,
        )

        metadata: dict[
            str,
            Any,
        ] = {
            "model_name": self.model_name,
            "spec_revision": "v1.2",
            "model_family": (
                "hierarchical_gamma_poisson"
            ),
            "inference": (
                "mean_field_variational"
            ),
            "n_factors": self.n_factors,
            "max_iterations": (
                self.max_iterations
            ),
            "tolerance": (
                self.tolerance
            ),
            "a0": self.a0,
            "a": self.a,
            "b0": self.b0,
            "c0": self.c0,
            "c": self.c,
            "d0": self.d0,
            "initialization_scale": (
                self.initialization_scale
            ),
            "seed": self.seed,
            "legal_cell_count": (
                self.legal_cell_count
            ),
            "positive_cell_count": (
                self.positive_cell_count
            ),
            "zero_cell_count": (
                self.zero_cell_count
            ),
            "heldout_cell_count": (
                self.heldout_cell_count
            ),
            "uses_negative_sampler": False,
            "converged": (
                self._converged
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
    ) -> HPF:
        """Load fitted HPF for deterministic scoring."""
        path = Path(
            path
        )

        if path.suffix.lower() != ".npz":
            raise ValueError(
                "HPF load path must end in .npz"
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
                "Saved model is not HPF"
            )

        if metadata.get(
            "spec_revision"
        ) != "v1.2":
            raise ValueError(
                "Saved HPF does not use "
                "model-spec revision v1.2"
            )

        if metadata.get(
            "uses_negative_sampler"
        ) is not False:
            raise ValueError(
                "Saved HPF unexpectedly uses "
                "negative sampling"
            )

        model = cls(
            heldout_interactions=None,
            n_factors=int(
                metadata[
                    "n_factors"
                ]
            ),
            max_iterations=int(
                metadata[
                    "max_iterations"
                ]
            ),
            tolerance=float(
                metadata[
                    "tolerance"
                ]
            ),
            a0=float(
                metadata[
                    "a0"
                ]
            ),
            a=float(
                metadata[
                    "a"
                ]
            ),
            b0=float(
                metadata[
                    "b0"
                ]
            ),
            c0=float(
                metadata[
                    "c0"
                ]
            ),
            c=float(
                metadata[
                    "c"
                ]
            ),
            d0=float(
                metadata[
                    "d0"
                ]
            ),
            initialization_scale=float(
                metadata[
                    "initialization_scale"
                ]
            ),
            seed=int(
                metadata[
                    "seed"
                ]
            ),
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

            gamma_shape = np.asarray(
                state[
                    "gamma_shape"
                ],
                dtype=np.float64,
            )

            gamma_rate = np.asarray(
                state[
                    "gamma_rate"
                ],
                dtype=np.float64,
            )

            lambda_shape = np.asarray(
                state[
                    "lambda_shape"
                ],
                dtype=np.float64,
            )

            lambda_rate = np.asarray(
                state[
                    "lambda_rate"
                ],
                dtype=np.float64,
            )

            kappa_shape = np.asarray(
                state[
                    "kappa_shape"
                ],
                dtype=np.float64,
            )

            kappa_rate = np.asarray(
                state[
                    "kappa_rate"
                ],
                dtype=np.float64,
            )

            tau_shape = np.asarray(
                state[
                    "tau_shape"
                ],
                dtype=np.float64,
            )

            tau_rate = np.asarray(
                state[
                    "tau_rate"
                ],
                dtype=np.float64,
            )

        n_users = len(
            user_ids
        )

        n_items = len(
            item_ids
        )

        k = model.n_factors

        expected_shapes = {
            "gamma_shape": (
                gamma_shape,
                (
                    n_users,
                    k,
                ),
            ),
            "gamma_rate": (
                gamma_rate,
                (
                    n_users,
                    k,
                ),
            ),
            "lambda_shape": (
                lambda_shape,
                (
                    n_items,
                    k,
                ),
            ),
            "lambda_rate": (
                lambda_rate,
                (
                    n_items,
                    k,
                ),
            ),
            "kappa_shape": (
                kappa_shape,
                (
                    n_users,
                ),
            ),
            "kappa_rate": (
                kappa_rate,
                (
                    n_users,
                ),
            ),
            "tau_shape": (
                tau_shape,
                (
                    n_items,
                ),
            ),
            "tau_rate": (
                tau_rate,
                (
                    n_items,
                ),
            ),
        }

        for name, (
            array,
            expected_shape,
        ) in expected_shapes.items():
            if array.shape != expected_shape:
                raise ValueError(
                    f"Saved HPF {name} has "
                    "invalid shape"
                )

            _require_finite_positive(
                array,
                name=(
                    f"saved {name}"
                ),
            )

        if len(
            np.unique(
                user_ids
            )
        ) != n_users:
            raise ValueError(
                "Saved HPF user IDs contain duplicates"
            )

        if len(
            np.unique(
                item_ids
            )
        ) != n_items:
            raise ValueError(
                "Saved HPF item IDs contain duplicates"
            )

        model._user_ids = user_ids
        model._item_ids = item_ids

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

        model._gamma_shape = gamma_shape
        model._gamma_rate = gamma_rate

        model._lambda_shape = lambda_shape
        model._lambda_rate = lambda_rate

        model._kappa_shape = kappa_shape
        model._kappa_rate = kappa_rate

        model._tau_shape = tau_shape
        model._tau_rate = tau_rate

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

        model._converged = bool(
            metadata.get(
                "converged",
                False,
            )
        )

        model._is_fitted = True

        model._validate_variational_state()

        return model