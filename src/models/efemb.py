"""Shared exponential-family embedding context framework for Paper 2.

This module contains the representation/context machinery shared by:

    b-emb
    p-emb
    ap-emb
    g-emb
    ng-emb

It intentionally contains no family-specific likelihood.

Frozen Paper 2 rules implemented here:

- histories use canonical TRAIN interactions only;
- validation/test interactions never enter train-only contexts;
- histories are deterministic sorted sequences;
- target and context embeddings remain conceptually separate;
- a positive training target is excluded from its own context;
- an unobserved/evaluation target uses the complete train history;
- context values are binary interaction-presence values;
- supported normalization:
      raw   -> q = 0
      sqrt  -> q = 1/2
      mean  -> q = 1
- empty context maps to the zero vector;
- primary configuration uses the complete context;
- no hidden 2,000-pair cap exists;
- full-history sums are computed exactly through a sparse
  user-item incidence matrix rather than O(|H_u|^2) pair generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


REQUIRED_COLUMNS = {
    "userId",
    "movieId",
}


NORMALIZATION_EXPONENTS = {
    "raw": 0.0,
    "sqrt": 0.5,
    "mean": 1.0,
}


def _validate_interactions(
    data: pd.DataFrame,
    *,
    name: str,
    allow_empty: bool,
) -> None:
    """Validate a canonical user-item interaction frame."""
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


def _canonicalize(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Return canonical deterministic user/item ordering."""
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


def normalization_exponent(
    mode: str,
) -> float:
    """Return q for the frozen normalization family."""
    if mode not in NORMALIZATION_EXPONENTS:
        raise ValueError(
            "context normalization must be one of "
            f"{sorted(NORMALIZATION_EXPONENTS)}"
        )

    return float(
        NORMALIZATION_EXPONENTS[
            mode
        ]
    )


@dataclass(
    frozen=True
)
class EFContextMetadata:
    """Immutable structural metadata."""

    n_users: int
    n_items: int
    n_interactions: int
    min_history_length: int
    max_history_length: int
    mean_history_length: float


class EFContextStore:
    """Canonical train-only context representation shared by EF-EMB."""

    def __init__(
        self,
        train_data: pd.DataFrame,
        *,
        heldout_interactions: pd.DataFrame | None = None,
        max_context_items: int | None = None,
    ) -> None:
        _validate_interactions(
            train_data,
            name="train_data",
            allow_empty=False,
        )

        if heldout_interactions is not None:
            _validate_interactions(
                heldout_interactions,
                name="heldout_interactions",
                allow_empty=True,
            )

        # Primary Paper 2 uses the complete legal history.
        #
        # A future finite-context experiment requires explicit,
        # target-aware deterministic truncation machinery. Rejecting
        # finite values here prevents an accidental hidden cap.
        if max_context_items is not None:
            raise NotImplementedError(
                "Primary EF-EMB requires "
                "max_context_items=None. "
                "Finite context caps must be implemented "
                "as an explicit deterministic ablation."
            )

        train = _canonicalize(
            train_data
        )

        heldout = (
            None
            if heldout_interactions is None
            else _canonicalize(
                heldout_interactions
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

        if heldout is not None:
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
                    "canonical training interactions"
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
                    not in user_to_index
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
                    not in item_to_index
                }
            )

            if unknown_items:
                raise ValueError(
                    "heldout_interactions contains items "
                    "outside the training item catalog: "
                    f"{unknown_items[:10]}"
                )

        train_user_indices = np.fromiter(
            (
                user_to_index[
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

        train_item_indices = np.fromiter(
            (
                item_to_index[
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

        n_users = len(
            user_ids
        )

        n_items = len(
            item_ids
        )

        history_item_indices: list[
            tuple[int, ...]
        ] = [
            tuple()
            for _ in range(
                n_users
            )
        ]

        mutable_histories: list[
            list[int]
        ] = [
            []
            for _ in range(
                n_users
            )
        ]

        for user_index, item_index in zip(
            train_user_indices,
            train_item_indices,
        ):
            mutable_histories[
                int(
                    user_index
                )
            ].append(
                int(
                    item_index
                )
            )

        canonical_histories: list[
            tuple[int, ...]
        ] = []

        for history in mutable_histories:
            sorted_history = tuple(
                sorted(
                    history
                )
            )

            if len(
                set(
                    sorted_history
                )
            ) != len(
                sorted_history
            ):
                raise RuntimeError(
                    "Duplicate item detected in "
                    "canonical EF-EMB history"
                )

            canonical_histories.append(
                sorted_history
            )

        history_item_indices = (
            canonical_histories
        )

        history_lengths = np.asarray(
            [
                len(
                    history
                )
                for history
                in history_item_indices
            ],
            dtype=np.int64,
        )

        if np.any(
            history_lengths <= 0
        ):
            raise RuntimeError(
                "Every fitted EF-EMB user must have "
                "at least one training interaction"
            )

        self._train = train

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

        self._train_user_indices = (
            train_user_indices
        )

        self._train_item_indices = (
            train_item_indices
        )

        self._history_item_indices = tuple(
            history_item_indices
        )

        self._history_lengths = (
            history_lengths
        )

        self._heldout = heldout

        self._metadata = (
            EFContextMetadata(
                n_users=n_users,
                n_items=n_items,
                n_interactions=len(
                    train
                ),
                min_history_length=int(
                    history_lengths.min()
                ),
                max_history_length=int(
                    history_lengths.max()
                ),
                mean_history_length=float(
                    history_lengths.mean()
                ),
            )
        )

    @property
    def metadata(
        self,
    ) -> EFContextMetadata:
        return self._metadata

    @property
    def user_ids(
        self,
    ) -> np.ndarray:
        return self._user_ids.copy()

    @property
    def item_catalog(
        self,
    ) -> np.ndarray:
        return self._item_ids.copy()

    @property
    def history_lengths(
        self,
    ) -> np.ndarray:
        return self._history_lengths.copy()

    @property
    def train_user_indices(
        self,
    ) -> np.ndarray:
        return self._train_user_indices.copy()

    @property
    def train_item_indices(
        self,
    ) -> np.ndarray:
        return self._train_item_indices.copy()

    def user_index(
        self,
        user_id: int,
    ) -> int:
        user_id = int(
            user_id
        )

        if user_id not in self._user_to_index:
            raise ValueError(
                f"Unknown user ID: {user_id}"
            )

        return int(
            self._user_to_index[
                user_id
            ]
        )

    def item_index(
        self,
        item_id: int,
    ) -> int:
        item_id = int(
            item_id
        )

        if item_id not in self._item_to_index:
            raise ValueError(
                f"Unknown item ID: {item_id}"
            )

        return int(
            self._item_to_index[
                item_id
            ]
        )

    def history_indices(
        self,
        user_id: int,
    ) -> tuple[int, ...]:
        """Return immutable canonical training-history indices."""
        user_index = self.user_index(
            user_id
        )

        return self._history_item_indices[
            user_index
        ]

    def history_item_ids(
        self,
        user_id: int,
    ) -> tuple[int, ...]:
        """Return immutable sorted canonical item IDs."""
        indices = self.history_indices(
            user_id
        )

        return tuple(
            int(
                self._item_ids[
                    index
                ]
            )
            for index
            in indices
        )

    def positive_context_indices(
        self,
        user_id: int,
        target_item_id: int,
    ) -> tuple[int, ...]:
        """Return H_u \\ {target} for a positive train target."""
        user_index = self.user_index(
            user_id
        )

        target_index = self.item_index(
            target_item_id
        )

        history = (
            self._history_item_indices[
                user_index
            ]
        )

        if target_index not in history:
            raise ValueError(
                "Positive target is not present in "
                "the user's canonical training history"
            )

        return tuple(
            item_index
            for item_index
            in history
            if item_index
            !=
            target_index
        )

    def evaluation_context_indices(
        self,
        user_id: int,
    ) -> tuple[int, ...]:
        """Return complete H_u for validation/test candidates."""
        return self.history_indices(
            user_id
        )

    def positive_context_item_ids(
        self,
        user_id: int,
        target_item_id: int,
    ) -> tuple[int, ...]:
        indices = (
            self.positive_context_indices(
                user_id,
                target_item_id,
            )
        )

        return tuple(
            int(
                self._item_ids[
                    index
                ]
            )
            for index
            in indices
        )

    def evaluation_context_item_ids(
        self,
        user_id: int,
    ) -> tuple[int, ...]:
        indices = (
            self.evaluation_context_indices(
                user_id
            )
        )

        return tuple(
            int(
                self._item_ids[
                    index
                ]
            )
            for index
            in indices
        )

    def incidence_tensor(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return sparse user-item train-history incidence matrix.

        Shape:

            [n_users, n_items]

        Every canonical training interaction has value 1.
        """
        device = torch.device(
            device
        )

        indices = torch.as_tensor(
            np.vstack(
                [
                    self._train_user_indices,
                    self._train_item_indices,
                ]
            ),
            dtype=torch.long,
            device=device,
        )

        values = torch.ones(
            len(
                self._train_user_indices
            ),
            dtype=dtype,
            device=device,
        )

        incidence = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(
                self._metadata.n_users,
                self._metadata.n_items,
            ),
            dtype=dtype,
            device=device,
        ).coalesce()

        if incidence._nnz() != (
            self._metadata.n_interactions
        ):
            raise RuntimeError(
                "EF-EMB incidence matrix has an "
                "unexpected number of nonzero entries"
            )

        return incidence

    def aggregate_user_sums(
        self,
        context_embeddings: torch.Tensor,
        *,
        incidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute exact S_u = sum_{j in H_u} c_j.

        The sparse matrix multiplication remains differentiable with
        respect to the context embedding table.
        """
        if not isinstance(
            context_embeddings,
            torch.Tensor,
        ):
            raise TypeError(
                "context_embeddings must be a torch Tensor"
            )

        if context_embeddings.ndim != 2:
            raise ValueError(
                "context_embeddings must have shape "
                "[n_items, latent_dim]"
            )

        if context_embeddings.shape[0] != (
            self._metadata.n_items
        ):
            raise ValueError(
                "context_embeddings has the wrong "
                "number of item rows"
            )

        if not torch.is_floating_point(
            context_embeddings
        ):
            raise TypeError(
                "context_embeddings must use "
                "a floating-point dtype"
            )

        if incidence is None:
            incidence = self.incidence_tensor(
                device=(
                    context_embeddings.device
                ),
                dtype=(
                    context_embeddings.dtype
                ),
            )

        if not incidence.is_sparse:
            raise ValueError(
                "incidence must be a sparse tensor"
            )

        if incidence.shape != (
            self._metadata.n_users,
            self._metadata.n_items,
        ):
            raise ValueError(
                "incidence has invalid shape"
            )

        if incidence.device != (
            context_embeddings.device
        ):
            raise ValueError(
                "incidence and context_embeddings "
                "must be on the same device"
            )

        if incidence.dtype != (
            context_embeddings.dtype
        ):
            raise ValueError(
                "incidence and context_embeddings "
                "must use the same dtype"
            )

        sums = torch.sparse.mm(
            incidence,
            context_embeddings,
        )

        if sums.shape != (
            self._metadata.n_users,
            context_embeddings.shape[1],
        ):
            raise RuntimeError(
                "EF-EMB aggregate context sums "
                "have invalid shape"
            )

        if not torch.isfinite(
            sums
        ).all():
            raise RuntimeError(
                "EF-EMB aggregate context sums "
                "contain non-finite values"
            )

        return sums

    def context_vectors(
        self,
        *,
        context_embeddings: torch.Tensor,
        user_indices: torch.Tensor,
        target_item_indices: torch.Tensor | None,
        exclude_target: bool,
        normalization: str,
        incidence: torch.Tensor | None = None,
        precomputed_user_sums: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Construct exact normalized context vectors.

        For positive training targets:

            exclude_target = True

            context =
                H_u \\ {target}

        For legal unobserved/evaluation targets:

            exclude_target = False

            context =
                H_u

        Empty contexts return the exact zero vector.
        """
        q = normalization_exponent(
            normalization
        )

        if user_indices.ndim != 1:
            raise ValueError(
                "user_indices must be one-dimensional"
            )

        if user_indices.dtype != torch.long:
            raise TypeError(
                "user_indices must use torch.long"
            )

        if user_indices.device != (
            context_embeddings.device
        ):
            raise ValueError(
                "user_indices and context_embeddings "
                "must be on the same device"
            )

        if exclude_target:
            if target_item_indices is None:
                raise ValueError(
                    "target_item_indices is required "
                    "when exclude_target=True"
                )

            if target_item_indices.ndim != 1:
                raise ValueError(
                    "target_item_indices must be "
                    "one-dimensional"
                )

            if target_item_indices.dtype != (
                torch.long
            ):
                raise TypeError(
                    "target_item_indices must use torch.long"
                )

            if len(
                target_item_indices
            ) != len(
                user_indices
            ):
                raise ValueError(
                    "target_item_indices and user_indices "
                    "must have equal length"
                )

            if target_item_indices.device != (
                context_embeddings.device
            ):
                raise ValueError(
                    "target_item_indices and "
                    "context_embeddings must be "
                    "on the same device"
                )

        elif target_item_indices is not None:
            if len(
                target_item_indices
            ) != len(
                user_indices
            ):
                raise ValueError(
                    "target_item_indices and user_indices "
                    "must have equal length"
                )

        if precomputed_user_sums is None:
            user_sums = (
                self.aggregate_user_sums(
                    context_embeddings,
                    incidence=incidence,
                )
            )
        else:
            user_sums = (
                precomputed_user_sums
            )

            if user_sums.shape != (
                self._metadata.n_users,
                context_embeddings.shape[1],
            ):
                raise ValueError(
                    "precomputed_user_sums has invalid shape"
                )

            if user_sums.device != (
                context_embeddings.device
            ):
                raise ValueError(
                    "precomputed_user_sums is on "
                    "the wrong device"
                )

        selected_sums = (
            user_sums[
                user_indices
            ]
        )

        lengths = torch.as_tensor(
            self._history_lengths,
            dtype=(
                context_embeddings.dtype
            ),
            device=(
                context_embeddings.device
            ),
        )[
            user_indices
        ]

        if exclude_target:
            assert target_item_indices is not None

            selected_sums = (
                selected_sums
                -
                context_embeddings[
                    target_item_indices
                ]
            )

            lengths = (
                lengths
                -
                1.0
            )

        if torch.any(
            lengths < 0.0
        ):
            raise RuntimeError(
                "EF-EMB encountered a negative "
                "context length"
            )

        nonempty = (
            lengths
            >
            0.0
        )

        output = torch.zeros_like(
            selected_sums
        )

        if q == 0.0:
            output[
                nonempty
            ] = (
                selected_sums[
                    nonempty
                ]
            )

        else:
            denominators = (
                lengths[
                    nonempty
                ]
                .pow(
                    q
                )
                .unsqueeze(
                    1
                )
            )

            output[
                nonempty
            ] = (
                selected_sums[
                    nonempty
                ]
                /
                denominators
            )

        if not torch.isfinite(
            output
        ).all():
            raise RuntimeError(
                "EF-EMB context vectors contain "
                "non-finite values"
            )

        return output