"""Permanent synthetic tests for Paper 2 literature-based models.

Models covered
--------------
- LogisticPCA
- Hierarchical Poisson Factorization (HPF)

These tests permanently protect the corrected model semantics frozen
in model specification revision v1.2.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.models.hpf import (
    HPF,
)
from src.models.logistic_pca import (
    LogisticPCA,
)


# ============================================================
# Shared synthetic fixture
#
# 4 users
# 5 items
# 8 training positives
# 4 held-out positives
#
# Total matrix:
#     4 * 5 = 20 cells
#
# Masked held-out:
#     4 cells
#
# Legal training likelihood:
#     16 cells
#
# Within legal cells:
#     8 positive
#     8 zero
#
# Every item appears in training.
# ============================================================


@pytest.fixture
def synthetic_data():
    train = pd.DataFrame(
        {
            "userId": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
            ],
            "movieId": [
                10,
                20,
                20,
                30,
                30,
                40,
                40,
                50,
            ],
            "rating": [
                5.0,
                4.0,
                4.0,
                3.5,
                4.5,
                2.5,
                3.0,
                5.0,
            ],
        }
    )

    heldout = pd.DataFrame(
        {
            "userId": [
                1,
                2,
                3,
                4,
            ],
            "movieId": [
                30,
                40,
                50,
                10,
            ],
            "rating": [
                4.0,
                2.0,
                4.5,
                3.5,
            ],
        }
    )

    all_observed = pd.concat(
        [
            train,
            heldout,
        ],
        ignore_index=True,
    )

    assert len(
        train
    ) == 8

    assert len(
        heldout
    ) == 4

    assert not all_observed.duplicated(
        subset=[
            "userId",
            "movieId",
        ]
    ).any()

    assert train[
        "userId"
    ].nunique() == 4

    assert train[
        "movieId"
    ].nunique() == 5

    assert set(
        train[
            "movieId"
        ]
    ) == {
        10,
        20,
        30,
        40,
        50,
    }

    return {
        "train": train,
        "heldout": heldout,
        "heldout_ids": heldout[
            [
                "userId",
                "movieId",
            ]
        ].copy(),
        "catalog": np.asarray(
            [
                10,
                20,
                30,
                40,
                50,
            ],
            dtype=np.int64,
        ),
    }


# ============================================================
# Shared helper
# ============================================================


def _assert_finite_scores(
    model,
    *,
    user_id: int,
    catalog: np.ndarray,
) -> np.ndarray:
    """Enforce common Paper 2 scoring contract."""

    scores = np.asarray(
        model.score(
            user_id,
            catalog,
        ),
        dtype=np.float64,
    )

    assert scores.shape == (
        len(
            catalog
        ),
    )

    assert np.isfinite(
        scores
    ).all()

    return scores


# ============================================================
# LogisticPCA helper
# ============================================================


def _fit_logistic_pca(
    synthetic_data,
) -> LogisticPCA:

    model = LogisticPCA(
        heldout_interactions=synthetic_data[
            "heldout_ids"
        ],
        n_factors=2,
        learning_rate=0.01,
        epochs=1,
        user_batch_size=2,
        max_grad_norm=5.0,
        intercept_epsilon=1e-6,
        seed=2026,
        device="cpu",
    )

    model.fit(
        synthetic_data[
            "train"
        ]
    )

    return model


# ============================================================
# 1. LogisticPCA API / B93 semantics
# ============================================================


def test_logistic_pca_constructor_has_no_negative_sampler() -> None:
    """LogisticPCA must not regress into sampled-BCE BernoulliMF."""

    signature = inspect.signature(
        LogisticPCA.__init__
    )

    forbidden = {
        "negative_sampler",
        "negative_ratio",
        "negative_weight",
        "zero_sampler",
        "zero_ratio",
    }

    assert forbidden.isdisjoint(
        signature.parameters
    )


def test_logistic_pca_constructor_has_no_b93_legacy_constraints() -> None:
    """B93 forbids the old artificial A constraints/regularizers."""

    signature = inspect.signature(
        LogisticPCA.__init__
    )

    forbidden = {
        "orthogonality_penalty",
        "orthogonalize",
        "centering_penalty",
        "factor_l2",
        "user_l2",
        "item_l2",
        "regularization",
    }

    assert forbidden.isdisjoint(
        signature.parameters
    )


# ============================================================
# 2. LogisticPCA exact masked-cell accounting
# ============================================================


def test_logistic_pca_uses_full_masked_legal_matrix(
    synthetic_data,
) -> None:

    model = _fit_logistic_pca(
        synthetic_data
    )

    assert model.is_fitted

    # 4 users * 5 items = 20
    # 4 held-out positives masked
    # 16 legal cells remain.
    assert (
        model.legal_cell_count
        ==
        16
    )

    assert (
        model.positive_cell_count
        ==
        8
    )

    assert (
        model.zero_cell_count
        ==
        8
    )

    assert (
        model.heldout_cell_count
        ==
        4
    )

    assert (
        model.positive_cell_count
        +
        model.zero_cell_count
        ==
        model.legal_cell_count
    )

    assert (
        model.legal_cell_count
        +
        model.heldout_cell_count
        ==
        20
    )


def test_logistic_pca_training_history_matches_mask(
    synthetic_data,
) -> None:

    model = _fit_logistic_pca(
        synthetic_data
    )

    history = (
        model.training_history
    )

    assert len(
        history
    ) == 1

    record = history[
        0
    ]

    if "legal_cells" in record:
        assert (
            record[
                "legal_cells"
            ]
            ==
            16
        )

    if "positive_cells" in record:
        assert (
            record[
                "positive_cells"
            ]
            ==
            8
        )

    if "zero_cells" in record:
        assert (
            record[
                "zero_cells"
            ]
            ==
            8
        )

    if "heldout_masked_cells" in record:
        assert (
            record[
                "heldout_masked_cells"
            ]
            ==
            4
        )

    for key, value in record.items():
        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            assert np.isfinite(
                float(
                    value
                )
            )


# ============================================================
# 3. LogisticPCA finite logits / probabilities
# ============================================================


def test_logistic_pca_scores_and_probabilities_are_finite(
    synthetic_data,
) -> None:

    model = _fit_logistic_pca(
        synthetic_data
    )

    catalog = synthetic_data[
        "catalog"
    ]

    scores = _assert_finite_scores(
        model,
        user_id=1,
        catalog=catalog,
    )

    probabilities = np.asarray(
        model.probability(
            1,
            catalog,
        ),
        dtype=np.float64,
    )

    assert probabilities.shape == (
        len(
            catalog
        ),
    )

    assert np.isfinite(
        probabilities
    ).all()

    assert np.all(
        probabilities > 0.0
    )

    assert np.all(
        probabilities < 1.0
    )

    expected_probabilities = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -scores
            )
        )
    )

    assert np.allclose(
        probabilities,
        expected_probabilities,
        rtol=1e-10,
        atol=1e-10,
    )


# ============================================================
# 4. LogisticPCA candidate-order invariance
# ============================================================


def test_logistic_pca_candidate_order_independent(
    synthetic_data,
) -> None:

    model = _fit_logistic_pca(
        synthetic_data
    )

    catalog = synthetic_data[
        "catalog"
    ]

    forward_scores = model.score(
        1,
        catalog,
    )

    reverse_catalog = (
        catalog[
            ::-1
        ]
    )

    reverse_scores = model.score(
        1,
        reverse_catalog,
    )

    forward_map = {
        int(item): float(score)
        for item, score
        in zip(
            catalog,
            forward_scores,
        )
    }

    reverse_map = {
        int(item): float(score)
        for item, score
        in zip(
            reverse_catalog,
            reverse_scores,
        )
    }

    for item_id in forward_map:
        assert np.isclose(
            forward_map[
                item_id
            ],
            reverse_map[
                item_id
            ],
            rtol=0.0,
            atol=1e-7,
        )


# ============================================================
# 5. LogisticPCA determinism
# ============================================================


def test_logistic_pca_is_deterministic(
    synthetic_data,
) -> None:

    model_1 = _fit_logistic_pca(
        synthetic_data
    )

    model_2 = _fit_logistic_pca(
        synthetic_data
    )

    catalog = synthetic_data[
        "catalog"
    ]

    scores_1 = model_1.score(
        1,
        catalog,
    )

    scores_2 = model_2.score(
        1,
        catalog,
    )

    assert np.allclose(
        scores_1,
        scores_2,
        rtol=0.0,
        atol=1e-7,
    )


# ============================================================
# 6. LogisticPCA input-row-order invariance
# ============================================================


def test_logistic_pca_input_order_invariant(
    synthetic_data,
) -> None:

    original = _fit_logistic_pca(
        synthetic_data
    )

    shuffled_train = (
        synthetic_data[
            "train"
        ]
        .sample(
            frac=1.0,
            random_state=12345,
        )
        .reset_index(
            drop=True
        )
    )

    shuffled_heldout = (
        synthetic_data[
            "heldout_ids"
        ]
        .sample(
            frac=1.0,
            random_state=54321,
        )
        .reset_index(
            drop=True
        )
    )

    shuffled_model = LogisticPCA(
        heldout_interactions=(
            shuffled_heldout
        ),
        n_factors=2,
        learning_rate=0.01,
        epochs=1,
        user_batch_size=2,
        max_grad_norm=5.0,
        intercept_epsilon=1e-6,
        seed=2026,
        device="cpu",
    )

    shuffled_model.fit(
        shuffled_train
    )

    catalog = synthetic_data[
        "catalog"
    ]

    scores_original = (
        original.score(
            1,
            catalog,
        )
    )

    scores_shuffled = (
        shuffled_model.score(
            1,
            catalog,
        )
    )

    assert np.allclose(
        scores_original,
        scores_shuffled,
        rtol=0.0,
        atol=1e-7,
    )


# ============================================================
# 7. LogisticPCA save/load
# ============================================================


def test_logistic_pca_save_load_roundtrip(
    synthetic_data,
) -> None:

    model = _fit_logistic_pca(
        synthetic_data
    )

    catalog = synthetic_data[
        "catalog"
    ]

    expected = model.score(
        1,
        catalog,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = (
            Path(
                tmp
            )
            /
            "logistic_pca.npz"
        )

        model.save(
            path
        )

        loaded = (
            LogisticPCA.load(
                path
            )
        )

        actual = loaded.score(
            1,
            catalog,
        )

    assert np.allclose(
        expected,
        actual,
        rtol=0.0,
        atol=1e-7,
    )


# ============================================================
# HPF helper
# ============================================================


def _fit_hpf(
    synthetic_data,
) -> HPF:

    model = HPF(
        heldout_interactions=synthetic_data[
            "heldout_ids"
        ],
        n_factors=2,
        max_iterations=3,
        tolerance=1e-12,
        a0=0.3,
        a=0.3,
        b0=1.0,
        c0=0.3,
        c=0.3,
        d0=1.0,
        initialization_scale=0.1,
        seed=2026,
    )

    model.fit(
        synthetic_data[
            "train"
        ]
    )

    return model


# ============================================================
# 8. HPF API semantics
# ============================================================


def test_hpf_has_no_negative_sampler_api() -> None:
    """HPF is Gamma-Poisson inference, not sampled-negative MF."""

    signature = inspect.signature(
        HPF.__init__
    )

    forbidden = {
        "negative_sampler",
        "negative_ratio",
        "negative_weight",
        "zero_sampler",
        "zero_ratio",
    }

    assert forbidden.isdisjoint(
        signature.parameters
    )


def test_hpf_literature_prior_defaults() -> None:
    """Permanent protection for frozen literature-aligned priors."""

    signature = inspect.signature(
        HPF.__init__
    )

    assert (
        signature.parameters[
            "a0"
        ].default
        ==
        0.3
    )

    assert (
        signature.parameters[
            "a"
        ].default
        ==
        0.3
    )

    assert (
        signature.parameters[
            "b0"
        ].default
        ==
        1.0
    )

    assert (
        signature.parameters[
            "c0"
        ].default
        ==
        0.3
    )

    assert (
        signature.parameters[
            "c"
        ].default
        ==
        0.3
    )

    assert (
        signature.parameters[
            "d0"
        ].default
        ==
        1.0
    )


# ============================================================
# 9. HPF fit / held-out mask
# ============================================================


def test_hpf_fits_and_masks_heldout_cells(
    synthetic_data,
) -> None:

    model = _fit_hpf(
        synthetic_data
    )

    assert model.is_fitted

    assert (
        model.heldout_cell_count
        ==
        4
    )

    assert len(
        model.user_ids
    ) == 4

    assert len(
        model.item_catalog
    ) == 5


# ============================================================
# 10. HPF posterior expected factors
# ============================================================


def test_hpf_expected_factors_positive_and_finite(
    synthetic_data,
) -> None:

    model = _fit_hpf(
        synthetic_data
    )

    user_factors = np.asarray(
        model.expected_user_factors,
        dtype=np.float64,
    )

    item_factors = np.asarray(
        model.expected_item_factors,
        dtype=np.float64,
    )

    assert user_factors.shape == (
        4,
        2,
    )

    assert item_factors.shape == (
        5,
        2,
    )

    assert np.isfinite(
        user_factors
    ).all()

    assert np.isfinite(
        item_factors
    ).all()

    assert np.all(
        user_factors > 0.0
    )

    assert np.all(
        item_factors > 0.0
    )


# ============================================================
# 11. HPF score = posterior-mean factor dot product
# ============================================================


def test_hpf_scoring_matches_expected_factor_dot_product(
    synthetic_data,
) -> None:

    model = _fit_hpf(
        synthetic_data
    )

    catalog = synthetic_data[
        "catalog"
    ]

    scores = _assert_finite_scores(
        model,
        user_id=1,
        catalog=catalog,
    )

    assert np.all(
        scores > 0.0
    )

    user_ids = np.asarray(
        model.user_ids,
        dtype=np.int64,
    )

    item_ids = np.asarray(
        model.item_catalog,
        dtype=np.int64,
    )

    user_index = int(
        np.where(
            user_ids == 1
        )[0][0]
    )

    item_indices = np.asarray(
        [
            int(
                np.where(
                    item_ids
                    ==
                    item_id
                )[0][0]
            )
            for item_id
            in catalog
        ],
        dtype=np.int64,
    )

    user_factors = np.asarray(
        model.expected_user_factors,
        dtype=np.float64,
    )

    item_factors = np.asarray(
        model.expected_item_factors,
        dtype=np.float64,
    )

    manual_scores = (
        item_factors[
            item_indices
        ]
        @
        user_factors[
            user_index
        ]
    )

    assert np.allclose(
        scores,
        manual_scores,
        rtol=1e-10,
        atol=1e-10,
    )


# ============================================================
# 12. HPF training history
# ============================================================


def test_hpf_training_history_is_finite(
    synthetic_data,
) -> None:

    model = _fit_hpf(
        synthetic_data
    )

    history = (
        model.training_history
    )

    assert 1 <= len(
        history
    ) <= 3

    for record in history:
        for key, value in record.items():

            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                assert np.isfinite(
                    float(
                        value
                    )
                )

    assert isinstance(
        model.converged,
        bool,
    )


# ============================================================
# 13. HPF candidate-order invariance
# ============================================================


def test_hpf_candidate_order_independent(
    synthetic_data,
) -> None:

    model = _fit_hpf(
        synthetic_data
    )

    catalog = synthetic_data[
        "catalog"
    ]

    forward = model.score(
        1,
        catalog,
    )

    reversed_catalog = (
        catalog[
            ::-1
        ]
    )

    backward = model.score(
        1,
        reversed_catalog,
    )

    forward_map = {
        int(item): float(score)
        for item, score
        in zip(
            catalog,
            forward,
        )
    }

    backward_map = {
        int(item): float(score)
        for item, score
        in zip(
            reversed_catalog,
            backward,
        )
    }

    for item_id in forward_map:
        assert np.isclose(
            forward_map[
                item_id
            ],
            backward_map[
                item_id
            ],
            rtol=0.0,
            atol=1e-10,
        )


# ============================================================
# 14. HPF determinism
# ============================================================


def test_hpf_is_deterministic(
    synthetic_data,
) -> None:

    model_1 = _fit_hpf(
        synthetic_data
    )

    model_2 = _fit_hpf(
        synthetic_data
    )

    catalog = synthetic_data[
        "catalog"
    ]

    scores_1 = model_1.score(
        1,
        catalog,
    )

    scores_2 = model_2.score(
        1,
        catalog,
    )

    assert np.allclose(
        scores_1,
        scores_2,
        rtol=0.0,
        atol=1e-10,
    )

    assert np.allclose(
        model_1.expected_user_factors,
        model_2.expected_user_factors,
        rtol=0.0,
        atol=1e-10,
    )

    assert np.allclose(
        model_1.expected_item_factors,
        model_2.expected_item_factors,
        rtol=0.0,
        atol=1e-10,
    )


# ============================================================
# 15. HPF input-row-order invariance
# ============================================================


def test_hpf_input_order_invariant(
    synthetic_data,
) -> None:

    original = _fit_hpf(
        synthetic_data
    )

    shuffled_train = (
        synthetic_data[
            "train"
        ]
        .sample(
            frac=1.0,
            random_state=12345,
        )
        .reset_index(
            drop=True
        )
    )

    shuffled_heldout = (
        synthetic_data[
            "heldout_ids"
        ]
        .sample(
            frac=1.0,
            random_state=54321,
        )
        .reset_index(
            drop=True
        )
    )

    shuffled_model = HPF(
        heldout_interactions=(
            shuffled_heldout
        ),
        n_factors=2,
        max_iterations=3,
        tolerance=1e-12,
        a0=0.3,
        a=0.3,
        b0=1.0,
        c0=0.3,
        c=0.3,
        d0=1.0,
        initialization_scale=0.1,
        seed=2026,
    )

    shuffled_model.fit(
        shuffled_train
    )

    catalog = synthetic_data[
        "catalog"
    ]

    original_scores = (
        original.score(
            1,
            catalog,
        )
    )

    shuffled_scores = (
        shuffled_model.score(
            1,
            catalog,
        )
    )

    assert np.allclose(
        original_scores,
        shuffled_scores,
        rtol=0.0,
        atol=1e-10,
    )


# ============================================================
# 16. HPF save/load
# ============================================================


def test_hpf_save_load_roundtrip(
    synthetic_data,
) -> None:

    model = _fit_hpf(
        synthetic_data
    )

    catalog = synthetic_data[
        "catalog"
    ]

    expected = model.score(
        1,
        catalog,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = (
            Path(
                tmp
            )
            /
            "hpf.npz"
        )

        model.save(
            path
        )

        loaded = HPF.load(
            path
        )

        actual = loaded.score(
            1,
            catalog,
        )

    assert np.allclose(
        expected,
        actual,
        rtol=0.0,
        atol=1e-10,
    )


# ============================================================
# 17. Literature models are genuinely different
# ============================================================


def test_logistic_pca_and_hpf_are_not_same_scoring_model(
    synthetic_data,
) -> None:

    logistic = _fit_logistic_pca(
        synthetic_data
    )

    hpf = _fit_hpf(
        synthetic_data
    )

    catalog = synthetic_data[
        "catalog"
    ]

    logistic_scores = np.asarray(
        logistic.score(
            1,
            catalog,
        ),
        dtype=np.float64,
    )

    hpf_scores = np.asarray(
        hpf.score(
            1,
            catalog,
        ),
        dtype=np.float64,
    )

    # They represent fundamentally different likelihoods and
    # parameterizations. They must not collapse to identical scores.
    assert not np.allclose(
        logistic_scores,
        hpf_scores,
        rtol=1e-7,
        atol=1e-7,
    )