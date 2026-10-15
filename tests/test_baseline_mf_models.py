"""Permanent synthetic tests for Paper 2 baselines and MF models."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    PopularityBaseline,
    RandomBaseline,
)
from src.models.bernoulli_mf import (
    BernoulliMF,
)
from src.models.gaussian_mf import (
    GaussianMF,
)
from src.models.pca_svd import (
    PCASVDBaseline,
)
from src.models.poisson_mf import (
    PoissonMF,
)
from src.sampling import (
    NegativeSampler,
)


# ============================================================
# Shared synthetic fixture
#
# 4 users
# 5 catalog items
# 8 training interactions
# 4 held-out observed interactions
#
# Every item appears in training.
# Every held-out item therefore belongs to the fitted catalog.
#
# For every user:
#   2 training positives
#   1 held-out positive
#   2 genuine legal zeros
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

    assert not all_observed.duplicated(
        subset=[
            "userId",
            "movieId",
        ]
    ).any()

    assert set(
        train["movieId"]
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
        "all_observed": all_observed,
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
# Helpers
# ============================================================


def _construct_with_supported_kwargs(
    model_class,
    **kwargs,
):
    """Construct using only kwargs supported by the implementation.

    This keeps these interface-level permanent tests focused on frozen
    semantics rather than requiring every model family to expose identical
    hyperparameter names.
    """

    signature = inspect.signature(
        model_class.__init__
    )

    supported = {
        key: value
        for key, value
        in kwargs.items()
        if key in signature.parameters
    }

    return model_class(
        **supported
    )


def _assert_finite_scores(
    model,
    *,
    user_id: int,
    catalog: np.ndarray,
) -> np.ndarray:
    """Check the frozen model-output scoring contract."""

    scores = np.asarray(
        model.score(
            user_id,
            catalog,
        ),
        dtype=np.float64,
    )

    assert scores.shape == (
        len(catalog),
    )

    assert np.isfinite(
        scores
    ).all()

    return scores


def _make_sampler(
    *,
    train: pd.DataFrame,
    all_observed: pd.DataFrame,
    namespace: str,
) -> NegativeSampler:
    return NegativeSampler(
        train=train,
        all_observed=all_observed,
        seed=2026,
        namespace=namespace,
    )


# ============================================================
# 1. Random baseline
# ============================================================


def test_random_baseline_is_deterministic_and_finite(
    synthetic_data,
) -> None:

    train = synthetic_data[
        "train"
    ]

    catalog = synthetic_data[
        "catalog"
    ]

    model = _construct_with_supported_kwargs(
        RandomBaseline,
        seed=2026,
        namespace="pytest.random",
    )

    model.fit(
        train
    )

    scores_1 = _assert_finite_scores(
        model,
        user_id=1,
        catalog=catalog,
    )

    scores_2 = _assert_finite_scores(
        model,
        user_id=1,
        catalog=catalog,
    )

    assert np.array_equal(
        scores_1,
        scores_2,
    )


def test_random_baseline_is_candidate_order_independent(
    synthetic_data,
) -> None:

    model = _construct_with_supported_kwargs(
        RandomBaseline,
        seed=2026,
        namespace="pytest.random.order",
    )

    model.fit(
        synthetic_data[
            "train"
        ]
    )

    catalog = synthetic_data[
        "catalog"
    ]

    forward_scores = model.score(
        1,
        catalog,
    )

    reversed_catalog = (
        catalog[
            ::-1
        ]
    )

    reverse_scores = model.score(
        1,
        reversed_catalog,
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
            reversed_catalog,
            reverse_scores,
        )
    }

    assert forward_map == reverse_map


# ============================================================
# 2. Popularity baseline
# ============================================================


def test_popularity_baseline_is_user_independent(
    synthetic_data,
) -> None:

    model = PopularityBaseline()

    model.fit(
        synthetic_data[
            "train"
        ]
    )

    catalog = synthetic_data[
        "catalog"
    ]

    scores_user_1 = _assert_finite_scores(
        model,
        user_id=1,
        catalog=catalog,
    )

    scores_user_4 = _assert_finite_scores(
        model,
        user_id=4,
        catalog=catalog,
    )

    assert np.array_equal(
        scores_user_1,
        scores_user_4,
    )


def test_popularity_baseline_uses_training_frequency(
    synthetic_data,
) -> None:

    train = synthetic_data[
        "train"
    ]

    model = PopularityBaseline()

    model.fit(
        train
    )

    candidates = np.asarray(
        [
            10,
            20,
        ],
        dtype=np.int64,
    )

    scores = model.score(
        1,
        candidates,
    )

    # item 20 occurs twice in TRAIN.
    # item 10 occurs once in TRAIN.
    assert scores[
        1
    ] > scores[
        0
    ]


# ============================================================
# 3. PCA / SVD baseline
# ============================================================


def _fit_pca(
    train: pd.DataFrame,
) -> PCASVDBaseline:

    model = _construct_with_supported_kwargs(
        PCASVDBaseline,
        n_components=2,
        seed=2026,
        n_iter=5,
    )

    model.fit(
        train
    )

    return model


def test_pca_svd_fits_and_returns_finite_scores(
    synthetic_data,
) -> None:

    model = _fit_pca(
        synthetic_data[
            "train"
        ]
    )

    _assert_finite_scores(
        model,
        user_id=1,
        catalog=synthetic_data[
            "catalog"
        ],
    )


def test_pca_svd_is_deterministic(
    synthetic_data,
) -> None:

    train = synthetic_data[
        "train"
    ]

    catalog = synthetic_data[
        "catalog"
    ]

    model_1 = _fit_pca(
        train
    )

    model_2 = _fit_pca(
        train
    )

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


# ============================================================
# 4. BernoulliMF
# ============================================================


def _fit_bernoulli_mf(
    synthetic_data,
    *,
    namespace: str,
) -> BernoulliMF:

    train = synthetic_data[
        "train"
    ]

    heldout = synthetic_data[
        "heldout"
    ]

    all_observed = synthetic_data[
        "all_observed"
    ]

    sampler = _make_sampler(
        train=train,
        all_observed=all_observed,
        namespace=namespace,
    )

    model = _construct_with_supported_kwargs(
        BernoulliMF,
        negative_sampler=sampler,
        heldout_interactions=heldout[
            [
                "userId",
                "movieId",
            ]
        ],
        n_factors=2,
        learning_rate=0.02,
        epochs=1,
        batch_size=16,
        negative_ratio=1,
        negative_weight=1.0,
        max_grad_norm=5.0,
        seed=2026,
        device="cpu",
    )

    model.fit(
        train
    )

    return model


def test_bernoulli_mf_fits_and_scores_finitely(
    synthetic_data,
) -> None:

    model = _fit_bernoulli_mf(
        synthetic_data,
        namespace="pytest.bernoulli",
    )

    _assert_finite_scores(
        model,
        user_id=1,
        catalog=synthetic_data[
            "catalog"
        ],
    )

    history = (
        model.training_history
    )

    assert len(
        history
    ) == 1

    assert all(
        np.isfinite(
            float(value)
        )
        for key, value
        in history[
            0
        ].items()
        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    )


def test_bernoulli_mf_is_deterministic(
    synthetic_data,
) -> None:

    model_1 = _fit_bernoulli_mf(
        synthetic_data,
        namespace="pytest.bernoulli.det",
    )

    model_2 = _fit_bernoulli_mf(
        synthetic_data,
        namespace="pytest.bernoulli.det",
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
# 5. PoissonMF
# ============================================================


def _fit_poisson_mf(
    synthetic_data,
    *,
    namespace: str,
) -> PoissonMF:

    train = synthetic_data[
        "train"
    ]

    heldout = synthetic_data[
        "heldout"
    ]

    all_observed = synthetic_data[
        "all_observed"
    ]

    sampler = _make_sampler(
        train=train,
        all_observed=all_observed,
        namespace=namespace,
    )

    model = _construct_with_supported_kwargs(
        PoissonMF,

        # IMPORTANT:
        # PoissonMF uses the shared NegativeSampler under the
        # constructor argument name "negative_sampler".
        #
        # The previous test incorrectly used zero_sampler here,
        # which was silently filtered out by
        # _construct_with_supported_kwargs().
        negative_sampler=sampler,

        heldout_interactions=heldout[
            [
                "userId",
                "movieId",
            ]
        ],
        n_factors=2,
        learning_rate=0.02,
        epochs=1,
        batch_size=16,
        zero_ratio=1,
        zero_sampling_strategy="uniform",
        zero_replace=True,
        log_rate_clip=20.0,
        max_grad_norm=5.0,
        seed=2026,
        device="cpu",
    )

    model.fit(
        train
    )

    return model


def test_poisson_mf_fits_with_masked_zero_objective(
    synthetic_data,
) -> None:

    model = _fit_poisson_mf(
        synthetic_data,
        namespace="pytest.poisson",
    )

    scores = _assert_finite_scores(
        model,
        user_id=1,
        catalog=synthetic_data[
            "catalog"
        ],
    )

    assert callable(
        getattr(
            model,
            "rate",
            None,
        )
    )

    rates = np.asarray(
        model.rate(
            1,
            synthetic_data[
                "catalog"
            ],
        ),
        dtype=np.float64,
    )

    assert rates.shape == (
        len(
            scores
        ),
    )

    assert np.isfinite(
        rates
    ).all()

    assert np.all(
        rates > 0.0
    )


def test_poisson_mf_masks_heldout_from_zero_population(
    synthetic_data,
) -> None:

    model = _fit_poisson_mf(
        synthetic_data,
        namespace="pytest.poisson.mask",
    )

    # 4 users * 5 items = 20 total cells.
    #
    # 4 held-out positives are masked.
    #
    # Therefore:
    #   16 legal training-likelihood cells
    #   = 8 train positives + 8 genuine zeros.
    if hasattr(
        model,
        "legal_cells",
    ):
        assert model.legal_cells == 16

    if hasattr(
        model,
        "zero_population_cells",
    ):
        assert (
            model.zero_population_cells
            ==
            8
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

    if "heldout_masked_cells" in record:
        assert record[
            "heldout_masked_cells"
        ] == 4

    if "zero_population_cells" in record:
        assert record[
            "zero_population_cells"
        ] == 8

    if "legal_cells" in record:
        assert record[
            "legal_cells"
        ] == 16

def test_poisson_mf_b91_importance_weights_are_nontrivial_and_exact(
) -> None:
    """Lock the B91 masked-full Poisson importance-weight estimator.

    This fixture deliberately makes |Z_u| / m_u != 1.

    There are:
      4 users
      4 catalog items
      1 train positive per user
      0 held-out positives

    Therefore, for every user:

      |Z_u| = 3
      m_u = 1
      raw zero weight = 3

    Across the complete stochastic epoch:

      4 positive cells, raw weight 1 each
      4 sampled zero cells, raw weight 3 each

      raw weight sum = 4 + 12 = 16

    The full legal matrix also contains:

      4 * 4 = 16 cells

    With N=8 stochastic examples and M=16 legal cells,
    the normalization factor is N/M = 0.5.

    Hence final sample weights must be:

      positives: 1 * 0.5 = 0.5
      zeros:     3 * 0.5 = 1.5
    """

    train = pd.DataFrame(
        {
            "userId": [
                1,
                2,
                3,
                4,
            ],
            "movieId": [
                10,
                20,
                30,
                40,
            ],
            "rating": [
                5.0,
                4.0,
                3.0,
                2.0,
            ],
        }
    )

    heldout = pd.DataFrame(
        {
            "userId": pd.Series(
                dtype="int64"
            ),
            "movieId": pd.Series(
                dtype="int64"
            ),
            "rating": pd.Series(
                dtype="float64"
            ),
        }
    )

    data = {
        "train": train,
        "heldout": heldout,
        "all_observed": train.copy(),
        "catalog": np.asarray(
            [
                10,
                20,
                30,
                40,
            ],
            dtype=np.int64,
        ),
    }

    model = _fit_poisson_mf(
        data,
        namespace="pytest.poisson.b91.weights",
    )

    assert model._user_ids is not None
    assert model._user_to_index is not None
    assert model._item_to_index is not None

    frame = (
        train[
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

    positive_user_ids = (
        frame[
            "userId"
        ]
        .to_numpy(
            dtype=np.int64,
        )
    )

    positive_user_indices = np.fromiter(
        (
            model._user_to_index[
                int(user_id)
            ]
            for user_id
            in positive_user_ids
        ),
        dtype=np.int64,
        count=len(
            frame
        ),
    )

    positive_item_indices = np.fromiter(
        (
            model._item_to_index[
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
        count=len(
            frame
        ),
    )

    zero_sizes = (
        model._calculate_legal_zero_sizes(
            model._user_ids
        )
    )

    assert zero_sizes == {
        1: 3,
        2: 3,
        3: 3,
        4: 3,
    }

    full_legal_cells = (
        len(
            frame
        )
        +
        sum(
            zero_sizes.values()
        )
    )

    assert full_legal_cells == 16

    (
        epoch_users,
        epoch_items,
        epoch_targets,
        sample_weights,
        stats,
    ) = model._build_epoch_examples(
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
        epoch=123,
    )

    assert len(
        epoch_users
    ) == 8

    assert len(
        epoch_items
    ) == 8

    assert len(
        epoch_targets
    ) == 8

    assert len(
        sample_weights
    ) == 8

    # _build_epoch_examples concatenates positives first,
    # followed by sampled legal zeros.
    assert np.array_equal(
        epoch_targets[
            :4
        ],
        np.ones(
            4,
            dtype=np.float32,
        ),
    )

    assert np.array_equal(
        epoch_targets[
            4:
        ],
        np.zeros(
            4,
            dtype=np.float32,
        ),
    )

    # N / M = 8 / 16 = 0.5.
    #
    # Positive raw weight:
    #   1 * 0.5 = 0.5
    #
    # Zero raw weight:
    #   (3 / 1) * 0.5 = 1.5
    assert np.allclose(
        sample_weights[
            :4
        ],
        0.5,
        rtol=0.0,
        atol=1e-12,
    )

    assert np.allclose(
        sample_weights[
            4:
        ],
        1.5,
        rtol=0.0,
        atol=1e-12,
    )

    scale = (
        float(
            len(
                sample_weights
            )
        )
        /
        float(
            full_legal_cells
        )
    )

    assert np.isclose(
        scale,
        0.5,
        rtol=0.0,
        atol=1e-12,
    )

    reconstructed_raw_weights = (
        sample_weights
        /
        scale
    )

    assert np.allclose(
        reconstructed_raw_weights[
            :4
        ],
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert np.allclose(
        reconstructed_raw_weights[
            4:
        ],
        3.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert np.isclose(
        reconstructed_raw_weights.sum(
            dtype=np.float64
        ),
        16.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert stats[
        "positive_examples"
    ] == 4

    assert stats[
        "zero_draws"
    ] == 4

    assert stats[
        "stochastic_examples"
    ] == 8

    assert stats[
        "full_legal_cells"
    ] == 16

    assert np.isclose(
        float(
            stats[
                "raw_weight_sum"
            ]
        ),
        16.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert np.isclose(
        float(
            stats[
                "sample_weight_mean"
            ]
        ),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

def test_poisson_mf_is_deterministic(
    synthetic_data,
) -> None:

    model_1 = _fit_poisson_mf(
        synthetic_data,
        namespace="pytest.poisson.det",
    )

    model_2 = _fit_poisson_mf(
        synthetic_data,
        namespace="pytest.poisson.det",
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
# 6. GaussianMF
# ============================================================


def _fit_gaussian_mf(
    synthetic_data,
) -> GaussianMF:

    model = _construct_with_supported_kwargs(
        GaussianMF,
        n_factors=2,
        learning_rate=0.02,
        epochs=1,
        batch_size=16,
        max_grad_norm=5.0,
        seed=2026,
        device="cpu",
    )

    model.fit(
        synthetic_data[
            "train"
        ]
    )

    return model


def test_gaussian_mf_fits_and_scores_finitely(
    synthetic_data,
) -> None:

    model = _fit_gaussian_mf(
        synthetic_data
    )

    _assert_finite_scores(
        model,
        user_id=1,
        catalog=synthetic_data[
            "catalog"
        ],
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

    for key in [
        "mse",
        "rmse",
        "mae",
    ]:
        if key in record:
            assert np.isfinite(
                float(
                    record[
                        key
                    ]
                )
            )


def test_gaussian_mf_native_metrics_are_finite(
    synthetic_data,
) -> None:

    model = _fit_gaussian_mf(
        synthetic_data
    )

    assert callable(
        getattr(
            model,
            "native_metrics",
            None,
        )
    )

    metrics = (
        model.native_metrics(
            synthetic_data[
                "heldout"
            ]
        )
    )

    assert set(
        metrics
    ) == {
        "rmse",
        "mae",
    }

    assert np.isfinite(
        metrics[
            "rmse"
        ]
    )

    assert np.isfinite(
        metrics[
            "mae"
        ]
    )

    assert metrics[
        "rmse"
    ] >= 0.0

    assert metrics[
        "mae"
    ] >= 0.0


def test_gaussian_mf_is_deterministic(
    synthetic_data,
) -> None:

    model_1 = _fit_gaussian_mf(
        synthetic_data
    )

    model_2 = _fit_gaussian_mf(
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