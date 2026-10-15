"""Permanent synthetic tests for the five Paper 2 EF-EMB models."""



from __future__ import annotations



import inspect

import math



import numpy as np

import pandas as pd

import pytest

import torch

from torch.nn import functional as F



from src.models.ap_emb import AdditivePoissonEFEmb

from src.models.b_emb import BernoulliEFEmb

from src.models.efemb import (

    EFContextStore,

    NORMALIZATION_EXPONENTS,

    normalization_exponent,

)

from src.models.g_emb import GaussianEFEmb

from src.models.ng_emb import NonnegativeGaussianEFEmb

from src.models.p_emb import PoissonEFEmb

from src.sampling import NegativeSampler





MODEL_CLASSES = {

    "b-emb": BernoulliEFEmb,

    "p-emb": PoissonEFEmb,

    "ap-emb": AdditivePoissonEFEmb,

    "g-emb": GaussianEFEmb,

    "ng-emb": NonnegativeGaussianEFEmb,

}





# ============================================================

# Shared synthetic fixture

#

# 4 users

# 5 catalog items

# 8 training interactions

# 4 held-out interactions

#

# Every user:

#   2 train positives

#   1 held-out positive

#   2 genuinely unobserved items

#

# Total user-item cells = 4 * 5 = 20

# Held-out masked cells = 4

# Legal train objective cells = 16

#   8 positives + 8 genuine zeros

# ============================================================





@pytest.fixture(scope="module")

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



    assert len(train) == 8

    assert len(heldout) == 4



    assert (

        train["userId"].nunique()

        ==

        4

    )



    assert (

        train["movieId"].nunique()

        ==

        5

    )



    assert not all_observed.duplicated(

        subset=[

            "userId",

            "movieId",

        ]

    ).any()



    return {

        "train": train,

        "heldout": heldout,

        "heldout_ids": heldout[

            [

                "userId",

                "movieId",

            ]

        ].copy(),

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





def _make_sampler(

    train,

    all_observed,

    namespace,

):

    return NegativeSampler(

        train=train,

        all_observed=all_observed,

        seed=2026,

        namespace=namespace,

    )





def _fit_model(

    model_name,

    data,

    *,

    shuffled=False,

):

    """Fit one tiny EF-EMB model using frozen dev semantics."""



    train = data[

        "train"

    ]



    heldout = data[

        "heldout"

    ]



    if shuffled:

        train = (

            train.sample(

                frac=1.0,

                random_state=12345,

            )

            .reset_index(

                drop=True

            )

        )



        heldout = (

            heldout.sample(

                frac=1.0,

                random_state=54321,

            )

            .reset_index(

                drop=True

            )

        )



    heldout_ids = heldout[

        [

            "userId",

            "movieId",

        ]

    ].copy()



    all_observed = pd.concat(

        [

            train,

            heldout,

        ],

        ignore_index=True,

    )



    common = dict(

        heldout_interactions=heldout_ids,

        n_factors=2,

        learning_rate=0.02,

        epochs=1,

        batch_size=16,

        context_normalization="mean",

        target_l2=1e-6,

        context_l2=1e-6,

        bias_l2=1e-6,

        max_grad_norm=5.0,

        seed=2026,

        device="cpu",

    )



    if model_name == "b-emb":



        model = BernoulliEFEmb(

            negative_sampler=_make_sampler(

                train,

                all_observed,

                "pytest.efemb.b-emb",

            ),

            negative_ratio=1,

            negative_weight=1.0,

            negative_strategy="uniform",

            negative_replace=True,

            **common,

        )



    elif model_name == "p-emb":



        model = PoissonEFEmb(

            zero_sampler=_make_sampler(

                train,

                all_observed,

                "pytest.efemb.p-emb",

            ),

            zero_ratio=1,

            zero_sampling_strategy="uniform",

            zero_replace=True,

            log_rate_clip=20.0,

            **common,

        )



    elif model_name == "ap-emb":



        model = AdditivePoissonEFEmb(

            zero_sampler=_make_sampler(

                train,

                all_observed,

                "pytest.efemb.ap-emb",

            ),

            zero_ratio=1,

            zero_sampling_strategy="uniform",

            zero_replace=True,

            initial_interaction_rate=0.01,

            **common,

        )



    elif model_name == "g-emb":



        model = GaussianEFEmb(

            **common,

        )



    elif model_name == "ng-emb":



        model = NonnegativeGaussianEFEmb(

            **common,

        )



    else:

        raise AssertionError(

            f"Unknown model: {model_name}"

        )



    model.fit(

        train

    )



    return model





@pytest.fixture(scope="module")

def fitted_models(

    synthetic_data,

):

    return {

        name: _fit_model(

            name,

            synthetic_data,

        )

        for name

        in MODEL_CLASSES

    }





def _finite_scores(

    model,

    user_id,

    catalog,

):

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





def _embedding_parameter_names(

    model,

):

    names = [

        name

        for name, _

        in model._module.named_parameters()

    ]



    target = [

        name

        for name

        in names

        if (

            "target"

            in name.lower()

            and

            "embedding"

            in name.lower()

        )

    ]



    context = [

        name

        for name

        in names

        if (

            "context"

            in name.lower()

            and

            "embedding"

            in name.lower()

        )

    ]



    assert len(

        target

    ) == 1



    assert len(

        context

    ) == 1



    return (

        target[

            0

        ],

        context[

            0

        ],

    )





# ============================================================

# Shared EF-EMB framework

# ============================================================





def test_normalization_registry_is_frozen():

    assert NORMALIZATION_EXPONENTS == {

        "raw": 0.0,

        "sqrt": 0.5,

        "mean": 1.0,

    }



    assert (

        normalization_exponent(

            "raw"

        )

        ==

        0.0

    )



    assert (

        normalization_exponent(

            "sqrt"

        )

        ==

        0.5

    )



    assert (

        normalization_exponent(

            "mean"

        )

        ==

        1.0

    )





def test_empty_context_is_zero_for_all_normalizations():

    train = pd.DataFrame(

        {

            "userId": [

                1,

                2,

            ],

            "movieId": [

                10,

                20,

            ],

        }

    )



    store = EFContextStore(

        train,

        heldout_interactions=None,

        max_context_items=None,

    )



    embeddings = torch.tensor(

        [

            [

                2.0,

                4.0,

            ],

            [

                6.0,

                8.0,

            ],

        ],

        dtype=torch.float32,

    )



    incidence = (

        store.incidence_tensor(

            device="cpu",

            dtype=torch.float32,

        )

    )



    user_sums = (

        store.aggregate_user_sums(

            embeddings,

            incidence=incidence,

        )

    )



    user_indices = torch.tensor(

        [

            store.user_index(

                1

            )

        ],

        dtype=torch.long,

    )



    target_indices = torch.tensor(

        [

            store.item_index(

                10

            )

        ],

        dtype=torch.long,

    )



    for mode in [

        "raw",

        "sqrt",

        "mean",

    ]:



        context = (

            store.context_vectors(

                context_embeddings=embeddings,

                user_indices=user_indices,

                target_item_indices=target_indices,

                exclude_target=True,

                normalization=mode,

                incidence=incidence,

                precomputed_user_sums=user_sums,

            )[0]

        )



        assert torch.allclose(

            context,

            torch.zeros(

                2,

                dtype=torch.float32,

            ),

            rtol=0.0,

            atol=0.0,

        )





@pytest.mark.parametrize(

    "model_name",

    MODEL_CLASSES,

)

def test_all_efemb_models_fit_and_score_finitely(

    model_name,

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        model_name

    ]



    assert model.is_fitted



    assert (

        model.context_normalization

        ==

        "mean"

    )



    _finite_scores(

        model,

        1,

        synthetic_data[

            "catalog"

        ],

    )





@pytest.mark.parametrize(

    "model_name",

    MODEL_CLASSES,

)

def test_train_contexts_never_contain_heldout_items(

    model_name,

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        model_name

    ]



    assert (

        model._context_store

        is not None

    )



    for row in (

        synthetic_data[

            "heldout_ids"

        ]

        .itertuples(

            index=False

        )

    ):



        history = set(

            int(item)

            for item

            in model._context_store

            .history_item_ids(

                int(

                    row.userId

                )

            )

        )



        assert (

            int(

                row.movieId

            )

            not in history

        )





@pytest.mark.parametrize(

    "model_name",

    MODEL_CLASSES,

)

def test_target_and_context_embedding_tables_are_separate(

    model_name,

    fitted_models,

):

    model = fitted_models[

        model_name

    ]



    parameters = dict(

        model._module.named_parameters()

    )



    (

        target_name,

        context_name,

    ) = _embedding_parameter_names(

        model

    )



    assert (

        parameters[

            target_name

        ].data_ptr()

        !=

        parameters[

            context_name

        ].data_ptr()

    )





@pytest.mark.parametrize(
    (
        "model_name",
        "model_class",
    ),
    MODEL_CLASSES.items(),
)
def test_train_and_eval_paths_use_shared_context_contract(
    model_name,
    model_class,
):
    """Protect the shared EF-EMB train/evaluation context contract.

    Training must explicitly use the shared context-vector machinery,
    exclude the positive target from its own context, and honor the
    configured normalization.

    Evaluation implementations may delegate through helper methods such
    as rate() or _candidate_rates(). Therefore this test inspects the
    complete model class rather than requiring the evaluation-context
    cache reference to appear literally inside score().
    """

    fit_source = inspect.getsource(
        model_class.fit
    )

    class_source = inspect.getsource(
        model_class
    )

    # Training-context contract.
    assert (
        "context_vectors"
        in fit_source
    ), (
        f"{model_name} fit() does not use "
        "the shared context_vectors machinery"
    )

    assert (
        "exclude_target=True"
        in fit_source
    ), (
        f"{model_name} fit() does not explicitly "
        "exclude the positive target from its context"
    )

    assert (
        "self.context_normalization"
        in fit_source
    ), (
        f"{model_name} fit() does not honor "
        "the configured context normalization"
    )

    # Evaluation-context contract.
    #
    # score() may delegate through one or more helpers:
    #
    # score()
    #   -> rate()
    #      -> _candidate_rates()
    #
    # The class as a whole must therefore implement the frozen
    # evaluation-context cache, without forcing one exact call layout.
    assert (
        "_evaluation_context_cache"
        in class_source
    ), (
        f"{model_name} class does not implement "
        "the frozen evaluation-context cache"
    )

    # Historical mechanisms that must stay removed.
    for forbidden in [
        "context_pair_cap",
        "pair_generation",
        "materialized_pairs",
    ]:
        assert (
            forbidden
            not in fit_source
        ), (
            f"{model_name} fit() contains forbidden "
            f"legacy context mechanism: {forbidden}"
        )


@pytest.mark.parametrize(

    "model_name",

    MODEL_CLASSES,

)

def test_candidate_order_independence(

    model_name,

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        model_name

    ]



    catalog = synthetic_data[

        "catalog"

    ]



    forward = model.score(

        1,

        catalog,

    )



    reverse_catalog = (

        catalog[

            ::-1

        ]

    )



    backward = model.score(

        1,

        reverse_catalog,

    )



    forward_map = dict(

        zip(

            catalog.tolist(),

            map(

                float,

                forward,

            ),

        )

    )



    backward_map = dict(

        zip(

            reverse_catalog.tolist(),

            map(

                float,

                backward,

            ),

        )

    )



    for item_id in forward_map:



        assert np.isclose(

            forward_map[

                item_id

            ],

            backward_map[

                item_id

            ],

            rtol=0.0,

            atol=1e-7,

        )





@pytest.mark.parametrize(

    "model_name",

    MODEL_CLASSES,

)

def test_same_seed_and_shuffled_input_reproduce_scores(

    model_name,

    fitted_models,

    synthetic_data,

):

    original = fitted_models[

        model_name

    ]



    shuffled = _fit_model(

        model_name,

        synthetic_data,

        shuffled=True,

    )



    catalog = synthetic_data[

        "catalog"

    ]



    assert np.allclose(

        original.score(

            1,

            catalog,

        ),

        shuffled.score(

            1,

            catalog,

        ),

        rtol=0.0,

        atol=1e-7,

    )





# ============================================================

# b-emb

# ============================================================





def test_b_emb_negatives_are_genuinely_unobserved(

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        "b-emb"

    ]



    sampler = (

        model.negative_sampler

    )



    assert sampler is not None



    all_observed = synthetic_data[

        "all_observed"

    ]



    for user_id in [

        1,

        2,

        3,

        4,

    ]:



        observed = set(

            map(

                int,

                sampler.observed_items(

                    user_id

                ),

            )

        )



        available = set(

            map(

                int,

                sampler.available_items(

                    user_id

                ),

            )

        )



        expected_observed = set(

            all_observed.loc[

                (

                    all_observed[

                        "userId"

                    ]

                    ==

                    user_id

                ),

                "movieId",

            ]

            .astype(

                "int64"

            )

        )



        assert (

            observed

            ==

            expected_observed

        )



        assert observed.isdisjoint(

            available

        )





def test_b_emb_bce_diagnostics_are_finite(

    fitted_models,

):

    record = (

        fitted_models[

            "b-emb"

        ]

        .training_history[

            0

        ]

    )



    assert (

        record[

            "positive_examples"

        ]

        ==

        8

    )



    assert (

        record[

            "negative_examples"

        ]

        ==

        8

    )



    assert (

        record[

            "negative_ratio"

        ]

        ==

        1

    )



    for key in [

        "positive_loss",

        "negative_loss",

        "data_loss",

        "regularization",

        "total_loss",

        "max_abs_logit",

    ]:



        assert np.isfinite(

            float(

                record[

                    key

                ]

            )

        )





def test_b_emb_probability_matches_sigmoid(

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        "b-emb"

    ]



    catalog = synthetic_data[

        "catalog"

    ]



    scores = np.asarray(

        model.score(

            1,

            catalog,

        ),

        dtype=np.float64,

    )



    probabilities = np.asarray(

        model.probability(

            1,

            catalog,

        ),

        dtype=np.float64,

    )



    expected = (

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

        expected,

        rtol=1e-10,

        atol=1e-10,

    )





# ============================================================

# p-emb

# ============================================================





def test_p_emb_masked_poisson_accounting_and_zero_rescaling(

    fitted_models,

):

    model = fitted_models[

        "p-emb"

    ]



    assert (

        model.zero_population_cells

        ==

        8

    )



    assert (

        model.legal_cells

        ==

        16

    )



    record = (

        model.training_history[

            0

        ]

    )



    assert (

        record[

            "positive_examples"

        ]

        ==

        8

    )



    assert (

        record[

            "zero_samples"

        ]

        ==

        8

    )



    assert (

        record[

            "zero_population_cells"

        ]

        ==

        8

    )



    assert (

        record[

            "legal_cells"

        ]

        ==

        16

    )



    assert (

        record[

            "heldout_masked_cells"

        ]

        ==

        4

    )



    assert np.isclose(

        float(

            record[

                "zero_weight_sum"

            ]

        ),

        8.0,

        rtol=0.0,

        atol=1e-12,

    )





def test_p_emb_rates_are_positive_and_match_log_rate_scores(

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        "p-emb"

    ]



    catalog = synthetic_data[

        "catalog"

    ]



    scores = np.asarray(

        model.score(

            1,

            catalog,

        ),

        dtype=np.float64,

    )



    rates = np.asarray(

        model.rate(

            1,

            catalog,

        ),

        dtype=np.float64,

    )



    assert np.isfinite(

        rates

    ).all()



    assert np.all(

        rates

        >

        0.0

    )



    expected = np.exp(

        np.clip(

            scores,

            -20.0,

            20.0,

        )

    )



    assert np.allclose(

        rates,

        expected,

        rtol=0.0,

        atol=1e-12,

    )





# ============================================================

# ap-emb

# ============================================================





def test_ap_emb_masked_poisson_accounting(

    fitted_models,

):

    model = fitted_models[

        "ap-emb"

    ]



    assert (

        model.zero_population_cells

        ==

        8

    )



    assert (

        model.legal_cells

        ==

        16

    )



    record = (

        model.training_history[

            0

        ]

    )



    assert (

        record[

            "positive_examples"

        ]

        ==

        8

    )



    assert (

        record[

            "zero_samples"

        ]

        ==

        8

    )



    assert (

        record[

            "zero_population_cells"

        ]

        ==

        8

    )



    assert (

        record[

            "legal_cells"

        ]

        ==

        16

    )



    assert (

        record[

            "heldout_masked_cells"

        ]

        ==

        4

    )



    assert np.isclose(

        float(

            record[

                "zero_weight_sum"

            ]

        ),

        8.0,

        rtol=0.0,

        atol=1e-12,

    )





def test_ap_emb_uses_positive_exponential_embeddings(

    fitted_models,

):

    model = fitted_models[

        "ap-emb"

    ]



    parameters = dict(

        model._module.named_parameters()

    )



    (

        target_name,

        context_name,

    ) = _embedding_parameter_names(

        model

    )



    assert (

        "raw"

        in target_name.lower()

        or

        "log"

        in target_name.lower()

    )



    assert (

        "raw"

        in context_name.lower()

        or

        "log"

        in context_name.lower()

    )



    target_plus = torch.exp(

        parameters[

            target_name

        ]

    )



    context_plus = torch.exp(

        parameters[

            context_name

        ]

    )



    assert torch.isfinite(

        target_plus

    ).all()



    assert torch.isfinite(

        context_plus

    ).all()



    assert torch.all(

        target_plus

        >

        0.0

    )



    assert torch.all(

        context_plus

        >

        0.0

    )



    module_source = inspect.getsource(

        type(

            model._module

        )

    )



    assert (

        "torch.exp"

        in module_source

        or

        ".exp("

        in module_source

    )





def test_ap_emb_rate_is_positive_and_score_is_log_rate(

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        "ap-emb"

    ]



    catalog = synthetic_data[

        "catalog"

    ]



    scores = np.asarray(

        model.score(

            1,

            catalog,

        ),

        dtype=np.float64,

    )



    rates = np.asarray(

        model.rate(

            1,

            catalog,

        ),

        dtype=np.float64,

    )



    assert np.isfinite(

        rates

    ).all()



    assert np.all(

        rates

        >

        0.0

    )



    assert np.allclose(

        scores,

        np.log(

            rates

        ),

        rtol=0.0,

        atol=1e-7,

    )





def test_p_emb_and_ap_emb_have_different_rate_functions(

    fitted_models,

    synthetic_data,

):

    catalog = synthetic_data[

        "catalog"

    ]



    p_rates = np.asarray(

        fitted_models[

            "p-emb"

        ].rate(

            1,

            catalog,

        ),

        dtype=np.float64,

    )



    ap_rates = np.asarray(

        fitted_models[

            "ap-emb"

        ].rate(

            1,

            catalog,

        ),

        dtype=np.float64,

    )



    assert not np.allclose(

        p_rates,

        ap_rates,

        rtol=1e-7,

        atol=1e-7,

    )





# ============================================================

# g-emb

# ============================================================





def test_g_emb_uses_training_rating_mean_and_native_metrics(

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        "g-emb"

    ]



    expected_mean = float(

        np.mean(

            synthetic_data[

                "train"

            ][

                "rating"

            ]

            .to_numpy(

                dtype=np.float64,

            )

        )

    )



    assert np.isclose(

        model.global_mean,

        expected_mean,

        rtol=0.0,

        atol=1e-12,

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





def test_g_emb_training_diagnostics_are_finite(

    fitted_models,

):

    record = (

        fitted_models[

            "g-emb"

        ]

        .training_history[

            0

        ]

    )



    assert (

        record[

            "examples"

        ]

        ==

        8

    )



    assert (

        record[

            "nonfinite_events"

        ]

        ==

        0

    )



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



        assert np.isfinite(

            float(

                record[

                    key

                ]

            )

        )



    assert np.isclose(

        float(

            record[

                "rmse"

            ]

        ),

        math.sqrt(

            float(

                record[

                    "mse"

                ]

            )

        ),

        rtol=0.0,

        atol=1e-12,

    )





# ============================================================

# ng-emb / B92

# ============================================================





def test_ng_emb_uses_exact_softplus_for_both_embedding_tables(

    fitted_models,

):

    model = fitted_models[

        "ng-emb"

    ]



    raw_targets = (

        model._module

        .raw_target_embeddings

        .weight

    )



    raw_contexts = (

        model._module

        .raw_context_embeddings

        .weight

    )



    transformed_targets = (

        model._module

        .positive_target_embeddings()

    )



    transformed_contexts = (

        model._module

        .positive_context_embeddings()

    )



    assert torch.allclose(

        transformed_targets,

        F.softplus(

            raw_targets

        ),

        rtol=0.0,

        atol=0.0,

    )



    assert torch.allclose(

        transformed_contexts,

        F.softplus(

            raw_contexts

        ),

        rtol=0.0,

        atol=0.0,

    )



    assert torch.all(

        transformed_targets

        >

        0.0

    )



    assert torch.all(

        transformed_contexts

        >

        0.0

    )





def test_ng_emb_inference_uses_softplus_not_raw_parameters(

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        "ng-emb"

    ]



    user_id = 1



    item_id = int(

        synthetic_data[

            "catalog"

        ][

            0

        ]

    )



    actual = float(

        model.score(

            user_id,

            np.asarray(

                [

                    item_id

                ],

                dtype=np.int64,

            ),

        )[

            0

        ]

    )



    user_index = (

        model._context_store

        .user_index(

            user_id

        )

    )



    item_index = (

        model._context_store

        .item_index(

            item_id

        )

    )



    history_indices = (

        model._context_store

        .history_indices(

            user_id

        )

    )



    with torch.no_grad():



        item_tensor = torch.tensor(

            [

                item_index

            ],

            dtype=torch.long,

        )



        context_plus = torch.as_tensor(

            model._evaluation_context_cache[

                user_index

            ],

            dtype=torch.float32,

        )



        target_plus = F.softplus(

            model._module

            .raw_target_embeddings(

                item_tensor

            )[

                0

            ]

        )



        raw_target = (

            model._module

            .raw_target_embeddings(

                item_tensor

            )[

                0

            ]

        )



        bias = (

            model._module

            .item_bias(

                item_tensor

            )

            .squeeze(

                1

            )[

                0

            ]

        )



        expected = (

            model.global_mean

            +

            bias

            +

            (

                target_plus

                *

                context_plus

            ).sum()

        ).item()



        wrong_raw_score = (

            model.global_mean

            +

            bias

            +

            (

                raw_target

                *

                context_plus

            ).sum()

        ).item()



        manual_softplus_context = (

            F.softplus(

                model._module

                .raw_context_embeddings

                .weight[

                    torch.tensor(

                        history_indices,

                        dtype=torch.long,

                    )

                ]

            )

            .mean(

                dim=0

            )

        )



        raw_context_mean = (

            model._module

            .raw_context_embeddings

            .weight[

                torch.tensor(

                    history_indices,

                    dtype=torch.long,

                )

            ]

            .mean(

                dim=0

            )

        )



    assert np.isclose(

        actual,

        expected,

        rtol=0.0,

        atol=1e-7,

    )



    assert not np.isclose(

        actual,

        wrong_raw_score,

        rtol=1e-5,

        atol=1e-6,

    )



    assert torch.allclose(

        context_plus,

        manual_softplus_context,

        rtol=0.0,

        atol=1e-6,

    )



    assert not torch.allclose(

        context_plus,

        raw_context_mean,

        rtol=1e-5,

        atol=1e-6,

    )





def test_ng_emb_native_metrics_and_training_diagnostics_are_finite(

    fitted_models,

    synthetic_data,

):

    model = fitted_models[

        "ng-emb"

    ]



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



    record = (

        model.training_history[

            0

        ]

    )



    assert (

        record[

            "examples"

        ]

        ==

        8

    )



    assert (

        record[

            "nonfinite_events"

        ]

        ==

        0

    )



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



        assert np.isfinite(

            float(

                record[

                    key

                ]

            )

        )

 # ============================================================
# B43: exact family-native predictive likelihood diagnostics
# ============================================================


def _manual_bernoulli_native_metrics(
    model,
    evaluation_frame,
):
    """Independently reconstruct exact masked Bernoulli NLL."""
    sampler = model.negative_sampler

    assert sampler is not None

    positive_terms = []
    zero_terms = []

    for user_id, group in evaluation_frame.groupby(
        "userId",
        sort=True,
    ):

        items = (
            group[
                "movieId"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        logits = model.score(
            int(user_id),
            items,
        )

        positive_terms.extend(
            np.logaddexp(
                0.0,
                -logits,
            ).tolist()
        )

    for user_id_raw in model.user_ids:

        user_id = int(
            user_id_raw
        )

        zeros = np.asarray(
            sampler.available_items(
                user_id
            ),
            dtype=np.int64,
        )

        if len(zeros) == 0:
            continue

        logits = model.score(
            user_id,
            zeros,
        )

        zero_terms.extend(
            np.logaddexp(
                0.0,
                logits,
            ).tolist()
        )

    positive_terms = np.asarray(
        positive_terms,
        dtype=np.float64,
    )

    zero_terms = np.asarray(
        zero_terms,
        dtype=np.float64,
    )

    assert len(
        positive_terms
    ) == len(
        evaluation_frame
    )

    assert len(
        zero_terms
    ) == 8

    positive_mean = float(
        positive_terms.mean()
    )

    zero_mean = float(
        zero_terms.mean()
    )

    masked_mean = float(
        (
            positive_terms.sum(
                dtype=np.float64
            )
            +
            zero_terms.sum(
                dtype=np.float64
            )
        )
        /
        (
            len(
                positive_terms
            )
            +
            len(
                zero_terms
            )
        )
    )

    return {
        "bernoulli_positive_mean_nll": (
            positive_mean
        ),
        "bernoulli_zero_mean_nll": (
            zero_mean
        ),
        "masked_predictive_bernoulli_nll": (
            masked_mean
        ),
    }


def _manual_poisson_native_metrics(
    model,
    evaluation_frame,
):
    """Independently reconstruct exact masked Poisson NLL."""
    sampler = model.zero_sampler

    assert sampler is not None

    positive_terms = []
    zero_terms = []

    for user_id, group in evaluation_frame.groupby(
        "userId",
        sort=True,
    ):

        items = (
            group[
                "movieId"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        rates = model.rate(
            int(user_id),
            items,
        )

        positive_terms.extend(
            (
                rates
                -
                np.log(
                    rates
                )
            ).tolist()
        )

    for user_id_raw in model.user_ids:

        user_id = int(
            user_id_raw
        )

        zeros = np.asarray(
            sampler.available_items(
                user_id
            ),
            dtype=np.int64,
        )

        if len(zeros) == 0:
            continue

        rates = model.rate(
            user_id,
            zeros,
        )

        zero_terms.extend(
            rates.tolist()
        )

    positive_terms = np.asarray(
        positive_terms,
        dtype=np.float64,
    )

    zero_terms = np.asarray(
        zero_terms,
        dtype=np.float64,
    )

    assert len(
        positive_terms
    ) == len(
        evaluation_frame
    )

    assert len(
        zero_terms
    ) == 8

    positive_mean = float(
        positive_terms.mean()
    )

    zero_mean = float(
        zero_terms.mean()
    )

    masked_mean = float(
        (
            positive_terms.sum(
                dtype=np.float64
            )
            +
            zero_terms.sum(
                dtype=np.float64
            )
        )
        /
        (
            len(
                positive_terms
            )
            +
            len(
                zero_terms
            )
        )
    )

    return {
        "poisson_positive_mean_nll": (
            positive_mean
        ),
        "poisson_zero_mean_nll": (
            zero_mean
        ),
        "masked_predictive_poisson_nll": (
            masked_mean
        ),
    }


def test_b_emb_native_metrics_match_exact_masked_bernoulli_nll(
    fitted_models,
    synthetic_data,
):
    model = fitted_models[
        "b-emb"
    ]

    heldout = synthetic_data[
        "heldout"
    ]

    actual = model.native_metrics(
        heldout
    )

    expected = (
        _manual_bernoulli_native_metrics(
            model,
            heldout,
        )
    )

    assert set(
        actual
    ) == {
        "bernoulli_positive_mean_nll",
        "bernoulli_zero_mean_nll",
        "masked_predictive_bernoulli_nll",
    }

    for key in expected:

        assert np.isfinite(
            actual[
                key
            ]
        )

        assert (
            actual[
                key
            ]
            >=
            0.0
        )

        assert np.isclose(
            actual[
                key
            ],
            expected[
                key
            ],
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "p-emb",
        "ap-emb",
    ],
)
def test_poisson_efemb_native_metrics_match_exact_masked_nll(
    model_name,
    fitted_models,
    synthetic_data,
):
    model = fitted_models[
        model_name
    ]

    heldout = synthetic_data[
        "heldout"
    ]

    actual = model.native_metrics(
        heldout
    )

    expected = (
        _manual_poisson_native_metrics(
            model,
            heldout,
        )
    )

    assert set(
        actual
    ) == {
        "poisson_positive_mean_nll",
        "poisson_zero_mean_nll",
        "masked_predictive_poisson_nll",
    }

    assert (
        model.zero_population_cells
        ==
        8
    )

    for key in expected:

        assert np.isfinite(
            actual[
                key
            ]
        )

        assert (
            actual[
                key
            ]
            >=
            0.0
        )

        assert np.isclose(
            actual[
                key
            ],
            expected[
                key
            ],
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "b-emb",
        "p-emb",
        "ap-emb",
    ],
)
def test_native_metrics_reject_training_positive_as_evaluation_positive(
    model_name,
    fitted_models,
    synthetic_data,
):
    model = fitted_models[
        model_name
    ]

    illegal = (
        synthetic_data[
            "train"
        ]
        .iloc[
            [
                0,
            ]
        ]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match=(
            "must be a subset "
            "of heldout_interactions"
        ),
    ):
        model.native_metrics(
            illegal
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "b-emb",
        "p-emb",
        "ap-emb",
    ],
)
def test_native_metrics_are_invariant_to_evaluation_row_order(
    model_name,
    fitted_models,
    synthetic_data,
):
    model = fitted_models[
        model_name
    ]

    heldout = synthetic_data[
        "heldout"
    ]

    shuffled = (
        heldout.sample(
            frac=1.0,
            random_state=9876,
        )
        .reset_index(
            drop=True
        )
    )

    original = model.native_metrics(
        heldout
    )

    reordered = model.native_metrics(
        shuffled
    )

    assert (
        original.keys()
        ==
        reordered.keys()
    )

    for key in original:

        assert np.isclose(
            original[
                key
            ],
            reordered[
                key
            ],
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "b-emb",
        "p-emb",
        "ap-emb",
    ],
)
def test_native_metrics_support_heldout_subset_without_changing_zero_universe(
    model_name,
    fitted_models,
    synthetic_data,
):
    model = fitted_models[
        model_name
    ]

    full = synthetic_data[
        "heldout"
    ]

    subset = (
        full.iloc[
            [
                0,
                2,
            ]
        ]
        .copy()
    )

    metrics = model.native_metrics(
        subset
    )

    assert all(
        np.isfinite(
            float(
                value
            )
        )
        for value
        in metrics.values()
    )

    if model_name == "b-emb":

        sampler = (
            model.negative_sampler
        )

    else:

        sampler = (
            model.zero_sampler
        )

    assert sampler is not None

    zero_cells = sum(
        len(
            sampler.available_items(
                int(user_id)
            )
        )
        for user_id
        in model.user_ids
    )

    # The other held-out positives remain masked rather than
    # becoming zeros when only a subset is evaluated.
    assert zero_cells == 8


@pytest.mark.parametrize(
    "model_name",
    [
        "b-emb",
        "p-emb",
        "ap-emb",
    ],
)
def test_native_metrics_do_not_resample_training_negatives_or_zeros(
    model_name,
    fitted_models,
    synthetic_data,
    monkeypatch,
):
    model = fitted_models[
        model_name
    ]

    if model_name == "b-emb":

        sampler = (
            model.negative_sampler
        )

    else:

        sampler = (
            model.zero_sampler
        )

    assert sampler is not None

    def forbidden_sample(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "native_metrics must not sample "
            "new negatives or zero cells"
        )

    monkeypatch.setattr(
        sampler,
        "sample",
        forbidden_sample,
    )

    metrics = model.native_metrics(
        synthetic_data[
            "heldout"
        ]
    )

    assert metrics

    assert all(
        np.isfinite(
            float(
                value
            )
        )
        for value
        in metrics.values()
    )       