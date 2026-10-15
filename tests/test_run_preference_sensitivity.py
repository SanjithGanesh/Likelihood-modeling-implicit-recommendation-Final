from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import scripts.run_preference_sensitivity as pref
from src.split import SplitResult


def _synthetic_split() -> SplitResult:
    # Build a training catalog with at least 25 items so the evaluator can
    # validly exercise all frozen cutoffs K={5,10,20}.
    #
    # Users 1 and 2 have small personal histories. User 4 exists only to
    # place additional items in the canonical training catalog.
    extra_items = list(range(60, 260, 10))  # 20 items: 60,...,250

    train = pd.DataFrame(
        {
            "userId": (
                [1, 1, 2, 2, 4, 4]
                + [4] * len(extra_items)
            ),
            "movieId": (
                [10, 20, 10, 30, 40, 50]
                + extra_items
            ),
            "rating": (
                [5.0, 3.0, 4.0, 2.0, 4.0, 5.0]
                + [4.0] * len(extra_items)
            ),
            "timestamp": (
                [1, 2, 1, 2, 1, 2]
                + list(
                    range(
                        3,
                        3 + len(extra_items),
                    )
                )
            ),
        }
    )

    validation = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [30, 20],
            "rating": [4.0, 5.0],
            "timestamp": [3, 3],
        }
    )

    test = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [40, 50, 40, 50, 40],
            "rating": [5.0, 2.0, 4.0, 1.0, 3.0],
            "timestamp": [4, 5, 4, 5, 1],
        }
    )

    return SplitResult(
        train=train,
        validation=validation,
        test=test,
        seed=2026,
        attempt=1,
        enforce_train_item_coverage=True,
    )


def test_frozen_hashes_are_exact():
    assert pref.PREFERENCE_PROTOCOL_SHA256 == (
        "0ca60905d5cd71e9a589481e81919d77"
        "2db1903517f03603d9490ffbfd7708bc"
    )

    assert pref.FINAL_HYPERPARAMETERS_SHA256 == (
        "9aae84dd59d7c34c82a67563139bdb272"
        "56874ba79c645c749cbcf788d72c981"
    )

    assert pref.FINAL_EVALUATION_SHA256 == (
        "1874aeba214b3a049b439fa4d6b8aabb"
        "b8c20e6aa98fdfbeb94c0bb6348002f8"
    )


def test_exact_model_seed_matrix_is_39():
    matrix = pref._planned_matrix(
        model_filter="all",
        seed_filter=None,
    )

    assert len(matrix) == 39
    assert len(set(matrix)) == 39

    assert {
        model
        for model, _
        in matrix
    } == set(pref.MODEL_ORDER)

    assert {
        seed
        for _, seed
        in matrix
    } == {
        2026,
        2027,
        2028,
    }


def test_final_fit_split_is_train_plus_validation():
    split = _synthetic_split()

    final_fit = pref._final_fit_split(
        split
    )

    assert len(final_fit.train) == 28
    assert len(final_fit.validation) == 0
    assert len(final_fit.test) == 5

    assert len(
        final_fit.train["movieId"].unique()
    ) == 25

    assert final_fit.train.iloc[:6][
        ["userId", "movieId"]
    ].values.tolist() == [
        [1, 10],
        [1, 20],
        [2, 10],
        [2, 30],
        [4, 40],
        [4, 50],
    ]

    assert final_fit.train.iloc[-2:][
        ["userId", "movieId"]
    ].values.tolist() == [
        [1, 30],
        [2, 20],
    ]

    assert split.validation.shape[0] == 2


def test_deterministic_rank_uses_score_desc_item_asc():
    items = np.asarray(
        [40, 20, 30, 10],
        dtype=np.int64,
    )

    scores = np.asarray(
        [0.7, 0.7, 0.9, 0.7],
        dtype=np.float64,
    )

    ranked = pref._deterministic_rank(
        items,
        scores,
    )

    assert ranked.tolist() == [
        30,
        10,
        20,
        40,
    ]


def test_preference_evaluator_keeps_low_rated_test_items_as_candidates(
    monkeypatch,
):
    split = _synthetic_split()

    monkeypatch.setattr(
        pref,
        "EXPECTED_ELIGIBLE_USERS",
        2,
    )

    seen = {}

    def score(user_id, candidates):
        seen[int(user_id)] = set(
            int(x)
            for x in candidates
        )

        # Favor item 40.
        return np.asarray(
            [
                10.0 if int(item) == 40 else 0.0
                for item in candidates
            ],
            dtype=np.float64,
        )

    aggregate, per_user = (
        pref.evaluate_preference_ranking(
            canonical=split,
            score_function=score,
        )
    )

    # User 1's rating-2 test item 50 is still in the candidate set.
    assert 50 in seen[1]

    # User 3 has only rating 3 in test and is therefore omitted.
    assert 3 not in seen

    assert set(
        per_user["userId"].tolist()
    ) == {
        1,
        2,
    }

    assert aggregate["ndcg_at_5"] == pytest.approx(
        1.0
    )


def test_selected_hyperparameters_replace_only_seed_for_tuned_model():
    frozen = {
        "tuned_models": {
            "g-emb": {
                "hyperparameters": {
                    "n_factors": 64,
                    "learning_rate": 0.05,
                    "context_normalization": "mean",
                    "seed": 2026,
                }
            }
        }
    }

    original = deepcopy(
        frozen["tuned_models"]["g-emb"]["hyperparameters"]
    )

    result = pref._selected_hyperparameters(
        frozen_hp=frozen,
        model_name="g-emb",
        seed=2028,
    )

    assert result["seed"] == 2028
    assert result["n_factors"] == 64
    assert result["learning_rate"] == 0.05
    assert result["context_normalization"] == "mean"

    assert frozen[
        "tuned_models"
    ][
        "g-emb"
    ][
        "hyperparameters"
    ] == original


def test_baseline_hyperparameter_identity():
    frozen = {
        "tuned_models": {}
    }

    assert pref._selected_hyperparameters(
        frozen_hp=frozen,
        model_name="Popularity",
        seed=2027,
    ) == {}

    assert pref._selected_hyperparameters(
        frozen_hp=frozen,
        model_name="Random",
        seed=2027,
    ) == {
        "seed": 2027,
        "namespace": "random_baseline",
    }


def test_spec_is_secondary_variant():
    spec = pref._build_spec(
        model_name="PoissonMF",
        seed=2026,
        hyperparameters={
            "seed": 2026,
        },
    )

    assert spec.model_name == "PoissonMF"
    assert spec.family == "poisson"
    assert spec.variant == "preference-r4"
    assert spec.seed == 2026


def test_manifest_forbids_retuning_and_primary_replacement():
    payload = pref._protocol_manifest_payload()

    assert payload["planned_experiments"] == 39
    assert payload["target_split"] == "test"
    assert payload["relevance_threshold"] == 4.0
    assert payload["secondary_analysis_only"] is True
    assert payload["retuning_allowed"] is False
    assert payload["primary_results_affected"] is False


def test_cli_requires_explicit_execution_gate():
    parser = pref.build_argument_parser()

    args = parser.parse_args([])

    assert args.execute_preference_test is False
    assert args.model == "all"
    assert args.seed is None

    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--target-split" not in option_strings
    assert "--allow-test" not in option_strings
    assert "--execute-final-test" not in option_strings


def test_preference_threshold_is_frozen_at_four():
    assert pref.PREFERENCE_THRESHOLD == 4.0


def test_k_values_are_frozen():
    assert pref.K_VALUES == (
        5,
        10,
        20,
    )


def test_model_order_matches_frozen_inventory():
    assert pref.MODEL_ORDER == (
        "Popularity",
        "Random",
        "PCA/SVD",
        "BernoulliMF",
        "PoissonMF",
        "GaussianMF",
        "LogisticPCA",
        "HPF",
        "b-emb",
        "p-emb",
        "ap-emb",
        "g-emb",
        "ng-emb",
    )


def test_per_user_sidecar_schema_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pref,
        "EXPECTED_ELIGIBLE_USERS",
        2,
    )

    rows = []

    for user_id in [1, 2]:
        row = {
            "userId": user_id,
            "n_relevant": 1,
            "n_candidates": 100,
        }

        for k in pref.K_VALUES:
            for metric in (
                "precision",
                "recall",
                "hit_rate",
                "ndcg",
                "mrr",
                "map",
            ):
                row[
                    f"{metric}_at_{k}"
                ] = 0.5

        rows.append(row)

    frame = pd.DataFrame(rows)

    path = tmp_path / "per_user.csv"

    frame.to_csv(
        path,
        index=False,
    )

    loaded = pref._verify_per_user_sidecar(
        path
    )

    assert len(loaded) == 2


def test_preflight_matrix_filter_is_deterministic():
    matrix = pref._planned_matrix(
        model_filter="LogisticPCA",
        seed_filter=2028,
    )

    assert matrix == [
        ("LogisticPCA", 2028)
    ]
