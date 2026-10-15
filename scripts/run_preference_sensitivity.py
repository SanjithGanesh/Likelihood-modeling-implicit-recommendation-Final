"""Run Paper 2 Step 58 preference-aware relevance sensitivity analysis.

The scientific model is unchanged from the frozen Step-56 final evaluation.
Every model is deterministically refit from scratch on canonical
train+validation with the exact Step-55 selected hyperparameters and final
seeds.  Only the relevance definition changes:

    relevant test item iff rating >= 4.0

All canonical test items remain candidate items unless excluded by the same
train+validation history rule used in the primary final evaluation.
Users with no >=4.0 test item are excluded from this secondary metric
aggregation because Recall/NDCG/MRR/MAP are undefined for an empty relevant
set.

This runner cannot tune, select hyperparameters, change seeds, change the
candidate catalog, or replace the primary final results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_baselines import build_baseline
from scripts.run_mf import load_frozen_split, prepare_run_context
from scripts.tune_models import build_tuning_model
from src.data import load_yaml
from src.experiment import ExperimentResult, ExperimentSpec
from src.metrics import ranking_metrics_at_k
from src.results import (
    experiment_id_for_spec,
    load_completed_experiment,
    write_experiment_result,
)
from src.split import SplitResult


# =============================================================================
# Frozen Step 58 protocol
# =============================================================================

PREFERENCE_PROTOCOL_PATH = (
    PROJECT_ROOT / "config" / "preference_sensitivity_v1.yaml"
)

FINAL_HYPERPARAMETERS_PATH = (
    PROJECT_ROOT / "config" / "final_hyperparameters_v1.yaml"
)

FINAL_EVALUATION_PATH = (
    PROJECT_ROOT / "config" / "final_evaluation_v1.yaml"
)

PUBLICATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "publication.yaml"
)

PREFERENCE_PROTOCOL_SHA256 = (
    "0ca60905d5cd71e9a589481e81919d77"
    "2db1903517f03603d9490ffbfd7708bc"
)

FINAL_HYPERPARAMETERS_SHA256 = (
    "9aae84dd59d7c34c82a67563139bdb272"
    "56874ba79c645c749cbcf788d72c981"
)

FINAL_EVALUATION_SHA256 = (
    "1874aeba214b3a049b439fa4d6b8aabb"
    "b8c20e6aa98fdfbeb94c0bb6348002f8"
)

RUN_ID = "preference-ml1m-v1-0ca60905"

VERSION = "v1"

PREFERENCE_THRESHOLD = 4.0

EXPECTED_TRAIN_ROWS = 805443
EXPECTED_VALIDATION_ROWS = 97383
EXPECTED_TEST_ROWS = 97383
EXPECTED_FINAL_FIT_ROWS = 902826
EXPECTED_CATALOG_ITEMS = 3706
EXPECTED_PREFERRED_TEST_ROWS = 55964
EXPECTED_NONPREFERRED_TEST_ROWS = 41419
EXPECTED_ELIGIBLE_USERS = 5851
EXPECTED_ZERO_PREFERRED_USERS = 189
EXPECTED_UNIQUE_PREFERRED_ITEMS = 2887

K_VALUES = (5, 10, 20)

SEED_ORDER = (2026, 2027, 2028)

MODEL_ORDER = (
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

EXPECTED_EXPERIMENTS = 39

MODEL_FAMILY = {
    "Popularity": "baseline",
    "Random": "baseline",
    "PCA/SVD": "baseline",
    "BernoulliMF": "bernoulli",
    "PoissonMF": "poisson",
    "GaussianMF": "gaussian",
    "LogisticPCA": "bernoulli",
    "HPF": "poisson",
    "b-emb": "bernoulli",
    "p-emb": "poisson",
    "ap-emb": "poisson",
    "g-emb": "gaussian",
    "ng-emb": "gaussian",
}

BASELINE_KEY = {
    "Popularity": "popularity",
    "Random": "random",
}


# =============================================================================
# Generic helpers
# =============================================================================


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    # PowerShell 5.1 Set-Content -Encoding UTF8 emits a BOM.
    with path.open("r", encoding="utf-8-sig") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise TypeError(
            f"YAML root must be a mapping: {path}"
        )

    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON value must be finite")
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("JSON value must be finite")
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(child)
            for key, child in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(child)
            for child in value
        ]

    return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    safe = _json_safe(payload)

    encoded = (
        json.dumps(
            safe,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    if path.exists():
        existing = json.loads(
            path.read_text(encoding="utf-8")
        )
        if existing != safe:
            raise ValueError(
                "Existing frozen JSON differs from current payload: "
                f"{path}"
            )
        return

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    try:
        temporary.write_text(
            encoded,
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv_replace(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    """Atomically create/replace a deterministic derived sidecar.

    The ordinary ExperimentResult checkpoint is written only after this file.
    If a process dies after the sidecar but before the checkpoint, the next
    deterministic refit is allowed to replace the orphan sidecar.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    try:
        frame.to_csv(
            temporary,
            index=False,
            lineterminator="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _training_history(model: Any) -> list[dict[str, Any]]:
    try:
        history = getattr(model, "training_history")
    except Exception:
        return []

    if callable(history):
        try:
            history = history()
        except Exception:
            return []

    if history is None:
        return []

    if not isinstance(history, list):
        raise TypeError(
            "model.training_history must be a list when exposed"
        )

    return deepcopy(history)


def _deterministic_rank(
    candidate_items: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Sort score descending, then item ID ascending."""
    items = np.asarray(
        candidate_items,
        dtype=np.int64,
    )
    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    if items.ndim != 1 or values.ndim != 1:
        raise ValueError(
            "candidate_items and scores must be one-dimensional"
        )

    if len(items) != len(values):
        raise ValueError(
            "candidate_items and scores must have equal length"
        )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Preference evaluator received non-finite scores"
        )

    order = np.lexsort(
        (
            items,
            -values,
        )
    )

    return items[order]


def _items_by_user(frame: pd.DataFrame) -> dict[int, frozenset[int]]:
    result: dict[int, set[int]] = {}

    for row in (
        frame[["userId", "movieId"]]
        .itertuples(index=False)
    ):
        result.setdefault(
            int(row.userId),
            set(),
        ).add(
            int(row.movieId)
        )

    return {
        user_id: frozenset(items)
        for user_id, items in result.items()
    }


def _candidate_items(
    catalog: np.ndarray,
    excluded_items: set[int],
) -> np.ndarray:
    if not excluded_items:
        return catalog.copy()

    mask = np.fromiter(
        (
            int(item_id) not in excluded_items
            for item_id in catalog
        ),
        dtype=bool,
        count=len(catalog),
    )

    return catalog[mask]


# =============================================================================
# Frozen-input validation
# =============================================================================


def _validate_protocol(protocol: dict[str, Any]) -> None:
    _require(
        protocol.get("schema_version") == "1.0",
        "Unexpected Step 58 schema_version",
    )
    _require(
        protocol.get("freeze_id")
        == "paper2-preference-sensitivity-v1",
        "Unexpected Step 58 freeze_id",
    )
    _require(
        protocol.get("status") == "frozen",
        "Step 58 protocol is not frozen",
    )

    training = protocol.get("training_protocol")
    _require(
        isinstance(training, dict),
        "Step 58 training_protocol is missing",
    )
    _require(
        training.get("fit_partitions")
        == ["train", "validation"],
        "Step 58 must refit train+validation",
    )
    _require(
        int(training.get("final_fit_rows", -1))
        == EXPECTED_FINAL_FIT_ROWS,
        "Step 58 final-fit row count changed",
    )
    _require(
        training.get("hyperparameter_retuning") is False,
        "Step 58 retuning must remain forbidden",
    )
    _require(
        tuple(training.get("seeds", ())) == SEED_ORDER,
        "Step 58 seed set changed",
    )

    _require(
        tuple(protocol.get("model_inventory", ()))
        == MODEL_ORDER,
        "Step 58 model inventory/order changed",
    )
    _require(
        int(protocol.get("planned_experiments", -1))
        == EXPECTED_EXPERIMENTS,
        "Step 58 experiment count changed",
    )

    evaluation = protocol.get("evaluation")
    _require(
        isinstance(evaluation, dict),
        "Step 58 evaluation mapping missing",
    )
    _require(
        evaluation.get("target_split") == "test",
        "Step 58 must target canonical test",
    )
    _require(
        evaluation.get("ranking_mode") == "full_catalog",
        "Step 58 must remain full-catalog",
    )
    _require(
        evaluation.get("candidate_catalog")
        == "canonical_train_item_catalog",
        "Step 58 candidate catalog changed",
    )

    relevance = protocol.get("preference_relevance")
    _require(
        isinstance(relevance, dict),
        "Step 58 preference_relevance missing",
    )
    _require(
        float(relevance.get("threshold"))
        == PREFERENCE_THRESHOLD,
        "Step 58 rating threshold changed",
    )
    _require(
        int(relevance.get("eligible_users"))
        == EXPECTED_ELIGIBLE_USERS,
        "Step 58 eligible-user count changed",
    )
    _require(
        int(
            relevance.get(
                "users_with_zero_preferred_test_items"
            )
        )
        == EXPECTED_ZERO_PREFERRED_USERS,
        "Step 58 zero-preferred-user count changed",
    )

    metrics = protocol.get("metrics")
    _require(
        isinstance(metrics, dict),
        "Step 58 metrics mapping missing",
    )
    _require(
        tuple(metrics.get("k_values", ())) == K_VALUES,
        "Step 58 K values changed",
    )

    interpretation = protocol.get("interpretation")
    _require(
        isinstance(interpretation, dict),
        "Step 58 interpretation mapping missing",
    )

    for key in (
        "may_replace_primary_results",
        "may_trigger_retuning",
        "may_trigger_model_redefinition",
        "may_trigger_seed_selection",
    ):
        _require(
            interpretation.get(key) is False,
            f"Step 58 forbidden interpretation changed: {key}",
        )


def _validate_frozen_files() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    actual_preference_sha = _sha256_file(
        PREFERENCE_PROTOCOL_PATH
    )
    _require(
        actual_preference_sha == PREFERENCE_PROTOCOL_SHA256,
        (
            "Step 58 protocol SHA mismatch. "
            f"Expected {PREFERENCE_PROTOCOL_SHA256}; "
            f"found {actual_preference_sha}"
        ),
    )

    actual_hp_sha = _sha256_file(
        FINAL_HYPERPARAMETERS_PATH
    )
    _require(
        actual_hp_sha == FINAL_HYPERPARAMETERS_SHA256,
        (
            "Final hyperparameter SHA mismatch. "
            f"Expected {FINAL_HYPERPARAMETERS_SHA256}; "
            f"found {actual_hp_sha}"
        ),
    )

    actual_final_eval_sha = _sha256_file(
        FINAL_EVALUATION_PATH
    )
    _require(
        actual_final_eval_sha == FINAL_EVALUATION_SHA256,
        (
            "Final-evaluation protocol SHA mismatch. "
            f"Expected {FINAL_EVALUATION_SHA256}; "
            f"found {actual_final_eval_sha}"
        ),
    )

    protocol = _load_yaml_mapping(
        PREFERENCE_PROTOCOL_PATH
    )
    frozen_hp = _load_yaml_mapping(
        FINAL_HYPERPARAMETERS_PATH
    )

    _validate_protocol(protocol)

    tuned = frozen_hp.get("tuned_models")
    _require(
        isinstance(tuned, dict),
        "final_hyperparameters_v1.yaml missing tuned_models",
    )

    expected_tuned = set(MODEL_ORDER) - {
        "Popularity",
        "Random",
    }

    _require(
        expected_tuned.issubset(set(tuned)),
        (
            "Frozen hyperparameter file is missing one or more "
            "Step 58 tuned models"
        ),
    )

    return protocol, frozen_hp


# =============================================================================
# Canonical split / preference population
# =============================================================================


def _final_fit_split(
    canonical: SplitResult,
) -> SplitResult:
    final_train = pd.concat(
        [
            canonical.train,
            canonical.validation,
        ],
        ignore_index=True,
    )

    # Preserve exact schema/dtypes as far as pandas permits.
    empty_validation = canonical.validation.iloc[
        0:0
    ].copy()

    return SplitResult(
        train=final_train,
        validation=empty_validation,
        test=canonical.test.copy(),
        seed=canonical.seed,
        attempt=canonical.attempt,
        enforce_train_item_coverage=(
            canonical.enforce_train_item_coverage
        ),
    )


def _preference_population(
    canonical: SplitResult,
) -> dict[str, Any]:
    train = canonical.train
    validation = canonical.validation
    test = canonical.test

    _require(
        len(train) == EXPECTED_TRAIN_ROWS,
        "Canonical train row count changed",
    )
    _require(
        len(validation) == EXPECTED_VALIDATION_ROWS,
        "Canonical validation row count changed",
    )
    _require(
        len(test) == EXPECTED_TEST_ROWS,
        "Canonical test row count changed",
    )

    _require(
        "rating" in test.columns,
        "Canonical test has no rating column",
    )

    catalog = np.sort(
        train["movieId"]
        .astype("int64")
        .unique()
    )

    _require(
        len(catalog) == EXPECTED_CATALOG_ITEMS,
        "Canonical train catalog size changed",
    )

    preferred = test.loc[
        test["rating"] >= PREFERENCE_THRESHOLD
    ].copy()

    nonpreferred = test.loc[
        test["rating"] < PREFERENCE_THRESHOLD
    ].copy()

    _require(
        len(preferred) == EXPECTED_PREFERRED_TEST_ROWS,
        "Preferred-test row count changed",
    )
    _require(
        len(nonpreferred)
        == EXPECTED_NONPREFERRED_TEST_ROWS,
        "Nonpreferred-test row count changed",
    )

    test_users = set(
        test["userId"]
        .astype(int)
        .tolist()
    )

    eligible_users = set(
        preferred["userId"]
        .astype(int)
        .tolist()
    )

    zero_preferred_users = (
        test_users - eligible_users
    )

    _require(
        len(eligible_users)
        == EXPECTED_ELIGIBLE_USERS,
        "Eligible preference user count changed",
    )
    _require(
        len(zero_preferred_users)
        == EXPECTED_ZERO_PREFERRED_USERS,
        "Zero-preferred user count changed",
    )

    preferred_items = set(
        preferred["movieId"]
        .astype(int)
        .tolist()
    )

    catalog_set = set(
        catalog.astype(int).tolist()
    )

    _require(
        len(preferred_items)
        == EXPECTED_UNIQUE_PREFERRED_ITEMS,
        "Unique preferred-test item count changed",
    )
    _require(
        preferred_items.issubset(catalog_set),
        "A preferred test item is outside the canonical train catalog",
    )

    final_fit = _final_fit_split(canonical)

    _require(
        len(final_fit.train)
        == EXPECTED_FINAL_FIT_ROWS,
        "Final train+validation row count changed",
    )
    _require(
        len(final_fit.validation) == 0,
        "Derived final-fit validation must be empty",
    )
    _require(
        len(final_fit.test) == EXPECTED_TEST_ROWS,
        "Derived final-fit test row count changed",
    )

    return {
        "catalog": catalog,
        "preferred": preferred,
        "nonpreferred": nonpreferred,
        "eligible_users": eligible_users,
        "zero_preferred_users": zero_preferred_users,
        "fit_split": final_fit,
    }


# =============================================================================
# Frozen model reconstruction
# =============================================================================


def _selected_hyperparameters(
    *,
    frozen_hp: dict[str, Any],
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    if model_name == "Popularity":
        return {}

    if model_name == "Random":
        return {
            "seed": int(seed),
            "namespace": "random_baseline",
        }

    source = (
        frozen_hp[
            "tuned_models"
        ][
            model_name
        ][
            "hyperparameters"
        ]
    )

    if not isinstance(source, dict):
        raise TypeError(
            f"{model_name}: frozen hyperparameters must be a mapping"
        )

    hp = deepcopy(source)

    # Step 55 freezes the selected scientific hyperparameters;
    # Step 56/58 replace the selection seed with each final replicate seed.
    hp["seed"] = int(seed)

    return hp


def _build_model(
    *,
    model_name: str,
    seed: int,
    fit_split: SplitResult,
    config: dict[str, Any],
    frozen_hp: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    hp = _selected_hyperparameters(
        frozen_hp=frozen_hp,
        model_name=model_name,
        seed=seed,
    )

    if model_name in BASELINE_KEY:
        model, baseline_hp = build_baseline(
            BASELINE_KEY[model_name],
            split=fit_split,
            config=config,
            seed=seed,
        )

        _require(
            baseline_hp == hp,
            (
                f"{model_name}: baseline constructor hyperparameters "
                "do not match frozen Step 58 identity"
            ),
        )

        return model, hp

    model = build_tuning_model(
        model_name=model_name,
        split=fit_split,
        hyperparameters=hp,
    )

    return model, hp


def _build_spec(
    *,
    model_name: str,
    seed: int,
    hyperparameters: dict[str, Any],
) -> ExperimentSpec:
    return ExperimentSpec(
        model_name=model_name,
        family=MODEL_FAMILY[model_name],
        variant="preference-r4",
        seed=seed,
        hyperparameters=hyperparameters,
    )


def _planned_matrix(
    *,
    model_filter: str,
    seed_filter: int | None,
) -> list[tuple[str, int]]:
    models: Iterable[str] = (
        MODEL_ORDER
        if model_filter == "all"
        else (model_filter,)
    )

    seeds: Iterable[int] = (
        SEED_ORDER
        if seed_filter is None
        else (seed_filter,)
    )

    return [
        (model_name, seed)
        for seed in seeds
        for model_name in models
    ]


# =============================================================================
# Preference-aware evaluator
# =============================================================================


def evaluate_preference_ranking(
    *,
    canonical: SplitResult,
    score_function: Any,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate test ranking with relevance restricted to rating >= 4."""
    if not callable(score_function):
        raise TypeError(
            "score_function must be callable"
        )

    catalog = np.sort(
        canonical.train["movieId"]
        .astype("int64")
        .unique()
    )

    train_by_user = _items_by_user(
        canonical.train
    )
    validation_by_user = _items_by_user(
        canonical.validation
    )

    preferred = canonical.test.loc[
        canonical.test["rating"]
        >= PREFERENCE_THRESHOLD
    ]

    relevant_by_user = _items_by_user(
        preferred
    )

    _require(
        len(relevant_by_user)
        == EXPECTED_ELIGIBLE_USERS,
        "Preference evaluator eligible-user count changed",
    )

    catalog_set = set(
        catalog.astype(int).tolist()
    )

    rows: list[dict[str, float | int]] = []

    max_k = max(K_VALUES)

    for user_id in sorted(
        relevant_by_user
    ):
        relevant = set(
            relevant_by_user[user_id]
        )

        if not relevant:
            raise AssertionError(
                "Eligible user unexpectedly has empty relevance set"
            )

        missing_from_catalog = (
            relevant - catalog_set
        )

        if missing_from_catalog:
            raise RuntimeError(
                f"User {user_id} has preferred test items outside "
                f"the train catalog: {sorted(missing_from_catalog)[:5]}"
            )

        excluded = set(
            train_by_user.get(
                user_id,
                frozenset(),
            )
        )

        excluded.update(
            validation_by_user.get(
                user_id,
                frozenset(),
            )
        )

        overlap = relevant & excluded

        if overlap:
            raise RuntimeError(
                f"User {user_id}: preferred test item overlaps "
                f"train/validation exclusions: {sorted(overlap)[:5]}"
            )

        candidates = _candidate_items(
            catalog=catalog,
            excluded_items=excluded,
        )

        candidate_set = set(
            candidates.astype(int).tolist()
        )

        missing_relevant = (
            relevant - candidate_set
        )

        if missing_relevant:
            raise RuntimeError(
                f"User {user_id}: preferred test item missing from "
                f"candidate set: {sorted(missing_relevant)[:5]}"
            )

        if len(candidates) < max_k:
            raise RuntimeError(
                f"User {user_id} has fewer than {max_k} candidates"
            )

        scores = np.asarray(
            score_function(
                int(user_id),
                candidates.copy(),
            ),
            dtype=np.float64,
        )

        if scores.shape != candidates.shape:
            raise RuntimeError(
                f"User {user_id}: score shape mismatch"
            )

        ranked = _deterministic_rank(
            candidates,
            scores,
        )

        top_ranked = ranked[:max_k]

        row: dict[str, float | int] = {
            "userId": int(user_id),
            "n_relevant": int(len(relevant)),
            "n_candidates": int(len(candidates)),
        }

        for k in K_VALUES:
            metrics = ranking_metrics_at_k(
                ranked_items=top_ranked,
                relevant_items=relevant,
                k=k,
            )

            row[f"precision_at_{k}"] = float(
                metrics.precision
            )
            row[f"recall_at_{k}"] = float(
                metrics.recall
            )
            row[f"hit_rate_at_{k}"] = float(
                metrics.hit_rate
            )
            row[f"ndcg_at_{k}"] = float(
                metrics.ndcg
            )
            row[f"mrr_at_{k}"] = float(
                metrics.mrr
            )
            row[f"map_at_{k}"] = float(
                metrics.map
            )

        rows.append(row)

    per_user = pd.DataFrame(rows)

    _require(
        len(per_user) == EXPECTED_ELIGIBLE_USERS,
        "Preference evaluator did not produce 5,851 user rows",
    )

    metric_columns = [
        column
        for column in per_user.columns
        if column not in {
            "userId",
            "n_relevant",
            "n_candidates",
        }
    ]

    aggregate = {
        column: float(
            per_user[column].mean()
        )
        for column in metric_columns
    }

    if not all(
        math.isfinite(value)
        for value in aggregate.values()
    ):
        raise RuntimeError(
            "Preference evaluator produced non-finite aggregate metrics"
        )

    return aggregate, per_user


# =============================================================================
# Result / resume handling
# =============================================================================


def _protocol_manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "freeze_id": "paper2-preference-sensitivity-v1",
        "preference_protocol_sha256": PREFERENCE_PROTOCOL_SHA256,
        "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
        "final_evaluation_sha256": FINAL_EVALUATION_SHA256,
        "run_id": RUN_ID,
        "models": list(MODEL_ORDER),
        "seeds": list(SEED_ORDER),
        "planned_experiments": EXPECTED_EXPERIMENTS,
        "fit_partitions": ["train", "validation"],
        "final_fit_rows": EXPECTED_FINAL_FIT_ROWS,
        "target_split": "test",
        "relevance_threshold": PREFERENCE_THRESHOLD,
        "eligible_users": EXPECTED_ELIGIBLE_USERS,
        "zero_preferred_users_excluded": EXPECTED_ZERO_PREFERRED_USERS,
        "secondary_analysis_only": True,
        "retuning_allowed": False,
        "primary_results_affected": False,
    }


def _result_metadata(
    *,
    model: Any,
    hyperparameters: dict[str, Any],
    per_user: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "analysis": "preference-aware-relevance-sensitivity",
        "preference_protocol_sha256": PREFERENCE_PROTOCOL_SHA256,
        "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
        "final_evaluation_sha256": FINAL_EVALUATION_SHA256,
        "primary_final_run_id": "final-ml1m-v1-1874aeba",
        "model_class": type(model).__name__,
        "hyperparameters": deepcopy(hyperparameters),
        "canonical_train_rows": EXPECTED_TRAIN_ROWS,
        "canonical_validation_rows": EXPECTED_VALIDATION_ROWS,
        "canonical_test_rows": EXPECTED_TEST_ROWS,
        "final_fit_rows": EXPECTED_FINAL_FIT_ROWS,
        "final_fit_partitions": ["train", "validation"],
        "candidate_catalog": "canonical_train_item_catalog",
        "catalog_size": EXPECTED_CATALOG_ITEMS,
        "test_candidate_exclusions": [
            "final_fit_train_interactions",
            "final_fit_validation_interactions",
        ],
        "relevance_definition": "test_rating >= 4.0",
        "relevance_threshold": PREFERENCE_THRESHOLD,
        "preferred_test_rows": EXPECTED_PREFERRED_TEST_ROWS,
        "nonpreferred_test_rows": EXPECTED_NONPREFERRED_TEST_ROWS,
        "eligible_users": EXPECTED_ELIGIBLE_USERS,
        "zero_preferred_users_excluded": EXPECTED_ZERO_PREFERRED_USERS,
        "users_evaluated": int(len(per_user)),
        "per_user_metric_rows": int(len(per_user)),
        "secondary_analysis_only": True,
        "post_test_retuning_allowed": False,
    }


def _per_user_path(
    run_dir: Path,
    experiment_id: str,
) -> Path:
    return (
        run_dir
        / "preference_per_user"
        / f"{experiment_id}.csv"
    )


def _verify_per_user_sidecar(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Completed preference result is missing sidecar: {path}"
        )

    frame = pd.read_csv(path)

    expected_columns = [
        "userId",
        "n_relevant",
        "n_candidates",
        *[
            f"{metric}_at_{k}"
            for k in K_VALUES
            for metric in (
                "precision",
                "recall",
                "hit_rate",
                "ndcg",
                "mrr",
                "map",
            )
        ],
    ]

    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"Preference sidecar schema mismatch: {path}"
        )

    if len(frame) != EXPECTED_ELIGIBLE_USERS:
        raise ValueError(
            f"Preference sidecar row count mismatch: {path}"
        )

    if frame["userId"].duplicated().any():
        raise ValueError(
            f"Preference sidecar contains duplicate users: {path}"
        )

    if (frame["n_relevant"] <= 0).any():
        raise ValueError(
            f"Preference sidecar contains empty relevance row: {path}"
        )

    numeric = frame.drop(
        columns=["userId"]
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(numeric).all():
        raise ValueError(
            f"Preference sidecar contains non-finite values: {path}"
        )

    return frame


def _results_root(config: dict[str, Any]) -> Path:
    paths = config.get("paths")

    if not isinstance(paths, dict):
        raise ValueError(
            "config.paths must be a mapping"
        )

    value = paths.get("results_root")

    if not isinstance(value, (str, Path)):
        raise ValueError(
            "config.paths.results_root is missing"
        )

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# =============================================================================
# Preflight / execution
# =============================================================================


def _print_preflight(
    *,
    matrix: list[tuple[str, int]],
) -> None:
    print("=" * 92)
    print("STEP 58 PREFERENCE-SENSITIVITY PREFLIGHT: PASS")
    print("=" * 92)
    print("run_id:", RUN_ID)
    print("preference_protocol_sha256:", PREFERENCE_PROTOCOL_SHA256)
    print("final_hyperparameters_sha256:", FINAL_HYPERPARAMETERS_SHA256)
    print("final_evaluation_sha256:", FINAL_EVALUATION_SHA256)
    print("canonical_train_rows:", EXPECTED_TRAIN_ROWS)
    print("canonical_validation_rows:", EXPECTED_VALIDATION_ROWS)
    print("final_fit_rows:", EXPECTED_FINAL_FIT_ROWS)
    print("canonical_test_rows:", EXPECTED_TEST_ROWS)
    print("catalog_items:", EXPECTED_CATALOG_ITEMS)
    print("preference_threshold:", PREFERENCE_THRESHOLD)
    print("preferred_test_rows:", EXPECTED_PREFERRED_TEST_ROWS)
    print("eligible_users:", EXPECTED_ELIGIBLE_USERS)
    print("zero_preferred_users_excluded:", EXPECTED_ZERO_PREFERRED_USERS)
    print("models:", len(MODEL_ORDER))
    print("seeds:", list(SEED_ORDER))
    print("frozen_planned_experiments:", EXPECTED_EXPERIMENTS)
    print("selected_experiments_this_invocation:", len(matrix))
    print("analysis_role: secondary_sensitivity_only")
    print()
    print("NO MODEL WAS FIT.")
    print("NO TEST SCORE WAS GENERATED.")


def run_preference_sensitivity(
    *,
    execute_preference_test: bool,
    model_filter: str = "all",
    seed_filter: int | None = None,
) -> tuple[int, int]:
    _, frozen_hp = _validate_frozen_files()

    config = load_yaml(
        PUBLICATION_CONFIG_PATH
    )

    if not isinstance(config, dict):
        raise TypeError(
            "publication.yaml root must be a mapping"
        )

    canonical = load_frozen_split(
        config,
        version=VERSION,
    )

    population = _preference_population(
        canonical
    )

    fit_split = population[
        "fit_split"
    ]

    matrix = _planned_matrix(
        model_filter=model_filter,
        seed_filter=seed_filter,
    )

    if not execute_preference_test:
        _print_preflight(
            matrix=matrix,
        )
        return (0, 0)

    run = prepare_run_context(
        config_path=PUBLICATION_CONFIG_PATH,
        config=config,
        version=VERSION,
        results_root=_results_root(config),
        run_id=RUN_ID,
    )

    _atomic_json(
        run.run_dir
        / "preference_protocol_manifest.json",
        _protocol_manifest_payload(),
    )

    completed = 0
    resumed = 0

    for model_name, seed in matrix:
        hyperparameters = _selected_hyperparameters(
            frozen_hp=frozen_hp,
            model_name=model_name,
            seed=seed,
        )

        spec = _build_spec(
            model_name=model_name,
            seed=seed,
            hyperparameters=hyperparameters,
        )

        experiment_id = experiment_id_for_spec(
            spec,
            target_split="test",
        )

        sidecar_path = _per_user_path(
            run.run_dir,
            experiment_id,
        )

        print()
        print("=" * 92)
        print("STEP 58 PREFERENCE-SENSITIVITY RUN")
        print("Model:", model_name)
        print("Seed:", seed)
        print("Fit rows:", EXPECTED_FINAL_FIT_ROWS)
        print("Target split: test")
        print("Relevance: rating >= 4.0")
        print("Eligible users:", EXPECTED_ELIGIBLE_USERS)
        print("Experiment ID:", experiment_id)
        print("=" * 92)

        existing = load_completed_experiment(
            run=run,
            spec=spec,
            target_split="test",
        )

        if existing is not None:
            result, artifacts = existing

            _verify_per_user_sidecar(
                sidecar_path
            )

            resumed += 1
            completed += 1

            print(
                "Resume: SKIPPED - verified completed "
                "preference-sensitivity experiment"
            )
            print("Status:", result.status)
            print(
                "ndcg_at_10 =",
                f"{result.ranking_metrics['ndcg_at_10']:.6f}",
            )
            print("Per-user metrics:", sidecar_path)
            print("Stored at:", artifacts.experiment_dir)
            continue

        model, reconstructed_hp = _build_model(
            model_name=model_name,
            seed=seed,
            fit_split=fit_split,
            config=config,
            frozen_hp=frozen_hp,
        )

        _require(
            reconstructed_hp == hyperparameters,
            (
                f"{model_name}: model reconstruction changed "
                "the frozen hyperparameter identity"
            ),
        )

        start = perf_counter()

        model.fit(
            fit_split.train
        )

        aggregate, per_user = (
            evaluate_preference_ranking(
                canonical=canonical,
                score_function=model.score,
            )
        )

        runtime_seconds = (
            perf_counter() - start
        )

        result = ExperimentResult(
            model_name=model_name,
            family=MODEL_FAMILY[model_name],
            variant="preference-r4",
            seed=seed,
            target_split="test",
            ranking_metrics=aggregate,
            native_metrics={},
            training_history=_training_history(model),
            runtime_seconds=float(runtime_seconds),
            status="completed",
            error_message=None,
            metadata=_result_metadata(
                model=model,
                hyperparameters=hyperparameters,
                per_user=per_user,
            ),
        )

        # Write the deterministic derived sidecar first.  The standard
        # checkpoint is sealed only after the sidecar is durable.
        _atomic_csv_replace(
            sidecar_path,
            per_user,
        )

        artifacts = write_experiment_result(
            run=run,
            result=result,
        )

        completed += 1

        print("Status:", result.status)
        print(
            "Runtime seconds:",
            f"{result.runtime_seconds:.3f}",
        )
        print(
            "ndcg_at_10 =",
            f"{result.ranking_metrics['ndcg_at_10']:.6f}",
        )
        print("Per-user metrics:", sidecar_path)
        print("Stored at:", artifacts.experiment_dir)

    print()
    print("=" * 92)
    print("STEP 58 PREFERENCE-SENSITIVITY INVOCATION COMPLETE")
    print("Run ID:", RUN_ID)
    print("Experiments selected:", len(matrix))
    print("Completed outcomes:", completed)
    print("Resumed outcomes:", resumed)
    print("Preference threshold:", PREFERENCE_THRESHOLD)
    print("Eligible users:", EXPECTED_ELIGIBLE_USERS)
    print("Analysis role: secondary sensitivity only")
    print("=" * 92)

    return completed, resumed


# =============================================================================
# CLI
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Step 58 rating>=4 "
            "preference-aware relevance sensitivity analysis."
        )
    )

    parser.add_argument(
        "--execute-preference-test",
        action="store_true",
        help=(
            "Fit the frozen final models and perform the prespecified "
            "secondary test-set sensitivity analysis. Without this flag "
            "the command is preflight only."
        ),
    )

    parser.add_argument(
        "--model",
        choices=("all", *MODEL_ORDER),
        default="all",
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=SEED_ORDER,
        default=None,
    )

    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    run_preference_sensitivity(
        execute_preference_test=bool(
            args.execute_preference_test
        ),
        model_filter=args.model,
        seed_filter=args.seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
