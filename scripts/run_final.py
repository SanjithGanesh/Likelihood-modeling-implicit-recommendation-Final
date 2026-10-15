"""Paper 2 final publication runner.

This runner implements the Step-56A frozen final-evaluation policy:

- hyperparameters are loaded only from config/final_hyperparameters_v1.yaml;
- final seeds are fixed at 2026, 2027, and 2028;
- models are refit from scratch on canonical train + validation;
- the canonical test partition remains the only held-out positive set;
- test ranking uses the original canonical split so candidate exclusions are
  exactly canonical train + canonical validation interactions;
- test evaluation is impossible unless --execute-final-test is supplied;
- completed final experiments are resume-safe through src.results.

Important implementation detail
-------------------------------
The ordinary run_experiment() path intentionally requires a non-empty
validation partition.  Final fitting, however, requires validation positives
to become training positives and therefore they must not remain in the
held-out mask passed to LogisticPCA / HPF / EF-EMB models.

This runner therefore uses two representations:

1. fit_split
   train      = canonical train + canonical validation
   validation = schema-preserving empty frame
   test       = canonical test

   This split is used ONLY for model construction and model fitting, so model
   held-out masks contain test only and negative/zero samplers still know every
   observed interaction.

2. canonical_split
   unchanged frozen train / validation / test

   This split is used ONLY by the shared evaluator.  Consequently, final test
   candidates exclude canonical train + validation interactions exactly as
   frozen in config/final_evaluation_v1.yaml.

No tuning or winner-selection code is executed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import pandas as pd
import yaml


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import scripts.tune_models as tune
from scripts.run_baselines import build_baseline
from scripts.run_mf import (
    load_frozen_split,
    prepare_run_context,
)
from src.data import load_yaml
from src.evaluation import (
    EvaluationResult,
    evaluate_from_config,
)
from src.experiment import (
    ExperimentResult,
    ExperimentSpec,
    _collect_native_metrics,
    _collect_training_history,
    _validate_model_contract,
    _validate_model_identity,
)
from src.results import (
    RunContext,
    experiment_id_for_spec,
    load_completed_experiment,
    semantic_yaml_hash,
    write_experiment_result,
)
from src.split import SplitResult


FINAL_HYPERPARAMETERS_PATH = Path(
    "config/final_hyperparameters_v1.yaml"
)

FINAL_EVALUATION_PATH = Path(
    "config/final_evaluation_v1.yaml"
)

EXPECTED_FINAL_HYPERPARAMETERS_SHA256 = (
    "9aae84dd59d7c34c82a67563139bdb27"
    "256874ba79c645c749cbcf788d72c981"
)

EXPECTED_FINAL_EVALUATION_SHA256 = (
    "1874aeba214b3a049b439fa4d6b8aab"
    "bb8c20e6aa98fdfbeb94c0bb6348002f8"
)

FINAL_MODEL_ORDER = [
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
]

TUNED_MODEL_ORDER = [
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
]

FINAL_MODEL_METADATA = {
    "Popularity": {
        "family": "baseline",
        "variant": "primary",
    },
    "Random": {
        "family": "baseline",
        "variant": "primary",
    },
    "PCA/SVD": {
        "family": "baseline",
        "variant": "primary",
    },
    "BernoulliMF": {
        "family": "bernoulli",
        "variant": "primary",
    },
    "PoissonMF": {
        "family": "poisson",
        "variant": "primary",
    },
    "GaussianMF": {
        "family": "gaussian",
        "variant": "primary",
    },
    "LogisticPCA": {
        "family": "bernoulli",
        "variant": "primary",
    },
    "HPF": {
        "family": "poisson",
        "variant": "primary",
    },
    "b-emb": {
        "family": "bernoulli",
        "variant": "primary",
    },
    "p-emb": {
        "family": "poisson",
        "variant": "primary",
    },
    "ap-emb": {
        "family": "poisson",
        "variant": "primary",
    },
    "g-emb": {
        "family": "gaussian",
        "variant": "primary",
    },
    "ng-emb": {
        "family": "gaussian",
        "variant": "primary",
    },
}

DEFAULT_FINAL_RUN_ID = (
    "final-ml1m-v1-"
    + EXPECTED_FINAL_EVALUATION_SHA256[:8]
)

FINAL_PROTOCOL_MANIFEST_FILENAME = (
    "final_protocol_manifest.json"
)


def _resolve_project_path(
    path: str | Path,
) -> Path:
    path = Path(
        str(path).replace(
            "\\",
            "/",
        )
    )

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _sha256_file(
    path: str | Path,
) -> str:
    path = _resolve_project_path(
        path
    )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def _load_yaml_mapping(
    path: str | Path,
) -> dict[str, Any]:
    path = _resolve_project_path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"YAML file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = yaml.safe_load(
            file
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"YAML root must be a mapping: {path}"
        )

    return payload


def _require_mapping(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(
            f"{name} must be a mapping"
        )

    return value


def _verify_raw_reference(
    mapping: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    label: str,
) -> None:
    raw_path = mapping.get(
        path_key
    )

    expected_hash = mapping.get(
        hash_key
    )

    if (
        not isinstance(
            raw_path,
            str,
        )
        or not raw_path.strip()
    ):
        raise ValueError(
            f"{label}.{path_key} must be a nonempty path"
        )

    if (
        not isinstance(
            expected_hash,
            str,
        )
        or len(expected_hash) != 64
    ):
        raise ValueError(
            f"{label}.{hash_key} must be a SHA-256 string"
        )

    actual_hash = _sha256_file(
        raw_path
    )

    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} reference hash mismatch for {raw_path}: "
            f"expected={expected_hash}, actual={actual_hash}"
        )


def verify_frozen_controls(
    *,
    config_path: str | Path = "config/publication.yaml",
    final_hyperparameters_path: str | Path = FINAL_HYPERPARAMETERS_PATH,
    final_evaluation_path: str | Path = FINAL_EVALUATION_PATH,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Verify the two Step-55/56A freeze artifacts before any final work."""

    config_path = _resolve_project_path(
        config_path
    )

    final_hyperparameters_path = (
        _resolve_project_path(
            final_hyperparameters_path
        )
    )

    final_evaluation_path = (
        _resolve_project_path(
            final_evaluation_path
        )
    )

    hp_hash = _sha256_file(
        final_hyperparameters_path
    )

    eval_hash = _sha256_file(
        final_evaluation_path
    )

    if (
        hp_hash
        !=
        EXPECTED_FINAL_HYPERPARAMETERS_SHA256
    ):
        raise ValueError(
            "Frozen final-hyperparameter SHA-256 mismatch: "
            f"expected={EXPECTED_FINAL_HYPERPARAMETERS_SHA256}, "
            f"actual={hp_hash}"
        )

    if (
        eval_hash
        !=
        EXPECTED_FINAL_EVALUATION_SHA256
    ):
        raise ValueError(
            "Frozen final-evaluation SHA-256 mismatch: "
            f"expected={EXPECTED_FINAL_EVALUATION_SHA256}, "
            f"actual={eval_hash}"
        )

    config = load_yaml(
        config_path
    )

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "Publication configuration must be a mapping"
        )

    frozen_hp = _load_yaml_mapping(
        final_hyperparameters_path
    )

    frozen_eval = _load_yaml_mapping(
        final_evaluation_path
    )

    if (
        frozen_hp.get("status")
        !=
        "frozen"
    ):
        raise ValueError(
            "Final hyperparameters are not marked frozen"
        )

    if (
        frozen_eval.get("status")
        !=
        "frozen"
    ):
        raise ValueError(
            "Final evaluation protocol is not marked frozen"
        )

    if (
        frozen_hp.get("freeze_id")
        !=
        "paper2-final-hyperparameters-v1"
    ):
        raise ValueError(
            "Unexpected final-hyperparameter freeze_id"
        )

    if (
        frozen_eval.get("freeze_id")
        !=
        "paper2-final-evaluation-v1"
    ):
        raise ValueError(
            "Unexpected final-evaluation freeze_id"
        )

    hp_dataset = _require_mapping(
        frozen_hp.get("dataset"),
        name="final_hyperparameters.dataset",
    )

    eval_dataset = _require_mapping(
        frozen_eval.get("dataset"),
        name="final_evaluation.dataset",
    )

    config_dataset = _require_mapping(
        config.get("dataset"),
        name="publication.dataset",
    )

    if (
        hp_dataset.get("id")
        !=
        "ml-1m"
        or eval_dataset.get("id")
        !=
        "ml-1m"
        or config_dataset.get("id")
        !=
        "ml-1m"
    ):
        raise ValueError(
            "Final runner is frozen to dataset ml-1m"
        )

    if (
        hp_dataset.get("artifact_version")
        !=
        "v1"
        or eval_dataset.get("artifact_version")
        !=
        "v1"
    ):
        raise ValueError(
            "Final runner is frozen to ML-1M artifact version v1"
        )

    if (
        hp_dataset.get(
            "processed_dataset_sha256"
        )
        !=
        eval_dataset.get(
            "processed_dataset_sha256"
        )
    ):
        raise ValueError(
            "Frozen processed-dataset hashes disagree"
        )

    if (
        hp_dataset.get(
            "split_sha256"
        )
        !=
        eval_dataset.get(
            "canonical_split_sha256"
        )
    ):
        raise ValueError(
            "Frozen canonical-split hashes disagree"
        )

    source_freezes = _require_mapping(
        frozen_eval.get(
            "source_freezes"
        ),
        name="final_evaluation.source_freezes",
    )

    if (
        source_freezes.get(
            "final_hyperparameters_sha256"
        )
        !=
        EXPECTED_FINAL_HYPERPARAMETERS_SHA256
    ):
        raise ValueError(
            "Final-evaluation artifact does not reference the "
            "expected final-hyperparameter freeze"
        )

    scientific_provenance = _require_mapping(
        frozen_hp.get(
            "scientific_provenance"
        ),
        name="final_hyperparameters.scientific_provenance",
    )

    publication_config_semantic_hash = (
        semantic_yaml_hash(
            config
        )
    )

    expected_config_hashes = {
        scientific_provenance.get(
            "publication_config_semantic_sha256"
        ),
        source_freezes.get(
            "publication_config_semantic_sha256"
        ),
    }

    if expected_config_hashes != {
        publication_config_semantic_hash
    }:
        raise ValueError(
            "Current publication.yaml semantic hash does not match "
            "both frozen control artifacts"
        )

    _verify_raw_reference(
        source_freezes,
        path_key="model_spec_path",
        hash_key="model_spec_sha256",
        label="final_evaluation.source_freezes",
    )

    _verify_raw_reference(
        source_freezes,
        path_key="publication_ground_truth_path",
        hash_key="publication_ground_truth_sha256",
        label="final_evaluation.source_freezes",
    )

    _verify_raw_reference(
        scientific_provenance,
        path_key="tuning_space_path",
        hash_key="tuning_space_sha256",
        label="final_hyperparameters.scientific_provenance",
    )

    _verify_raw_reference(
        scientific_provenance,
        path_key="tuning_summary_path",
        hash_key="tuning_summary_sha256",
        label="final_hyperparameters.scientific_provenance",
    )

    selection_policy = _require_mapping(
        frozen_hp.get(
            "selection_policy"
        ),
        name="final_hyperparameters.selection_policy",
    )

    if (
        selection_policy.get(
            "selection_split"
        )
        !=
        "validation"
        or selection_policy.get(
            "total_validation_trials"
        )
        !=
        102
        or selection_policy.get(
            "test_access_during_selection"
        )
        is not False
        or selection_policy.get(
            "hyperparameter_retuning_after_freeze"
        )
        is not False
    ):
        raise ValueError(
            "Frozen hyperparameter-selection policy is inconsistent "
            "with the completed Step-55 freeze"
        )

    hp_final_eval = _require_mapping(
        frozen_hp.get(
            "final_evaluation"
        ),
        name="final_hyperparameters.final_evaluation",
    )

    training_protocol = _require_mapping(
        frozen_eval.get(
            "training_protocol"
        ),
        name="final_evaluation.training_protocol",
    )

    evaluation_protocol = _require_mapping(
        frozen_eval.get(
            "evaluation_protocol"
        ),
        name="final_evaluation.evaluation_protocol",
    )

    seed_protocol = _require_mapping(
        frozen_eval.get(
            "seed_protocol"
        ),
        name="final_evaluation.seed_protocol",
    )

    model_inventory = _require_mapping(
        frozen_eval.get(
            "model_inventory"
        ),
        name="final_evaluation.model_inventory",
    )

    test_lock = _require_mapping(
        frozen_eval.get(
            "test_lock"
        ),
        name="final_evaluation.test_lock",
    )

    expected_seeds = [
        2026,
        2027,
        2028,
    ]

    if (
        hp_final_eval.get(
            "target_split"
        )
        !=
        "test"
        or hp_final_eval.get(
            "final_seeds"
        )
        !=
        expected_seeds
        or seed_protocol.get(
            "final_seeds"
        )
        !=
        expected_seeds
    ):
        raise ValueError(
            "Frozen final-seed or target-split policy mismatch"
        )

    if (
        training_protocol.get(
            "policy"
        )
        !=
        "refit_from_scratch_on_train_plus_validation"
        or training_protocol.get(
            "train_rows"
        )
        !=
        805443
        or training_protocol.get(
            "validation_rows"
        )
        !=
        97383
        or training_protocol.get(
            "final_fit_rows"
        )
        !=
        902826
        or training_protocol.get(
            "final_fit_partitions"
        )
        !=
        [
            "train",
            "validation",
        ]
        or training_protocol.get(
            "test_positive_training_signal"
        )
        is not False
        or training_protocol.get(
            "test_ratings_used_for_training"
        )
        is not False
        or training_protocol.get(
            "test_pair_identity_for_negative_masking_only"
        )
        is not True
        or training_protocol.get(
            "hyperparameter_retuning"
        )
        is not False
    ):
        raise ValueError(
            "Frozen final-training policy mismatch"
        )

    if (
        evaluation_protocol.get(
            "target_split"
        )
        !=
        "test"
        or evaluation_protocol.get(
            "test_rows"
        )
        !=
        97383
        or evaluation_protocol.get(
            "ranking_mode"
        )
        !=
        "full_catalog"
        or evaluation_protocol.get(
            "candidate_catalog"
        )
        !=
        "canonical_train_item_catalog"
        or evaluation_protocol.get(
            "exclude_from_test_candidates"
        )
        !=
        [
            "final_fit_train_interactions",
            "final_fit_validation_interactions",
        ]
        or evaluation_protocol.get(
            "retain_test_positives_in_candidates"
        )
        is not True
        or evaluation_protocol.get(
            "tie_break"
        )
        !=
        "ascending_item_id"
        or evaluation_protocol.get(
            "k_values"
        )
        !=
        [
            5,
            10,
            20,
        ]
        or evaluation_protocol.get(
            "common_primary_reporting_metric"
        )
        !=
        "ndcg_at_10"
    ):
        raise ValueError(
            "Frozen final-evaluation policy mismatch"
        )

    if (
        model_inventory.get(
            "models"
        )
        !=
        FINAL_MODEL_ORDER
        or model_inventory.get(
            "total_models"
        )
        !=
        13
        or model_inventory.get(
            "tuned_models"
        )
        !=
        11
        or model_inventory.get(
            "untuned_baselines"
        )
        !=
        2
    ):
        raise ValueError(
            "Frozen final model inventory mismatch"
        )

    tuned_models = _require_mapping(
        frozen_hp.get(
            "tuned_models"
        ),
        name="final_hyperparameters.tuned_models",
    )

    untuned_baselines = _require_mapping(
        frozen_hp.get(
            "untuned_baselines"
        ),
        name="final_hyperparameters.untuned_baselines",
    )

    if list(
        tuned_models.keys()
    ) != TUNED_MODEL_ORDER:
        raise ValueError(
            "Frozen tuned-model order/inventory mismatch"
        )

    if list(
        untuned_baselines.keys()
    ) != [
        "Popularity",
        "Random",
    ]:
        raise ValueError(
            "Frozen untuned-baseline inventory mismatch"
        )

    if (
        test_lock.get(
            "test_metrics_seen_before_protocol_freeze"
        )
        is not False
        or test_lock.get(
            "test_metrics_may_not_trigger_retuning"
        )
        is not True
        or test_lock.get(
            "test_metrics_may_not_trigger_model_redefinition"
        )
        is not True
        or test_lock.get(
            "test_metrics_may_not_trigger_seed_selection"
        )
        is not True
        or test_lock.get(
            "single_final_protocol"
        )
        is not True
    ):
        raise ValueError(
            "Frozen test-lock policy mismatch"
        )

    return (
        config,
        frozen_hp,
        frozen_eval,
    )


def build_final_fit_split(
    canonical_split: SplitResult,
    frozen_evaluation: Mapping[str, Any],
) -> SplitResult:
    """Create the model-construction/fitting representation for Step 56."""

    training_protocol = _require_mapping(
        frozen_evaluation.get(
            "training_protocol"
        ),
        name="final_evaluation.training_protocol",
    )

    evaluation_protocol = _require_mapping(
        frozen_evaluation.get(
            "evaluation_protocol"
        ),
        name="final_evaluation.evaluation_protocol",
    )

    expected_train_rows = int(
        training_protocol[
            "train_rows"
        ]
    )

    expected_validation_rows = int(
        training_protocol[
            "validation_rows"
        ]
    )

    expected_final_fit_rows = int(
        training_protocol[
            "final_fit_rows"
        ]
    )

    expected_test_rows = int(
        evaluation_protocol[
            "test_rows"
        ]
    )

    actual_counts = {
        "train": len(
            canonical_split.train
        ),
        "validation": len(
            canonical_split.validation
        ),
        "test": len(
            canonical_split.test
        ),
    }

    expected_counts = {
        "train": expected_train_rows,
        "validation": expected_validation_rows,
        "test": expected_test_rows,
    }

    if actual_counts != expected_counts:
        raise ValueError(
            "Canonical split row counts do not match the "
            f"frozen final protocol: expected={expected_counts}, "
            f"actual={actual_counts}"
        )

    final_train = pd.concat(
        [
            canonical_split.train,
            canonical_split.validation,
        ],
        ignore_index=True,
    )

    if len(
        final_train
    ) != expected_final_fit_rows:
        raise ValueError(
            "Final-fit row count does not match frozen protocol"
        )

    pair_columns = [
        "userId",
        "movieId",
    ]

    if final_train.duplicated(
        subset=pair_columns
    ).any():
        raise ValueError(
            "Final-fit train+validation contains duplicate user-item pairs"
        )

    test_pairs = canonical_split.test[
        pair_columns
    ]

    if test_pairs.duplicated(
        subset=pair_columns
    ).any():
        raise ValueError(
            "Canonical test contains duplicate user-item pairs"
        )

    final_pairs = pd.MultiIndex.from_frame(
        final_train[
            pair_columns
        ]
    )

    heldout_pairs = pd.MultiIndex.from_frame(
        test_pairs
    )

    overlap = final_pairs.intersection(
        heldout_pairs
    )

    if len(
        overlap
    ) != 0:
        raise ValueError(
            "Final-fit interactions overlap canonical test interactions"
        )

    canonical_catalog = set(
        canonical_split.train[
            "movieId"
        ]
        .astype(
            "int64"
        )
        .unique()
        .tolist()
    )

    final_fit_catalog = set(
        final_train[
            "movieId"
        ]
        .astype(
            "int64"
        )
        .unique()
        .tolist()
    )

    if (
        final_fit_catalog
        !=
        canonical_catalog
    ):
        raise ValueError(
            "Final-fit catalog differs from the frozen canonical "
            "training catalog; this would violate the Step-56A "
            "candidate-catalog policy"
        )

    test_catalog = set(
        canonical_split.test[
            "movieId"
        ]
        .astype(
            "int64"
        )
        .unique()
        .tolist()
    )

    missing_test_items = (
        test_catalog
        -
        canonical_catalog
    )

    if missing_test_items:
        raise ValueError(
            "Canonical test contains items absent from the "
            "canonical training catalog"
        )

    empty_validation = (
        canonical_split.validation
        .iloc[
            0:0
        ]
        .copy()
    )

    return SplitResult(
        train=final_train,
        validation=empty_validation,
        test=(
            canonical_split.test.copy()
        ),
        seed=canonical_split.seed,
        attempt=canonical_split.attempt,
        enforce_train_item_coverage=(
            canonical_split.enforce_train_item_coverage
        ),
    )


def resolve_models(
    requested: str,
) -> list[str]:
    if requested == "all":
        return list(
            FINAL_MODEL_ORDER
        )

    if requested not in FINAL_MODEL_ORDER:
        raise ValueError(
            f"Unknown final model: {requested}"
        )

    return [
        requested
    ]


def resolve_seeds(
    requested: str,
    frozen_evaluation: Mapping[str, Any],
) -> list[int]:
    seed_protocol = _require_mapping(
        frozen_evaluation.get(
            "seed_protocol"
        ),
        name="final_evaluation.seed_protocol",
    )

    frozen_seeds = [
        int(seed)
        for seed
        in seed_protocol[
            "final_seeds"
        ]
    ]

    if requested == "all":
        return frozen_seeds

    try:
        seed = int(
            requested
        )
    except ValueError as exc:
        raise ValueError(
            "Final seed must be 'all' or one of the frozen integer seeds"
        ) from exc

    if seed not in frozen_seeds:
        raise ValueError(
            f"Seed {seed} is not in the frozen final seed set "
            f"{frozen_seeds}"
        )

    return [
        seed
    ]


def materialize_final_hyperparameters(
    *,
    model_name: str,
    seed: int,
    frozen_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize exactly one frozen model configuration for one final seed."""

    if model_name == "Popularity":
        return {}

    if model_name == "Random":
        return {
            "seed": int(
                seed
            ),
            "namespace": (
                "random_baseline"
            ),
        }

    tuned_models = _require_mapping(
        frozen_hyperparameters.get(
            "tuned_models"
        ),
        name="final_hyperparameters.tuned_models",
    )

    model_record = _require_mapping(
        tuned_models.get(
            model_name
        ),
        name=(
            "final_hyperparameters.tuned_models."
            f"{model_name}"
        ),
    )

    frozen = _require_mapping(
        model_record.get(
            "hyperparameters"
        ),
        name=(
            "final_hyperparameters.tuned_models."
            f"{model_name}.hyperparameters"
        ),
    )

    hyperparameters = deepcopy(
        dict(
            frozen
        )
    )

    # Step 55 freezes all non-seed settings.  Step 56A explicitly freezes
    # replacement of the selection seed by each final seed.
    hyperparameters[
        "seed"
    ] = int(
        seed
    )

    return hyperparameters


def build_final_model_and_spec(
    *,
    model_name: str,
    seed: int,
    fit_split: SplitResult,
    config: dict[str, Any],
    frozen_hyperparameters: Mapping[str, Any],
):
    """Build one final model without invoking any tuning/selection logic."""

    hyperparameters = (
        materialize_final_hyperparameters(
            model_name=model_name,
            seed=seed,
            frozen_hyperparameters=(
                frozen_hyperparameters
            ),
        )
    )

    if model_name == "Popularity":
        model, built_hyperparameters = (
            build_baseline(
                "popularity",
                split=fit_split,
                config=config,
                seed=seed,
            )
        )

    elif model_name == "Random":
        model, built_hyperparameters = (
            build_baseline(
                "random",
                split=fit_split,
                config=config,
                seed=seed,
            )
        )

    else:
        model = tune.build_tuning_model(
            model_name=model_name,
            split=fit_split,
            hyperparameters=(
                hyperparameters
            ),
        )

        built_hyperparameters = (
            hyperparameters
        )

    if (
        built_hyperparameters
        !=
        hyperparameters
    ):
        raise RuntimeError(
            f"Builder hyperparameters differ from frozen final "
            f"hyperparameters for {model_name}"
        )

    metadata = (
        FINAL_MODEL_METADATA[
            model_name
        ]
    )

    spec = ExperimentSpec(
        model_name=model_name,
        family=metadata[
            "family"
        ],
        variant=metadata[
            "variant"
        ],
        seed=int(
            seed
        ),
        hyperparameters=(
            hyperparameters
        ),
    )

    _validate_model_contract(
        model
    )

    _validate_model_identity(
        model,
        spec,
    )

    return (
        model,
        spec,
    )


def _final_result_metadata(
    *,
    model: Any,
    spec: ExperimentSpec,
    evaluation: EvaluationResult,
    canonical_split: SplitResult,
    fit_split: SplitResult,
    frozen_hyperparameters: Mapping[str, Any],
    frozen_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    tuned_models = _require_mapping(
        frozen_hyperparameters.get(
            "tuned_models"
        ),
        name="final_hyperparameters.tuned_models",
    )

    selection_record = (
        tuned_models.get(
            spec.model_name
        )
    )

    selected_experiment_id = None
    selected_stage = None

    if isinstance(
        selection_record,
        Mapping,
    ):
        selected_experiment_id = (
            selection_record.get(
                "selected_experiment_id"
            )
        )

        selected_stage = (
            selection_record.get(
                "selected_stage"
            )
        )

    return {
        "model_class": (
            f"{model.__class__.__module__}."
            f"{model.__class__.__name__}"
        ),
        "hyperparameters": deepcopy(
            spec.hyperparameters
        ),
        "catalog_size": int(
            evaluation.catalog_size
        ),
        "users_evaluated": int(
            evaluation.users_evaluated
        ),
        "per_user_metric_rows": int(
            len(
                evaluation.per_user
            )
        ),
        "split_seed": int(
            canonical_split.seed
        ),
        "split_attempt": int(
            canonical_split.attempt
        ),
        "enforce_train_item_coverage": bool(
            canonical_split.enforce_train_item_coverage
        ),
        # Existing generic row-count keys describe the effective fit split.
        "train_rows": int(
            len(
                fit_split.train
            )
        ),
        "validation_rows": int(
            len(
                fit_split.validation
            )
        ),
        "test_rows": int(
            len(
                canonical_split.test
            )
        ),
        # Explicit canonical/final accounting avoids ambiguity.
        "canonical_train_rows": int(
            len(
                canonical_split.train
            )
        ),
        "canonical_validation_rows": int(
            len(
                canonical_split.validation
            )
        ),
        "canonical_test_rows": int(
            len(
                canonical_split.test
            )
        ),
        "final_fit_rows": int(
            len(
                fit_split.train
            )
        ),
        "final_fit_partitions": [
            "train",
            "validation",
        ],
        "model_construction_heldout_partitions": [
            "test",
        ],
        "evaluation_split_representation": (
            "original_canonical_train_validation_test"
        ),
        "candidate_catalog": (
            "canonical_train_item_catalog"
        ),
        "test_candidate_exclusions": [
            "canonical_train_interactions",
            "canonical_validation_interactions",
        ],
        "final_hyperparameters_freeze_id": (
            frozen_hyperparameters[
                "freeze_id"
            ]
        ),
        "final_hyperparameters_sha256": (
            EXPECTED_FINAL_HYPERPARAMETERS_SHA256
        ),
        "final_evaluation_freeze_id": (
            frozen_evaluation[
                "freeze_id"
            ]
        ),
        "final_evaluation_sha256": (
            EXPECTED_FINAL_EVALUATION_SHA256
        ),
        "source_selected_experiment_id": (
            selected_experiment_id
        ),
        "source_selected_stage": (
            selected_stage
        ),
        "post_test_retuning_allowed": False,
    }


def run_one_final_experiment(
    *,
    model: Any,
    spec: ExperimentSpec,
    fit_split: SplitResult,
    canonical_split: SplitResult,
    config: dict[str, Any],
    frozen_hyperparameters: Mapping[str, Any],
    frozen_evaluation: Mapping[str, Any],
    authorize_test: bool,
) -> ExperimentResult:
    """Fit one frozen model on train+validation and evaluate canonical test."""

    if not isinstance(
        authorize_test,
        bool,
    ):
        raise TypeError(
            "authorize_test must be boolean"
        )

    # Hard gate before fit/evaluation.
    if not authorize_test:
        raise PermissionError(
            "Final test execution is locked. "
            "Use the explicit final-test authorization path only after "
            "Step 56B/56C preflight is complete."
        )

    _validate_model_contract(
        model
    )

    _validate_model_identity(
        model,
        spec,
    )

    if len(
        fit_split.validation
    ) != 0:
        raise ValueError(
            "Final model-construction split must have empty validation"
        )

    start_time = perf_counter()

    model.fit(
        fit_split.train
    )

    # Evaluation intentionally uses the unchanged canonical split:
    # test candidates therefore exclude canonical train + validation.
    evaluation = evaluate_from_config(
        split=canonical_split,
        config=config,
        score_function=model.score,
        target_split="test",
        allow_test=True,
    )

    if not isinstance(
        evaluation,
        EvaluationResult,
    ):
        raise TypeError(
            "evaluate_from_config() did not return EvaluationResult"
        )

    if (
        evaluation.target_split
        !=
        "test"
    ):
        raise RuntimeError(
            "Final evaluator returned a non-test target"
        )

    native_metrics = (
        _collect_native_metrics(
            model,
            canonical_split.test,
        )
    )

    training_history = (
        _collect_training_history(
            model
        )
    )

    runtime_seconds = (
        perf_counter()
        -
        start_time
    )

    metadata = (
        _final_result_metadata(
            model=model,
            spec=spec,
            evaluation=evaluation,
            canonical_split=(
                canonical_split
            ),
            fit_split=fit_split,
            frozen_hyperparameters=(
                frozen_hyperparameters
            ),
            frozen_evaluation=(
                frozen_evaluation
            ),
        )
    )

    return ExperimentResult(
        model_name=spec.model_name,
        family=spec.family,
        variant=spec.variant,
        seed=spec.seed,
        target_split="test",
        ranking_metrics={
            key: float(value)
            for key, value
            in evaluation.aggregate.items()
        },
        native_metrics=(
            native_metrics
        ),
        training_history=(
            training_history
        ),
        runtime_seconds=float(
            runtime_seconds
        ),
        status="completed",
        error_message=None,
        metadata=metadata,
    )


def _final_protocol_manifest_payload(
    *,
    run: RunContext,
    frozen_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "purpose": (
            "Paper 2 final publication evaluation"
        ),
        "final_hyperparameters_sha256": (
            EXPECTED_FINAL_HYPERPARAMETERS_SHA256
        ),
        "final_evaluation_sha256": (
            EXPECTED_FINAL_EVALUATION_SHA256
        ),
        "final_evaluation_freeze_id": (
            frozen_evaluation[
                "freeze_id"
            ]
        ),
        "dataset_id": (
            run.provenance.dataset_id
        ),
        "artifact_version": (
            run.provenance.artifact_version
        ),
        "dataset_sha256": (
            run.provenance.dataset_hash
        ),
        "canonical_split_sha256": (
            run.provenance.split_hash
        ),
        "publication_config_semantic_sha256": (
            run.provenance.config_hash
        ),
        "final_runner_code_version": (
            run.provenance.code_version
        ),
        "final_fit_policy": (
            "refit_from_scratch_on_train_plus_validation"
        ),
        "final_fit_rows": 902826,
        "test_rows": 97383,
        "target_split": "test",
        "models": list(
            FINAL_MODEL_ORDER
        ),
        "seeds": [
            2026,
            2027,
            2028,
        ],
        "planned_experiments": (
            len(
                FINAL_MODEL_ORDER
            )
            *
            3
        ),
        "post_test_retuning_allowed": False,
    }


def ensure_final_protocol_manifest(
    *,
    run: RunContext,
    frozen_evaluation: Mapping[str, Any],
) -> Path:
    path = (
        run.run_dir
        /
        FINAL_PROTOCOL_MANIFEST_FILENAME
    )

    payload = (
        _final_protocol_manifest_payload(
            run=run,
            frozen_evaluation=(
                frozen_evaluation
            ),
        )
    )

    if path.exists():
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            existing = json.load(
                file
            )

        if existing != payload:
            raise ValueError(
                "Existing final protocol manifest does not match "
                "the current frozen final protocol"
            )

        return path

    temp = path.with_name(
        f".{path.name}.tmp"
    )

    if temp.exists():
        temp.unlink()

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temp.replace(
        path
    )

    return path


def _assert_run_provenance_matches_freezes(
    *,
    run: RunContext,
    frozen_hyperparameters: Mapping[str, Any],
    frozen_evaluation: Mapping[str, Any],
) -> None:
    hp_dataset = _require_mapping(
        frozen_hyperparameters[
            "dataset"
        ],
        name="final_hyperparameters.dataset",
    )

    eval_dataset = _require_mapping(
        frozen_evaluation[
            "dataset"
        ],
        name="final_evaluation.dataset",
    )

    source_freezes = _require_mapping(
        frozen_evaluation[
            "source_freezes"
        ],
        name="final_evaluation.source_freezes",
    )

    if (
        run.provenance.dataset_id
        !=
        hp_dataset[
            "id"
        ]
        or run.provenance.dataset_id
        !=
        eval_dataset[
            "id"
        ]
    ):
        raise ValueError(
            "Run provenance dataset ID does not match frozen controls"
        )

    if (
        run.provenance.artifact_version
        !=
        hp_dataset[
            "artifact_version"
        ]
        or run.provenance.artifact_version
        !=
        eval_dataset[
            "artifact_version"
        ]
    ):
        raise ValueError(
            "Run provenance artifact version does not match frozen controls"
        )

    if (
        run.provenance.dataset_hash
        !=
        hp_dataset[
            "processed_dataset_sha256"
        ]
        or run.provenance.dataset_hash
        !=
        eval_dataset[
            "processed_dataset_sha256"
        ]
    ):
        raise ValueError(
            "Run provenance dataset hash does not match frozen controls"
        )

    if (
        run.provenance.split_hash
        !=
        hp_dataset[
            "split_sha256"
        ]
        or run.provenance.split_hash
        !=
        eval_dataset[
            "canonical_split_sha256"
        ]
    ):
        raise ValueError(
            "Run provenance split hash does not match frozen controls"
        )

    if (
        run.provenance.config_hash
        !=
        source_freezes[
            "publication_config_semantic_sha256"
        ]
    ):
        raise ValueError(
            "Run provenance config hash does not match frozen controls"
        )


def _prepare_final_state(
    *,
    config_path: str | Path,
    version: str,
    final_hyperparameters_path: str | Path,
    final_evaluation_path: str | Path,
    results_root: str | Path | None,
    run_id: str | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    SplitResult,
    SplitResult,
    RunContext,
]:
    (
        config,
        frozen_hp,
        frozen_eval,
    ) = verify_frozen_controls(
        config_path=config_path,
        final_hyperparameters_path=(
            final_hyperparameters_path
        ),
        final_evaluation_path=(
            final_evaluation_path
        ),
    )

    canonical_split = (
        load_frozen_split(
            config,
            version=version,
        )
    )

    fit_split = (
        build_final_fit_split(
            canonical_split,
            frozen_eval,
        )
    )

    if results_root is None:
        paths = _require_mapping(
            config.get(
                "paths"
            ),
            name="publication.paths",
        )

        results_root = paths.get(
            "results_root"
        )

    if not isinstance(
        results_root,
        (
            str,
            Path,
        ),
    ):
        raise ValueError(
            "results_root must be configured or explicitly supplied"
        )

    run = prepare_run_context(
        config_path=config_path,
        config=config,
        version=version,
        results_root=results_root,
        run_id=run_id,
    )

    _assert_run_provenance_matches_freezes(
        run=run,
        frozen_hyperparameters=(
            frozen_hp
        ),
        frozen_evaluation=(
            frozen_eval
        ),
    )

    ensure_final_protocol_manifest(
        run=run,
        frozen_evaluation=(
            frozen_eval
        ),
    )

    return (
        config,
        frozen_hp,
        frozen_eval,
        canonical_split,
        fit_split,
        run,
    )


def preflight_final_protocol(
    *,
    config_path: str | Path = "config/publication.yaml",
    version: str = "v1",
    final_hyperparameters_path: str | Path = FINAL_HYPERPARAMETERS_PATH,
    final_evaluation_path: str | Path = FINAL_EVALUATION_PATH,
    results_root: str | Path | None = None,
    run_id: str | None = DEFAULT_FINAL_RUN_ID,
) -> dict[str, Any]:
    """Validate final protocol without fitting or evaluating any model."""

    (
        _config,
        frozen_hp,
        frozen_eval,
        canonical_split,
        fit_split,
        run,
    ) = _prepare_final_state(
        config_path=config_path,
        version=version,
        final_hyperparameters_path=(
            final_hyperparameters_path
        ),
        final_evaluation_path=(
            final_evaluation_path
        ),
        results_root=results_root,
        run_id=run_id,
    )

    seeds = resolve_seeds(
        "all",
        frozen_eval,
    )

    experiment_ids = []

    for model_name in FINAL_MODEL_ORDER:
        for seed in seeds:
            hp = (
                materialize_final_hyperparameters(
                    model_name=model_name,
                    seed=seed,
                    frozen_hyperparameters=(
                        frozen_hp
                    ),
                )
            )

            metadata = (
                FINAL_MODEL_METADATA[
                    model_name
                ]
            )

            spec = ExperimentSpec(
                model_name=model_name,
                family=metadata[
                    "family"
                ],
                variant=metadata[
                    "variant"
                ],
                seed=seed,
                hyperparameters=hp,
            )

            experiment_ids.append(
                experiment_id_for_spec(
                    spec,
                    target_split="test",
                )
            )

    if len(
        experiment_ids
    ) != 39:
        raise RuntimeError(
            "Final protocol must plan exactly 39 model-seed experiments"
        )

    if len(
        set(
            experiment_ids
        )
    ) != 39:
        raise RuntimeError(
            "Final experiment identities are not unique"
        )

    return {
        "run_id": run.run_id,
        "run_dir": str(
            run.run_dir
        ),
        "final_hyperparameters_sha256": (
            EXPECTED_FINAL_HYPERPARAMETERS_SHA256
        ),
        "final_evaluation_sha256": (
            EXPECTED_FINAL_EVALUATION_SHA256
        ),
        "canonical_train_rows": len(
            canonical_split.train
        ),
        "canonical_validation_rows": len(
            canonical_split.validation
        ),
        "final_fit_rows": len(
            fit_split.train
        ),
        "fit_validation_rows": len(
            fit_split.validation
        ),
        "test_rows": len(
            canonical_split.test
        ),
        "models": len(
            FINAL_MODEL_ORDER
        ),
        "seeds": seeds,
        "planned_experiments": len(
            experiment_ids
        ),
        "test_metrics_computed": False,
    }


def execute_final_protocol(
    *,
    config_path: str | Path = "config/publication.yaml",
    version: str = "v1",
    final_hyperparameters_path: str | Path = FINAL_HYPERPARAMETERS_PATH,
    final_evaluation_path: str | Path = FINAL_EVALUATION_PATH,
    results_root: str | Path | None = None,
    run_id: str | None = DEFAULT_FINAL_RUN_ID,
    model_selection: str = "all",
    seed_selection: str = "all",
    authorize_test: bool = False,
) -> tuple[
    RunContext,
    list[ExperimentResult],
]:
    """Execute requested subset of the already-frozen final protocol."""

    # This gate is deliberately before split/model preparation.
    if authorize_test is not True:
        raise PermissionError(
            "Final test execution is locked. "
            "Pass --execute-final-test only after the final preflight passes."
        )

    (
        config,
        frozen_hp,
        frozen_eval,
        canonical_split,
        fit_split,
        run,
    ) = _prepare_final_state(
        config_path=config_path,
        version=version,
        final_hyperparameters_path=(
            final_hyperparameters_path
        ),
        final_evaluation_path=(
            final_evaluation_path
        ),
        results_root=results_root,
        run_id=run_id,
    )

    models = resolve_models(
        model_selection
    )

    seeds = resolve_seeds(
        seed_selection,
        frozen_eval,
    )

    results: list[
        ExperimentResult
    ] = []

    for seed in seeds:
        for model_name in models:

            print()
            print(
                "=" * 88
            )
            print(
                "FINAL MODEL:",
                model_name,
            )
            print(
                "RUN ID:",
                run.run_id,
            )
            print(
                "FINAL SEED:",
                seed,
            )
            print(
                "FIT ROWS:",
                len(
                    fit_split.train
                ),
            )
            print(
                "TARGET SPLIT: test"
            )
            print(
                "=" * 88
            )

            (
                model,
                spec,
            ) = (
                build_final_model_and_spec(
                    model_name=model_name,
                    seed=seed,
                    fit_split=fit_split,
                    config=config,
                    frozen_hyperparameters=(
                        frozen_hp
                    ),
                )
            )

            resumed = (
                load_completed_experiment(
                    run=run,
                    spec=spec,
                    target_split="test",
                )
            )

            if resumed is not None:
                result, artifacts = (
                    resumed
                )

                print(
                    "Resume: SKIPPED - verified completed final experiment"
                )

            else:
                result = (
                    run_one_final_experiment(
                        model=model,
                        spec=spec,
                        fit_split=fit_split,
                        canonical_split=(
                            canonical_split
                        ),
                        config=config,
                        frozen_hyperparameters=(
                            frozen_hp
                        ),
                        frozen_evaluation=(
                            frozen_eval
                        ),
                        authorize_test=True,
                    )
                )

                artifacts = (
                    write_experiment_result(
                        run=run,
                        result=result,
                    )
                )

            results.append(
                result
            )

            print(
                "Status:",
                result.status,
            )

            print(
                "Experiment ID:",
                artifacts.experiment_id,
            )

            print(
                "Runtime seconds:",
                f"{result.runtime_seconds:.3f}",
            )

            if (
                "ndcg_at_10"
                in
                result.ranking_metrics
            ):
                print(
                    "ndcg_at_10 =",
                    f"{result.ranking_metrics['ndcg_at_10']:.6f}",
                )

            if result.native_metrics:
                print(
                    "Native metrics:",
                    result.native_metrics,
                )

            print(
                "Stored at:",
                artifacts.experiment_dir,
            )

    return (
        run,
        results,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paper 2 frozen final publication runner. "
            "Without --execute-final-test this performs preflight only."
        )
    )

    parser.add_argument(
        "--config",
        default="config/publication.yaml",
    )

    parser.add_argument(
        "--version",
        default="v1",
    )

    parser.add_argument(
        "--final-hyperparameters",
        default=str(
            FINAL_HYPERPARAMETERS_PATH
        ),
    )

    parser.add_argument(
        "--final-evaluation",
        default=str(
            FINAL_EVALUATION_PATH
        ),
    )

    parser.add_argument(
        "--results-root",
        default=None,
    )

    parser.add_argument(
        "--run-id",
        default=DEFAULT_FINAL_RUN_ID,
    )

    parser.add_argument(
        "--model",
        default="all",
        choices=(
            [
                "all",
            ]
            +
            FINAL_MODEL_ORDER
        ),
    )

    parser.add_argument(
        "--seed",
        default="all",
        choices=[
            "all",
            "2026",
            "2027",
            "2028",
        ],
    )

    parser.add_argument(
        "--execute-final-test",
        action="store_true",
        help=(
            "Explicitly authorize fitting the frozen final models and "
            "computing test metrics. Omit this flag for preflight."
        ),
    )

    return parser


def main() -> int:
    parser = build_argument_parser()

    args = parser.parse_args()

    if not args.execute_final_test:
        summary = (
            preflight_final_protocol(
                config_path=args.config,
                version=args.version,
                final_hyperparameters_path=(
                    args.final_hyperparameters
                ),
                final_evaluation_path=(
                    args.final_evaluation
                ),
                results_root=(
                    args.results_root
                ),
                run_id=args.run_id,
            )
        )

        print()
        print(
            "=" * 88
        )
        print(
            "STEP 56 FINAL-RUN PREFLIGHT: PASS"
        )
        print(
            "=" * 88
        )

        for key, value in summary.items():
            print(
                f"{key}: {value}"
            )

        print()
        print(
            "NO MODEL WAS FIT."
        )
        print(
            "NO TEST METRIC WAS COMPUTED."
        )

        return 0

    run, results = (
        execute_final_protocol(
            config_path=args.config,
            version=args.version,
            final_hyperparameters_path=(
                args.final_hyperparameters
            ),
            final_evaluation_path=(
                args.final_evaluation
            ),
            results_root=(
                args.results_root
            ),
            run_id=args.run_id,
            model_selection=args.model,
            seed_selection=args.seed,
            authorize_test=True,
        )
    )

    print()
    print(
        "=" * 88
    )
    print(
        "STEP 56 FINAL EXECUTION SUBSET COMPLETE"
    )
    print(
        "Run ID:",
        run.run_id,
    )
    print(
        "Experiments returned:",
        len(
            results
        ),
    )
    print(
        "Run directory:",
        run.run_dir,
    )
    print(
        "=" * 88
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
