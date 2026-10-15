"""Run the frozen Step 57 Gaussian EF-EMB normalization stability ablation.

Scientific scope
----------------
Models:
    g-emb
    ng-emb

Context normalizations:
    raw
    sqrt
    mean

Seeds:
    2026
    2027
    2028

Target:
    canonical validation only

The test set is never an evaluation target in this runner.  The canonical
SplitResult is still supplied to the existing model-construction layer so
held-out pair identities retain the same leakage-prevention semantics used
during validation tuning.

The primary 39 final-test experiments are not modified or replaced by this
diagnostic ablation.
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
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_mf import load_frozen_split, prepare_run_context
from scripts.tune_models import build_tuning_model
from src.data import load_yaml
from src.experiment import ExperimentResult, ExperimentSpec, run_experiment
from src.results import (
    experiment_id_for_spec,
    load_completed_experiment,
    write_experiment_result,
)


# =============================================================================
# Frozen Step 57 protocol
# =============================================================================

STABILITY_PROTOCOL_PATH = (
    PROJECT_ROOT / "config" / "efemb_gaussian_stability_v1.yaml"
)

FINAL_HYPERPARAMETERS_PATH = (
    PROJECT_ROOT / "config" / "final_hyperparameters_v1.yaml"
)

PUBLICATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "publication.yaml"
)

STABILITY_PROTOCOL_SHA256 = (
    "1e27f0b23684b83cccb2dd625110c597"
    "579e0d6f42489d191f06b45264360664"
)

FINAL_HYPERPARAMETERS_SHA256 = (
    "9aae84dd59d7c34c82a67563139bdb272"
    "56874ba79c645c749cbcf788d72c981"
)

RUN_ID = "stability-ml1m-v1-1e27f0b2"

VERSION = "v1"

EXPECTED_TRAIN_ROWS = 805443
EXPECTED_VALIDATION_ROWS = 97383
EXPECTED_TEST_ROWS = 97383

MODEL_ORDER = (
    "g-emb",
    "ng-emb",
)

NORMALIZATION_ORDER = (
    "raw",
    "sqrt",
    "mean",
)

SEED_ORDER = (
    2026,
    2027,
    2028,
)

EXPECTED_EXPERIMENTS = 18

DIAGNOSTIC_BATCH_SIZE = 16384

DISTRIBUTION_KEYS = (
    "count",
    "min",
    "mean",
    "std",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "max",
)

NUMERICAL_FAILURE_MARKERS = (
    "non-finite",
    "nonfinite",
    "nan",
    "infinite",
    "overflow",
)

MODEL_METADATA = {
    "g-emb": {
        "family": "gaussian",
    },
    "ng-emb": {
        "family": "gaussian",
    },
}


# =============================================================================
# Small helpers
# =============================================================================


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    # PowerShell 5.1 Set-Content -Encoding UTF8 writes a BOM.
    with path.open("r", encoding="utf-8-sig") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")

    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")

    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if json.loads(existing) != json.loads(encoded):
            raise ValueError(
                f"Existing JSON differs from requested frozen payload: {path}"
            )
        return

    try:
        temporary.write_text(
            encoded + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError("Distribution input must be one-dimensional")

    if len(array) == 0:
        raise ValueError("Distribution input must not be empty")

    if not np.isfinite(array).all():
        raise ValueError("Distribution input contains non-finite values")

    quantiles = np.quantile(
        array,
        [0.50, 0.75, 0.90, 0.95, 0.99],
        method="linear",
    )

    result: dict[str, float | int] = {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "mean": float(np.mean(array, dtype=np.float64)),
        "std": float(np.std(array, dtype=np.float64, ddof=0)),
        "p50": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(np.max(array)),
    }

    if tuple(result.keys()) != DISTRIBUTION_KEYS:
        raise AssertionError("Distribution summary schema drifted")

    return result


def _is_numerical_stability_failure(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in NUMERICAL_FAILURE_MARKERS)


def _safe_training_history(model: Any) -> list[dict[str, Any]]:
    try:
        history = getattr(model, "training_history")
    except Exception:
        return []

    if callable(history):
        try:
            history = history()
        except Exception:
            return []

    if not isinstance(history, list):
        return []

    return deepcopy(history)


# =============================================================================
# Frozen-input validation
# =============================================================================


def _validate_protocol(protocol: dict[str, Any]) -> None:
    _require(
        protocol.get("schema_version") == "1.0",
        "Unexpected stability protocol schema_version",
    )
    _require(
        protocol.get("freeze_id")
        == "paper2-efemb-gaussian-stability-v1",
        "Unexpected stability protocol freeze_id",
    )
    _require(
        protocol.get("status") == "frozen",
        "Stability protocol is not frozen",
    )

    information = protocol.get("information_boundary")
    _require(
        isinstance(information, dict),
        "Missing information_boundary mapping",
    )
    _require(
        information.get("fit_partition") == ["train"],
        "Step 57 must fit canonical train only",
    )
    _require(
        information.get("evaluation_partition") == "validation",
        "Step 57 must evaluate validation only",
    )
    _require(
        information.get("test_access_allowed") is False,
        "Step 57 test access must remain false",
    )
    _require(
        information.get("allow_test") is False,
        "Step 57 allow_test must remain false",
    )

    _require(
        tuple(protocol.get("models", ())) == MODEL_ORDER,
        "Frozen model order mismatch",
    )
    _require(
        tuple(protocol.get("normalizations", ()))
        == NORMALIZATION_ORDER,
        "Frozen normalization order mismatch",
    )
    _require(
        tuple(protocol.get("seeds", ())) == SEED_ORDER,
        "Frozen seed order mismatch",
    )
    _require(
        int(protocol.get("planned_experiments", -1))
        == EXPECTED_EXPERIMENTS,
        "Frozen experiment count mismatch",
    )

    policy = protocol.get("hyperparameter_policy")
    _require(
        isinstance(policy, dict),
        "Missing hyperparameter_policy mapping",
    )
    _require(
        policy.get("source")
        == "config/final_hyperparameters_v1.yaml",
        "Frozen hyperparameter source mismatch",
    )


def _validate_frozen_files() -> tuple[dict[str, Any], dict[str, Any]]:
    actual_protocol_sha = _sha256_file(STABILITY_PROTOCOL_PATH)
    _require(
        actual_protocol_sha == STABILITY_PROTOCOL_SHA256,
        (
            "Step 57 protocol SHA mismatch. "
            f"Expected {STABILITY_PROTOCOL_SHA256}; "
            f"found {actual_protocol_sha}"
        ),
    )

    actual_hp_sha = _sha256_file(FINAL_HYPERPARAMETERS_PATH)
    _require(
        actual_hp_sha == FINAL_HYPERPARAMETERS_SHA256,
        (
            "Final hyperparameter SHA mismatch. "
            f"Expected {FINAL_HYPERPARAMETERS_SHA256}; "
            f"found {actual_hp_sha}"
        ),
    )

    protocol = _load_yaml_mapping(STABILITY_PROTOCOL_PATH)
    frozen_hp = _load_yaml_mapping(FINAL_HYPERPARAMETERS_PATH)

    _validate_protocol(protocol)

    tuned_models = frozen_hp.get("tuned_models")
    _require(
        isinstance(tuned_models, dict),
        "final_hyperparameters_v1.yaml missing tuned_models",
    )

    for model_name in MODEL_ORDER:
        entry = tuned_models.get(model_name)
        _require(
            isinstance(entry, dict),
            f"Missing frozen selected model: {model_name}",
        )

        hyperparameters = entry.get("hyperparameters")
        _require(
            isinstance(hyperparameters, dict),
            f"{model_name} missing frozen hyperparameters",
        )
        _require(
            hyperparameters.get("context_normalization") == "mean",
            (
                f"{model_name} frozen primary normalization must be mean "
                "before the prespecified ablation override"
            ),
        )

    return protocol, frozen_hp


def _build_hyperparameters(
    *,
    frozen_hp: dict[str, Any],
    model_name: str,
    normalization: str,
    seed: int,
) -> dict[str, Any]:
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unsupported Step 57 model: {model_name}")

    if normalization not in NORMALIZATION_ORDER:
        raise ValueError(
            f"Unsupported Step 57 normalization: {normalization}"
        )

    if seed not in SEED_ORDER:
        raise ValueError(f"Unsupported Step 57 seed: {seed}")

    source = (
        frozen_hp["tuned_models"][model_name]["hyperparameters"]
    )

    hp = deepcopy(source)

    # Exactly two fields are allowed to differ from the Step-55 dictionary:
    # the prespecified normalization ablation and the final replicate seed.
    hp["context_normalization"] = normalization
    hp["seed"] = int(seed)

    for key, value in source.items():
        if key == "context_normalization":
            continue

        _require(
            hp[key] == value,
            f"{model_name}: non-ablation hyperparameter changed: {key}",
        )

    return hp


def _build_spec(
    *,
    model_name: str,
    normalization: str,
    seed: int,
    hyperparameters: dict[str, Any],
) -> ExperimentSpec:
    return ExperimentSpec(
        model_name=model_name,
        family=MODEL_METADATA[model_name]["family"],
        variant=f"stability-{normalization}",
        seed=seed,
        hyperparameters=hyperparameters,
    )


def _planned_matrix(
    *,
    model_filter: str,
    normalization_filter: str,
    seed_filter: int | None,
) -> list[tuple[str, str, int]]:
    models: Iterable[str] = (
        MODEL_ORDER
        if model_filter == "all"
        else (model_filter,)
    )
    normalizations: Iterable[str] = (
        NORMALIZATION_ORDER
        if normalization_filter == "all"
        else (normalization_filter,)
    )
    seeds: Iterable[int] = (
        SEED_ORDER
        if seed_filter is None
        else (seed_filter,)
    )

    matrix = [
        (model_name, normalization, seed)
        for seed in seeds
        for model_name in models
        for normalization in normalizations
    ]

    return matrix


# =============================================================================
# Context diagnostics
# =============================================================================


def _positive_context_lengths(model: Any) -> np.ndarray:
    store = getattr(model, "_context_store", None)

    if store is None:
        raise RuntimeError("Fitted model has no EFContextStore")

    history_lengths = np.asarray(
        store.history_lengths,
        dtype=np.float64,
    )

    train_users = np.asarray(
        store.train_user_indices,
        dtype=np.int64,
    )

    lengths = history_lengths[train_users] - 1.0

    if np.any(lengths < 0.0):
        raise RuntimeError(
            "Positive training context lengths became negative"
        )

    return lengths


def _context_embedding_weight(model: Any, model_name: str) -> torch.Tensor:
    module = getattr(model, "_module", None)

    if module is None:
        raise RuntimeError("Fitted model has no module")

    if model_name == "g-emb":
        return module.context_embeddings.weight

    if model_name == "ng-emb":
        method = getattr(model, "_positive_context_weight", None)
        if method is None or not callable(method):
            raise RuntimeError(
                "ng-emb positivity-consistent context accessor missing"
            )
        return method()

    raise ValueError(model_name)


def _positive_context_norms(
    model: Any,
    *,
    model_name: str,
    batch_size: int = DIAGNOSTIC_BATCH_SIZE,
) -> np.ndarray:
    store = getattr(model, "_context_store", None)
    incidence = getattr(model, "_incidence", None)

    if store is None or incidence is None:
        raise RuntimeError(
            "Fitted model is missing context diagnostic state"
        )

    user_indices_np = np.asarray(
        store.train_user_indices,
        dtype=np.int64,
    )
    item_indices_np = np.asarray(
        store.train_item_indices,
        dtype=np.int64,
    )

    if len(user_indices_np) != len(item_indices_np):
        raise RuntimeError(
            "EF context diagnostic train-index lengths mismatch"
        )

    output = np.empty(
        len(user_indices_np),
        dtype=np.float64,
    )

    with torch.no_grad():
        context_embeddings = _context_embedding_weight(
            model,
            model_name,
        )

        user_sums = store.aggregate_user_sums(
            context_embeddings,
            incidence=incidence,
        )

        device = context_embeddings.device

        for start in range(0, len(output), batch_size):
            stop = min(start + batch_size, len(output))

            users = torch.as_tensor(
                user_indices_np[start:stop],
                dtype=torch.long,
                device=device,
            )
            items = torch.as_tensor(
                item_indices_np[start:stop],
                dtype=torch.long,
                device=device,
            )

            contexts = store.context_vectors(
                context_embeddings=context_embeddings,
                user_indices=users,
                target_item_indices=items,
                exclude_target=True,
                normalization=model.context_normalization,
                incidence=incidence,
                precomputed_user_sums=user_sums,
            )

            norms = torch.linalg.vector_norm(
                contexts,
                ord=2,
                dim=1,
            )

            chunk = (
                norms.detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )

            if not np.isfinite(chunk).all():
                raise RuntimeError(
                    "Positive context norm diagnostic contains "
                    "non-finite values"
                )

            output[start:stop] = chunk

    return output


def _history_extrema(
    history: list[dict[str, Any]],
) -> dict[str, float | int]:
    if not history:
        raise RuntimeError(
            "Completed Gaussian EF-EMB run has no training history"
        )

    numeric_keys = (
        "mse",
        "rmse",
        "mae",
        "total_loss",
        "max_abs_prediction",
        "max_context_norm",
        "max_gradient_norm_before_clipping",
        "nonfinite_events",
    )

    result: dict[str, float | int] = {
        "epochs_recorded": int(len(history)),
    }

    final = history[-1]

    for key in numeric_keys:
        values = [
            float(record[key])
            for record in history
            if key in record
        ]

        if len(values) != len(history):
            raise RuntimeError(
                f"Training-history key missing from one or more epochs: {key}"
            )

        array = np.asarray(values, dtype=np.float64)

        if not np.isfinite(array).all():
            raise RuntimeError(
                f"Training-history key became non-finite: {key}"
            )

        result[f"max_{key}_over_epochs"] = float(np.max(array))
        result[f"final_{key}"] = float(final[key])

    return result


def _validation_max_abs_prediction(
    model: Any,
    validation_frame: Any,
) -> float:
    method = getattr(model, "predict_ratings", None)

    if method is None or not callable(method):
        raise RuntimeError(
            "Gaussian EF-EMB model lacks predict_ratings()"
        )

    predictions = np.asarray(
        method(validation_frame),
        dtype=np.float64,
    )

    if predictions.ndim != 1 or len(predictions) != len(validation_frame):
        raise RuntimeError(
            "Validation prediction diagnostic has invalid shape"
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            "Validation prediction diagnostic contains non-finite values"
        )

    return float(np.max(np.abs(predictions)))


def _diagnostic_payload(
    *,
    model: Any,
    model_name: str,
    normalization: str,
    seed: int,
    experiment_id: str,
    result: ExperimentResult,
    split: Any,
) -> dict[str, Any]:
    lengths = _positive_context_lengths(model)
    norms = _positive_context_norms(
        model,
        model_name=model_name,
    )

    if len(lengths) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Positive-context length diagnostic count does not "
            "match canonical training rows"
        )

    if len(norms) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Positive-context norm diagnostic count does not "
            "match canonical training rows"
        )

    history = deepcopy(result.training_history)

    payload = {
        "schema_version": "1.0",
        "freeze_id": "paper2-efemb-gaussian-stability-v1",
        "stability_protocol_sha256": STABILITY_PROTOCOL_SHA256,
        "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
        "run_id": RUN_ID,
        "experiment_id": experiment_id,
        "model_name": model_name,
        "variant": f"stability-{normalization}",
        "context_normalization": normalization,
        "seed": int(seed),
        "target_split": "validation",
        "test_metrics_computed": False,
        "status": "completed",
        "ranking_ndcg_at_10": float(
            result.ranking_metrics["ndcg_at_10"]
        ),
        "native_metrics": deepcopy(result.native_metrics),
        "validation_max_abs_prediction": (
            _validation_max_abs_prediction(
                model,
                split.validation,
            )
        ),
        "training_history_extrema": _history_extrema(history),
        "positive_training_context_length": (
            _distribution_summary(lengths)
        ),
        "positive_training_context_norm": (
            _distribution_summary(norms)
        ),
        "distribution_quantile_method": "numpy_linear",
    }

    return payload


# =============================================================================
# Run-level manifests and failure markers
# =============================================================================


def _protocol_manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "freeze_id": "paper2-efemb-gaussian-stability-v1",
        "stability_protocol_sha256": STABILITY_PROTOCOL_SHA256,
        "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
        "run_id": RUN_ID,
        "version": VERSION,
        "models": list(MODEL_ORDER),
        "normalizations": list(NORMALIZATION_ORDER),
        "seeds": list(SEED_ORDER),
        "planned_experiments": EXPECTED_EXPERIMENTS,
        "fit_partition": ["train"],
        "evaluation_partition": "validation",
        "test_access_allowed": False,
        "primary_results_affected": False,
        "retuning_allowed": False,
    }


def _failure_payload(
    *,
    experiment_id: str,
    model_name: str,
    normalization: str,
    seed: int,
    hyperparameters: dict[str, Any],
    runtime_seconds: float,
    exc: BaseException,
    model: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "freeze_id": "paper2-efemb-gaussian-stability-v1",
        "stability_protocol_sha256": STABILITY_PROTOCOL_SHA256,
        "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
        "run_id": RUN_ID,
        "experiment_id": experiment_id,
        "model_name": model_name,
        "variant": f"stability-{normalization}",
        "context_normalization": normalization,
        "seed": int(seed),
        "target_split": "validation",
        "test_metrics_computed": False,
        "status": "failed_numerical_stability",
        "runtime_seconds": float(runtime_seconds),
        "exception_type": type(exc).__name__,
        "error_message": str(exc),
        "hyperparameters": deepcopy(hyperparameters),
        "partial_training_history": _safe_training_history(model),
        "scientific_policy": {
            "retry_with_changed_configuration": False,
            "retune_after_failure": False,
            "continue_remaining_frozen_runs": True,
        },
    }


def _verified_failure_marker(
    path: Path,
    *,
    experiment_id: str,
    model_name: str,
    normalization: str,
    seed: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))

    checks = {
        "stability_protocol_sha256": STABILITY_PROTOCOL_SHA256,
        "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
        "run_id": RUN_ID,
        "experiment_id": experiment_id,
        "model_name": model_name,
        "variant": f"stability-{normalization}",
        "context_normalization": normalization,
        "seed": seed,
        "target_split": "validation",
        "status": "failed_numerical_stability",
    }

    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Failure marker identity mismatch for {key}: {path}"
            )

    if payload.get("test_metrics_computed") is not False:
        raise ValueError(
            f"Failure marker unexpectedly claims test access: {path}"
        )

    return payload



def _validate_distribution_summary_payload(
    summary: dict[str, Any],
    *,
    path: Path,
    quantity: str,
) -> None:
    # Persisted JSON is written with sort_keys=True, so key insertion order
    # is not part of the frozen diagnostic schema. Validate exact key
    # membership while still rejecting malformed/non-finite values.
    expected_keys = set(DISTRIBUTION_KEYS)
    actual_keys = set(summary)

    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "Diagnostic distribution schema mismatch "
            f"for {quantity}: {path}; "
            f"missing={missing}; extra={extra}"
        )

    count = summary["count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ValueError(
            "Diagnostic distribution count must be a positive integer "
            f"for {quantity}: {path}"
        )

    for key in DISTRIBUTION_KEYS:
        if key == "count":
            continue

        value = summary[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                "Diagnostic distribution contains an invalid numeric value "
                f"for {quantity}.{key}: {path}"
            )

    if float(summary["std"]) < 0.0:
        raise ValueError(
            "Diagnostic distribution std must be non-negative "
            f"for {quantity}: {path}"
        )

    ordered = (
        float(summary["min"]),
        float(summary["p50"]),
        float(summary["p75"]),
        float(summary["p90"]),
        float(summary["p95"]),
        float(summary["p99"]),
        float(summary["max"]),
    )

    if any(
        left > right
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError(
            "Diagnostic distribution quantiles are not monotone "
            f"for {quantity}: {path}"
        )

    mean = float(summary["mean"])
    if not (
        float(summary["min"])
        <= mean
        <= float(summary["max"])
    ):
        raise ValueError(
            "Diagnostic distribution mean lies outside [min, max] "
            f"for {quantity}: {path}"
        )



def _verified_diagnostic_sidecar(
    path: Path,
    *,
    experiment_id: str,
    model_name: str,
    normalization: str,
    seed: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Completed experiment is missing stability diagnostics: {path}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    checks = {
        "stability_protocol_sha256": STABILITY_PROTOCOL_SHA256,
        "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
        "run_id": RUN_ID,
        "experiment_id": experiment_id,
        "model_name": model_name,
        "variant": f"stability-{normalization}",
        "context_normalization": normalization,
        "seed": seed,
        "target_split": "validation",
        "status": "completed",
    }

    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Diagnostic sidecar identity mismatch for {key}: {path}"
            )

    if payload.get("test_metrics_computed") is not False:
        raise ValueError(
            f"Diagnostic sidecar unexpectedly claims test access: {path}"
        )

    for quantity in (
        "positive_training_context_length",
        "positive_training_context_norm",
    ):
        summary = payload.get(quantity)
        if not isinstance(summary, dict):
            raise ValueError(
                f"Diagnostic sidecar missing {quantity}: {path}"
            )

        _validate_distribution_summary_payload(
            summary,
            path=path,
            quantity=quantity,
        )

    return payload


# =============================================================================
# Execution
# =============================================================================


def _results_root(config: dict[str, Any]) -> Path:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")

    value = paths.get("results_root")
    if not isinstance(value, (str, Path)):
        raise ValueError("config.paths.results_root is missing")

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _print_preflight(
    *,
    split: Any,
    matrix: list[tuple[str, str, int]],
) -> None:
    print("=" * 88)
    print("STEP 57 GAUSSIAN EF-EMB STABILITY PREFLIGHT: PASS")
    print("=" * 88)
    print("run_id:", RUN_ID)
    print("stability_protocol_sha256:", STABILITY_PROTOCOL_SHA256)
    print("final_hyperparameters_sha256:", FINAL_HYPERPARAMETERS_SHA256)
    print("train_rows:", len(split.train))
    print("validation_rows:", len(split.validation))
    print("test_rows:", len(split.test))
    print("evaluation_target: validation")
    print("test_metrics_computed: False")
    print("models:", list(MODEL_ORDER))
    print("normalizations:", list(NORMALIZATION_ORDER))
    print("seeds:", list(SEED_ORDER))
    print("frozen_planned_experiments:", EXPECTED_EXPERIMENTS)
    print("selected_experiments_this_invocation:", len(matrix))
    print()
    print("NO MODEL WAS FIT.")
    print("NO TEST METRIC WAS COMPUTED.")


def run_stability_ablation(
    *,
    execute: bool,
    model_filter: str = "all",
    normalization_filter: str = "all",
    seed_filter: int | None = None,
) -> tuple[int, int, int]:
    _, frozen_hp = _validate_frozen_files()

    config = load_yaml(PUBLICATION_CONFIG_PATH)

    if not isinstance(config, dict):
        raise TypeError("publication.yaml root must be a mapping")

    split = load_frozen_split(
        config,
        version=VERSION,
    )

    _require(
        len(split.train) == EXPECTED_TRAIN_ROWS,
        "Canonical train row count changed",
    )
    _require(
        len(split.validation) == EXPECTED_VALIDATION_ROWS,
        "Canonical validation row count changed",
    )
    _require(
        len(split.test) == EXPECTED_TEST_ROWS,
        "Canonical test row count changed",
    )

    matrix = _planned_matrix(
        model_filter=model_filter,
        normalization_filter=normalization_filter,
        seed_filter=seed_filter,
    )

    if not execute:
        _print_preflight(
            split=split,
            matrix=matrix,
        )
        return (0, 0, 0)

    run = prepare_run_context(
        config_path=PUBLICATION_CONFIG_PATH,
        config=config,
        version=VERSION,
        results_root=_results_root(config),
        run_id=RUN_ID,
    )

    _atomic_json(
        run.run_dir / "stability_protocol_manifest.json",
        _protocol_manifest_payload(),
    )

    diagnostic_dir = run.run_dir / "stability_diagnostics"
    failure_dir = run.run_dir / "stability_failures"

    completed = 0
    failed = 0
    resumed = 0

    for model_name, normalization, seed in matrix:
        hp = _build_hyperparameters(
            frozen_hp=frozen_hp,
            model_name=model_name,
            normalization=normalization,
            seed=seed,
        )

        spec = _build_spec(
            model_name=model_name,
            normalization=normalization,
            seed=seed,
            hyperparameters=hp,
        )

        experiment_id = experiment_id_for_spec(
            spec,
            target_split="validation",
        )

        sidecar_path = (
            diagnostic_dir / f"{experiment_id}.json"
        )
        failure_path = (
            failure_dir / f"{experiment_id}.json"
        )

        print()
        print("=" * 88)
        print("STEP 57 STABILITY RUN")
        print("Model:", model_name)
        print("Normalization:", normalization)
        print("Seed:", seed)
        print("Target split: validation")
        print("Experiment ID:", experiment_id)
        print("=" * 88)

        existing = load_completed_experiment(
            run=run,
            spec=spec,
            target_split="validation",
        )

        if existing is not None:
            result, artifacts = existing

            _verified_diagnostic_sidecar(
                sidecar_path,
                experiment_id=experiment_id,
                model_name=model_name,
                normalization=normalization,
                seed=seed,
            )

            if failure_path.exists():
                raise RuntimeError(
                    "Experiment has both completed and failure artifacts: "
                    f"{experiment_id}"
                )

            resumed += 1
            completed += 1

            print(
                "Resume: SKIPPED - verified completed "
                "stability experiment"
            )
            print("Status:", result.status)
            print(
                "ndcg_at_10 =",
                f"{result.ranking_metrics['ndcg_at_10']:.6f}",
            )
            print("Native metrics:", result.native_metrics)
            print("Stored at:", artifacts.experiment_dir)
            continue

        previous_failure = _verified_failure_marker(
            failure_path,
            experiment_id=experiment_id,
            model_name=model_name,
            normalization=normalization,
            seed=seed,
        )

        if previous_failure is not None:
            resumed += 1
            failed += 1

            print(
                "Resume: SKIPPED - verified numerical-stability "
                "failure outcome"
            )
            print(
                "Failure:",
                previous_failure["error_message"],
            )
            continue

        model = build_tuning_model(
            model_name=model_name,
            split=split,
            hyperparameters=hp,
        )

        start = perf_counter()

        try:
            result = run_experiment(
                model=model,
                split=split,
                config=config,
                spec=spec,
                target_split="validation",
                allow_test=False,
            )
        except Exception as exc:
            runtime = perf_counter() - start

            if not _is_numerical_stability_failure(exc):
                raise

            payload = _failure_payload(
                experiment_id=experiment_id,
                model_name=model_name,
                normalization=normalization,
                seed=seed,
                hyperparameters=hp,
                runtime_seconds=runtime,
                exc=exc,
                model=model,
            )

            _atomic_json(
                failure_path,
                payload,
            )

            failed += 1

            print("Status: failed_numerical_stability")
            print("Runtime seconds:", f"{runtime:.3f}")
            print("Failure:", str(exc))
            print("Failure artifact:", failure_path)
            continue

        if result.status != "completed":
            raise RuntimeError(
                f"Unexpected noncompleted ExperimentResult: {result.status}"
            )

        if failure_path.exists():
            raise RuntimeError(
                "Fresh completed experiment conflicts with "
                f"failure marker: {experiment_id}"
            )

        diagnostics = _diagnostic_payload(
            model=model,
            model_name=model_name,
            normalization=normalization,
            seed=seed,
            experiment_id=experiment_id,
            result=result,
            split=split,
        )

        # Seal diagnostics before sealing the ordinary result checkpoint.
        _atomic_json(
            sidecar_path,
            diagnostics,
        )

        artifacts = write_experiment_result(
            run=run,
            result=result,
        )

        completed += 1

        print("Status:", result.status)
        print("Runtime seconds:", f"{result.runtime_seconds:.3f}")
        print(
            "ndcg_at_10 =",
            f"{result.ranking_metrics['ndcg_at_10']:.6f}",
        )
        print("Native metrics:", result.native_metrics)
        print("Diagnostics:", sidecar_path)
        print("Stored at:", artifacts.experiment_dir)

    print()
    print("=" * 88)
    print("STEP 57 STABILITY INVOCATION COMPLETE")
    print("Run ID:", RUN_ID)
    print("Experiments selected:", len(matrix))
    print("Completed outcomes:", completed)
    print("Numerical-failure outcomes:", failed)
    print("Resumed outcomes:", resumed)
    print("Test metrics computed: False")
    print("=" * 88)

    return completed, failed, resumed


# =============================================================================
# CLI
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Step 57 Gaussian EF-EMB "
            "context-normalization stability ablation."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually fit/evaluate the frozen validation-only "
            "ablation. Without this flag, only preflight runs."
        ),
    )

    parser.add_argument(
        "--model",
        choices=("all", *MODEL_ORDER),
        default="all",
    )

    parser.add_argument(
        "--normalization",
        choices=("all", *NORMALIZATION_ORDER),
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

    run_stability_ablation(
        execute=bool(args.execute),
        model_filter=args.model,
        normalization_filter=args.normalization,
        seed_filter=args.seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
