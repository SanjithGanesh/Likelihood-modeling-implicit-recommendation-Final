"""Validation-only hyperparameter tuning for Paper 2 publication experiments.

Consumes the frozen ML-1M publication artifacts plus
config/tuning_spaces_v1.yaml. This runner intentionally exposes no test
selection/evaluation arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_mf import load_frozen_split, prepare_run_context
from src.data import load_yaml
from src.experiment import ExperimentResult, ExperimentSpec, run_experiment
from src.models.ap_emb import AdditivePoissonEFEmb
from src.models.b_emb import BernoulliEFEmb
from src.models.bernoulli_mf import BernoulliMF
from src.models.g_emb import GaussianEFEmb
from src.models.gaussian_mf import GaussianMF
from src.models.hpf import HPF
from src.models.logistic_pca import LogisticPCA
from src.models.ng_emb import NonnegativeGaussianEFEmb
from src.models.p_emb import PoissonEFEmb
from src.models.pca_svd import PCASVDBaseline
from src.models.poisson_mf import PoissonMF
from src.results import (
    RunContext,
    experiment_id_for_spec,
    load_completed_experiment,
    write_experiment_result,
)
from src.sampling import NegativeSampler
from src.split import SplitResult


TARGET_SPLIT = "validation"

MODEL_ORDER = [
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

MODEL_METADATA: dict[str, dict[str, str]] = {
    "PCA/SVD": {"family": "baseline", "variant": "primary"},
    "BernoulliMF": {"family": "bernoulli", "variant": "primary"},
    "PoissonMF": {"family": "poisson", "variant": "primary"},
    "GaussianMF": {"family": "gaussian", "variant": "primary"},
    "LogisticPCA": {"family": "bernoulli", "variant": "primary"},
    "HPF": {"family": "poisson", "variant": "primary"},
    "b-emb": {"family": "bernoulli", "variant": "primary"},
    "p-emb": {"family": "poisson", "variant": "primary"},
    "ap-emb": {"family": "poisson", "variant": "primary"},
    "g-emb": {"family": "gaussian", "variant": "primary"},
    "ng-emb": {"family": "gaussian", "variant": "primary"},
}


@dataclass(frozen=True)
class CandidateOutcome:
    stage: str
    spec: ExperimentSpec
    experiment_id: str
    result: ExperimentResult
    resumed: bool


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(str(path).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {resolved}")
    return payload


def _sha256_file(path: str | Path) -> str:
    resolved = _resolve_project_path(path)
    hasher = hashlib.sha256()
    with resolved.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    content = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json_consistent(path: Path, payload: Any) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if _canonical_json_bytes(existing) != _canonical_json_bytes(payload):
            raise ValueError(
                "Existing tuning metadata differs from requested payload: "
                f"{path}"
            )
        return
    _write_json_atomic(path, payload)


def _validate_scientific_contract(
    *,
    config: Mapping[str, Any],
    tuning: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    version: str,
) -> tuple[int, str]:
    """Hard gate that runs before split/model access."""

    policy = tuning.get("tuning_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("tuning_policy must be a mapping")

    if policy.get("dataset_id") != "ml-1m":
        raise ValueError("Tuning is frozen to dataset ml-1m")
    if policy.get("artifact_version") != version:
        raise ValueError("Tuning artifact version mismatch")
    if policy.get("selection_split") != TARGET_SPLIT:
        raise PermissionError("Tuning selection split must be validation")
    if policy.get("test_access_allowed") is not False:
        raise PermissionError("Tuning configuration must prohibit test access")

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("config.evaluation must be a mapping")
    if evaluation.get("test_access_during_tuning") is not False:
        raise PermissionError(
            "publication config must disable test access during tuning"
        )

    gt_dataset = ground_truth.get("dataset")
    gt_rules = ground_truth.get("publication_rules")
    gt_evaluation = ground_truth.get("evaluation")
    if not isinstance(gt_dataset, Mapping):
        raise ValueError("Ground Truth dataset must be a mapping")
    if not isinstance(gt_rules, Mapping):
        raise ValueError("Ground Truth publication_rules must be a mapping")
    if not isinstance(gt_evaluation, Mapping):
        raise ValueError("Ground Truth evaluation must be a mapping")

    if gt_dataset.get("id") != "ml-1m":
        raise ValueError("Publication Ground Truth dataset mismatch")
    if gt_dataset.get("artifact_version") != version:
        raise ValueError("Publication Ground Truth version mismatch")
    if gt_rules.get("test_locked_during_tuning") is not True:
        raise PermissionError("Ground Truth must lock test during tuning")
    if gt_evaluation.get("test_access_during_tuning") is not False:
        raise PermissionError("Ground Truth unexpectedly permits test access")

    selection_seed = policy.get("selection_seed")
    if not isinstance(selection_seed, int) or isinstance(selection_seed, bool):
        raise ValueError("selection_seed must be an integer")

    primary_metric = policy.get("primary_metric")
    if not isinstance(primary_metric, str) or not primary_metric:
        raise ValueError("primary_metric must be a nonempty string")
    if evaluation.get("primary_selection_metric") != primary_metric:
        raise ValueError("Tuning metric differs from publication config")

    return selection_seed, primary_metric


def _model_selection(requested: str) -> list[str]:
    if requested == "all":
        return list(MODEL_ORDER)
    if requested not in MODEL_ORDER:
        raise ValueError(f"Unknown tuning model: {requested}")
    return [requested]


def _cartesian_candidates(stage_1: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not stage_1:
        raise ValueError("stage_1 must not be empty")

    keys = list(stage_1.keys())
    value_lists: list[list[Any]] = []
    for key in keys:
        values = stage_1[key]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Every stage_1 parameter must map to a nonempty list: {key}"
            )
        value_lists.append(values)

    return [
        dict(zip(keys, values))
        for values in itertools.product(*value_lists)
    ]


def _deduplicate_hyperparameters(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[bytes] = set()
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        encoded = _canonical_json_bytes(candidate)
        if encoded in seen:
            continue
        seen.add(encoded)
        result.append(candidate)
    return result


def _execution_device(config: Mapping[str, Any]) -> str:
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("config.runtime must be a mapping")
    device = runtime.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError("config.runtime.device must be a nonempty string")
    return device


def _complete_hyperparameters(
    *,
    model_name: str,
    candidate: Mapping[str, Any],
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(candidate)
    result["seed"] = seed

    if model_name in {
        "BernoulliMF",
        "PoissonMF",
        "GaussianMF",
        "LogisticPCA",
        "b-emb",
        "p-emb",
        "ap-emb",
        "g-emb",
        "ng-emb",
    }:
        result["device"] = _execution_device(config)

    return result


def _all_observed(split: SplitResult) -> pd.DataFrame:
    return pd.concat(
        [split.train, split.validation, split.test],
        ignore_index=True,
    )


def _heldout_interactions(split: SplitResult) -> pd.DataFrame:
    return pd.concat(
        [split.validation, split.test],
        ignore_index=True,
    )


def _negative_sampler(
    *,
    split: SplitResult,
    seed: int,
    namespace: str,
) -> NegativeSampler:
    return NegativeSampler(
        train=split.train,
        all_observed=_all_observed(split),
        seed=seed,
        namespace=namespace,
        popularity_alpha=1.0,
    )


def build_tuning_model(
    *,
    model_name: str,
    split: SplitResult,
    hyperparameters: Mapping[str, Any],
) -> Any:
    """Construct one model from an explicit candidate configuration."""

    hp = dict(hyperparameters)

    if model_name == "PCA/SVD":
        return PCASVDBaseline(**hp)

    if model_name == "BernoulliMF":
        return BernoulliMF(
            negative_sampler=_negative_sampler(
                split=split,
                seed=int(hp["seed"]),
                namespace="bernoulli_mf_negative_sampling",
            ),
            **hp,
        )

    if model_name == "PoissonMF":
        return PoissonMF(
            negative_sampler=_negative_sampler(
                split=split,
                seed=int(hp["seed"]),
                namespace="poisson_mf_zero_sampling",
            ),
            **hp,
        )

    if model_name == "GaussianMF":
        return GaussianMF(**hp)

    if model_name == "LogisticPCA":
        return LogisticPCA(
            heldout_interactions=_heldout_interactions(split),
            **hp,
        )

    if model_name == "HPF":
        return HPF(
            heldout_interactions=_heldout_interactions(split),
            **hp,
        )

    if model_name == "b-emb":
        return BernoulliEFEmb(
            negative_sampler=_negative_sampler(
                split=split,
                seed=int(hp["seed"]),
                namespace="b_emb_negative_sampling",
            ),
            heldout_interactions=_heldout_interactions(split),
            **hp,
        )

    if model_name == "p-emb":
        return PoissonEFEmb(
            zero_sampler=_negative_sampler(
                split=split,
                seed=int(hp["seed"]),
                namespace="p_emb_zero_sampling",
            ),
            heldout_interactions=_heldout_interactions(split),
            **hp,
        )

    if model_name == "ap-emb":
        return AdditivePoissonEFEmb(
            zero_sampler=_negative_sampler(
                split=split,
                seed=int(hp["seed"]),
                namespace="ap_emb_zero_sampling",
            ),
            heldout_interactions=_heldout_interactions(split),
            **hp,
        )

    if model_name == "g-emb":
        return GaussianEFEmb(
            heldout_interactions=_heldout_interactions(split),
            **hp,
        )

    if model_name == "ng-emb":
        return NonnegativeGaussianEFEmb(
            heldout_interactions=_heldout_interactions(split),
            **hp,
        )

    raise ValueError(f"Unsupported tuning model: {model_name}")


def build_tuning_spec(
    *,
    model_name: str,
    seed: int,
    hyperparameters: dict[str, Any],
) -> ExperimentSpec:
    metadata = MODEL_METADATA.get(model_name)
    if metadata is None:
        raise ValueError(f"Unknown tuning model: {model_name}")
    return ExperimentSpec(
        model_name=model_name,
        family=metadata["family"],
        variant=metadata["variant"],
        seed=seed,
        hyperparameters=hyperparameters,
    )


def _metric_value(result: ExperimentResult, metric_name: str) -> float:
    if metric_name not in result.ranking_metrics:
        raise ValueError(
            f"Result for {result.model_name} is missing {metric_name}"
        )
    value = float(result.ranking_metrics[metric_name])
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite {metric_name} for {result.model_name}"
        )
    return value


def run_or_resume_candidate(
    *,
    run: RunContext,
    split: SplitResult,
    config: dict[str, Any],
    model_name: str,
    seed: int,
    hyperparameters: dict[str, Any],
    stage: str,
    primary_metric: str,
) -> CandidateOutcome:
    spec = build_tuning_spec(
        model_name=model_name,
        seed=seed,
        hyperparameters=hyperparameters,
    )
    experiment_id = experiment_id_for_spec(spec, target_split=TARGET_SPLIT)

    resumed_payload = load_completed_experiment(
        run=run,
        spec=spec,
        target_split=TARGET_SPLIT,
    )

    if resumed_payload is not None:
        result, _ = resumed_payload
        resumed = True
        print(
            "Resume: SKIPPED - verified completed experiment "
            f"{experiment_id}"
        )
    else:
        model = build_tuning_model(
            model_name=model_name,
            split=split,
            hyperparameters=hyperparameters,
        )
        result = run_experiment(
            model=model,
            split=split,
            config=config,
            spec=spec,
            target_split=TARGET_SPLIT,
            allow_test=False,
        )
        artifacts = write_experiment_result(run=run, result=result)
        if artifacts.experiment_id != experiment_id:
            raise RuntimeError(
                "Stored experiment ID differs from expected tuning identity"
            )
        resumed = False

    if result.status != "completed":
        raise RuntimeError(
            "Tuning candidate did not complete successfully: "
            f"{experiment_id}; status={result.status}; "
            f"error={result.error_message}"
        )

    metric = _metric_value(result, primary_metric)
    print(
        f"{stage} | {model_name} | {experiment_id} | "
        f"{primary_metric}={metric:.6f}"
    )

    return CandidateOutcome(
        stage=stage,
        spec=spec,
        experiment_id=experiment_id,
        result=result,
        resumed=resumed,
    )


def _latent_dimension(hyperparameters: Mapping[str, Any]) -> int:
    if "n_factors" in hyperparameters:
        return int(hyperparameters["n_factors"])
    if "n_components" in hyperparameters:
        return int(hyperparameters["n_components"])
    return 2**31 - 1


def select_best_candidate(
    outcomes: list[CandidateOutcome],
    *,
    primary_metric: str,
) -> CandidateOutcome:
    if not outcomes:
        raise ValueError("Cannot select a winner from zero candidates")

    def sort_key(outcome: CandidateOutcome) -> tuple[float, int, str]:
        return (
            -_metric_value(outcome.result, primary_metric),
            _latent_dimension(outcome.spec.hyperparameters),
            outcome.experiment_id,
        )

    return min(outcomes, key=sort_key)


def _stage_1_hyperparameters(
    *,
    model_name: str,
    model_space: Mapping[str, Any],
    seed: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fixed = model_space.get("fixed", {})
    stage_1 = model_space.get("stage_1")
    if not isinstance(fixed, Mapping):
        raise ValueError(f"{model_name}.fixed must be a mapping")
    if not isinstance(stage_1, Mapping):
        raise ValueError(f"{model_name}.stage_1 must be a mapping")

    candidates: list[dict[str, Any]] = []
    for stage_values in _cartesian_candidates(stage_1):
        candidate = dict(fixed)
        candidate.update(stage_values)
        candidates.append(
            _complete_hyperparameters(
                model_name=model_name,
                candidate=candidate,
                seed=seed,
                config=config,
            )
        )
    return _deduplicate_hyperparameters(candidates)


def _stage_2_hyperparameters(
    *,
    model_space: Mapping[str, Any],
    stage_1_winner: CandidateOutcome,
) -> list[dict[str, Any]]:
    base = dict(stage_1_winner.spec.hyperparameters)
    stage_2 = model_space.get("stage_2_at_best_stage_1", [])
    additional = model_space.get("additional_capacity_checks", [])

    if not isinstance(stage_2, list):
        raise ValueError("stage_2_at_best_stage_1 must be a list")
    if not isinstance(additional, list):
        raise ValueError("additional_capacity_checks must be a list")

    candidates: list[dict[str, Any]] = []
    for override in [*stage_2, *additional]:
        if not isinstance(override, Mapping):
            raise ValueError("Every Stage-2 override must be a mapping")
        candidate = dict(base)
        candidate.update(dict(override))
        candidates.append(candidate)

    return _deduplicate_hyperparameters(candidates)


def _tuning_manifest_payload(
    *,
    tuning_path: str | Path,
    ground_truth_path: str | Path,
    tuning: Mapping[str, Any],
) -> dict[str, Any]:
    policy = tuning["tuning_policy"]
    return {
        "schema_version": "1.0",
        "purpose": "Paper 2 validation-only publication tuning",
        "tuning_space_path": str(Path(tuning_path)).replace("\\", "/"),
        "tuning_space_sha256": _sha256_file(tuning_path),
        "publication_ground_truth_path": str(Path(ground_truth_path)).replace(
            "\\", "/"
        ),
        "publication_ground_truth_sha256": _sha256_file(ground_truth_path),
        "selection_split": policy["selection_split"],
        "selection_seed": policy["selection_seed"],
        "primary_metric": policy["primary_metric"],
        "test_access_allowed": policy["test_access_allowed"],
    }


def _ensure_tuning_manifest(
    *,
    run: RunContext,
    tuning_path: str | Path,
    ground_truth_path: str | Path,
    tuning: Mapping[str, Any],
) -> None:
    payload = _tuning_manifest_payload(
        tuning_path=tuning_path,
        ground_truth_path=ground_truth_path,
        tuning=tuning,
    )
    _write_json_consistent(run.run_dir / "tuning_manifest.json", payload)


def _selection_payload(
    *,
    model_name: str,
    primary_metric: str,
    stage_1: list[CandidateOutcome],
    stage_2: list[CandidateOutcome],
    winner: CandidateOutcome,
) -> dict[str, Any]:
    all_outcomes = [*stage_1, *stage_2]
    return {
        "schema_version": "1.0",
        "model_name": model_name,
        "selection_split": TARGET_SPLIT,
        "primary_metric": primary_metric,
        "selected_experiment_id": winner.experiment_id,
        "selected_stage": winner.stage,
        "selected_metric_value": _metric_value(winner.result, primary_metric),
        "selected_hyperparameters": dict(winner.spec.hyperparameters),
        "stage_1_trials": len(stage_1),
        "stage_2_trials": len(stage_2),
        "total_trials_considered": len(all_outcomes),
        "candidate_experiment_ids": [x.experiment_id for x in all_outcomes],
    }


def tune_one_model(
    *,
    run: RunContext,
    split: SplitResult,
    config: dict[str, Any],
    tuning: Mapping[str, Any],
    model_name: str,
    seed: int,
    primary_metric: str,
    stage: str,
) -> CandidateOutcome:
    model_spaces = tuning.get("models")
    if not isinstance(model_spaces, Mapping):
        raise ValueError("tuning.models must be a mapping")
    model_space = model_spaces.get(model_name)
    if not isinstance(model_space, Mapping):
        raise ValueError(f"No tuning space for {model_name}")

    stage_1_hps = _stage_1_hyperparameters(
        model_name=model_name,
        model_space=model_space,
        seed=seed,
        config=config,
    )

    print()
    print("=" * 90)
    print(f"{model_name}: STAGE 1 ({len(stage_1_hps)} candidates)")
    print("=" * 90)

    stage_1_outcomes = [
        run_or_resume_candidate(
            run=run,
            split=split,
            config=config,
            model_name=model_name,
            seed=seed,
            hyperparameters=hyperparameters,
            stage="stage_1",
            primary_metric=primary_metric,
        )
        for hyperparameters in stage_1_hps
    ]

    stage_1_winner = select_best_candidate(
        stage_1_outcomes,
        primary_metric=primary_metric,
    )
    print(
        f"{model_name} Stage-1 winner: {stage_1_winner.experiment_id} "
        f"{primary_metric}="
        f"{_metric_value(stage_1_winner.result, primary_metric):.6f}"
    )

    if stage == "stage1":
        return stage_1_winner

    stage_2_hps = _stage_2_hyperparameters(
        model_space=model_space,
        stage_1_winner=stage_1_winner,
    )

    stage_2_outcomes: list[CandidateOutcome] = []
    if stage_2_hps:
        print()
        print("=" * 90)
        print(f"{model_name}: STAGE 2 ({len(stage_2_hps)} candidates)")
        print("=" * 90)

    for hyperparameters in stage_2_hps:
        stage_2_outcomes.append(
            run_or_resume_candidate(
                run=run,
                split=split,
                config=config,
                model_name=model_name,
                seed=seed,
                hyperparameters=hyperparameters,
                stage="stage_2",
                primary_metric=primary_metric,
            )
        )

    winner = select_best_candidate(
        [*stage_1_outcomes, *stage_2_outcomes],
        primary_metric=primary_metric,
    )
    print(
        f"{model_name} FINAL VALIDATION WINNER: {winner.experiment_id} "
        f"{primary_metric}={_metric_value(winner.result, primary_metric):.6f}"
    )

    safe_name = model_name.lower().replace("/", "_").replace("-", "_")
    _write_json_consistent(
        run.run_dir / "tuning_selections" / f"{safe_name}.json",
        _selection_payload(
            model_name=model_name,
            primary_metric=primary_metric,
            stage_1=stage_1_outcomes,
            stage_2=stage_2_outcomes,
            winner=winner,
        ),
    )

    return winner


def _dry_run_report(
    *,
    tuning: Mapping[str, Any],
    model_names: list[str],
) -> None:
    model_spaces = tuning.get("models")
    if not isinstance(model_spaces, Mapping):
        raise ValueError("tuning.models must be a mapping")

    print("=== PAPER 2 TUNING DRY RUN ===")
    total_stage_1 = 0
    total_stage_2 = 0

    for model_name in model_names:
        space = model_spaces.get(model_name)
        if not isinstance(space, Mapping):
            raise ValueError(f"No tuning space for {model_name}")
        stage_1 = space.get("stage_1")
        if not isinstance(stage_1, Mapping):
            raise ValueError(f"{model_name}.stage_1 must be a mapping")

        stage_1_count = len(_cartesian_candidates(stage_1))
        stage_2_count = len(space.get("stage_2_at_best_stage_1", []))
        stage_2_count += len(space.get("additional_capacity_checks", []))
        total_stage_1 += stage_1_count
        total_stage_2 += stage_2_count

        print(
            f"{model_name:<16} stage1={stage_1_count:<3} "
            f"stage2_max={stage_2_count:<3} "
            f"total_max={stage_1_count + stage_2_count}"
        )

    print()
    print("Stage-1 total:", total_stage_1)
    print("Stage-2 maximum:", total_stage_2)
    print("Overall maximum:", total_stage_1 + total_stage_2)
    print("Target split:", TARGET_SPLIT)
    print("Test CLI surface: NONE")


def run_tuning(
    *,
    config_path: str | Path,
    tuning_path: str | Path,
    ground_truth_path: str | Path,
    version: str,
    model_selection: str,
    stage: str,
    run_id: str | None,
    results_root: str | Path | None,
    dry_run: bool,
) -> tuple[RunContext | None, dict[str, CandidateOutcome]]:
    config = load_yaml(_resolve_project_path(config_path))
    if not isinstance(config, dict):
        raise TypeError("Loaded publication config must be a mapping")

    tuning = _load_yaml_mapping(tuning_path)
    ground_truth = _load_yaml_mapping(ground_truth_path)

    # Hard test/scientific gate before split/model access.
    selection_seed, primary_metric = _validate_scientific_contract(
        config=config,
        tuning=tuning,
        ground_truth=ground_truth,
        version=version,
    )
    model_names = _model_selection(model_selection)

    if dry_run:
        _dry_run_report(tuning=tuning, model_names=model_names)
        return None, {}

    split = load_frozen_split(config, version=version)

    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("config.paths must be a mapping")
    if results_root is None:
        results_root = paths.get("results_root")
    if not isinstance(results_root, (str, Path)):
        raise ValueError("results_root must be configured or explicitly supplied")

    if run_id is None:
        run_id = f"tuning-ml1m-v1-{_sha256_file(tuning_path)[:8]}"

    run = prepare_run_context(
        config_path=_resolve_project_path(config_path),
        config=config,
        version=version,
        results_root=results_root,
        run_id=run_id,
    )

    _ensure_tuning_manifest(
        run=run,
        tuning_path=tuning_path,
        ground_truth_path=ground_truth_path,
        tuning=tuning,
    )

    winners: dict[str, CandidateOutcome] = {}
    for model_name in model_names:
        winners[model_name] = tune_one_model(
            run=run,
            split=split,
            config=config,
            tuning=tuning,
            model_name=model_name,
            seed=selection_seed,
            primary_metric=primary_metric,
            stage=stage,
        )

    if stage == "all":
        suffix = (
            "all"
            if model_selection == "all"
            else model_selection.lower().replace("/", "_").replace("-", "_")
        )
        _write_json_consistent(
            run.run_dir / f"tuning_summary_{suffix}.json",
            {
                "schema_version": "1.0",
                "selection_split": TARGET_SPLIT,
                "selection_seed": selection_seed,
                "primary_metric": primary_metric,
                "models": {
                    model_name: {
                        "experiment_id": outcome.experiment_id,
                        "metric_value": _metric_value(
                            outcome.result,
                            primary_metric,
                        ),
                        "hyperparameters": dict(outcome.spec.hyperparameters),
                    }
                    for model_name, outcome in winners.items()
                },
            },
        )

    return run, winners


def build_argument_parser() -> argparse.ArgumentParser:
    """Build a validation-only CLI. There is intentionally no test flag."""

    parser = argparse.ArgumentParser(
        description=(
            "Validation-only Paper 2 ML-1M publication hyperparameter tuning."
        )
    )
    parser.add_argument(
        "--config",
        default="config/publication.yaml",
    )
    parser.add_argument(
        "--tuning-config",
        default="config/tuning_spaces_v1.yaml",
    )
    parser.add_argument(
        "--ground-truth",
        default="docs/publication_ground_truth_v1.yaml",
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--model",
        required=True,
        choices=[*MODEL_ORDER, "all"],
    )
    parser.add_argument(
        "--stage",
        choices=["stage1", "all"],
        default="all",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate policy and print candidate counts without loading "
            "the split, creating a run, or fitting any model."
        ),
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    run, winners = run_tuning(
        config_path=args.config,
        tuning_path=args.tuning_config,
        ground_truth_path=args.ground_truth,
        version=args.version,
        model_selection=args.model,
        stage=args.stage,
        run_id=args.run_id,
        results_root=args.results_root,
        dry_run=args.dry_run,
    )

    print()
    print("=" * 90)
    print("VALIDATION-ONLY TUNING RUNNER COMPLETE")
    print("=" * 90)

    if args.dry_run:
        print("Dry run only: no split/model/result access.")
        return 0

    assert run is not None
    tuning = _load_yaml_mapping(args.tuning_config)
    metric_name = tuning["tuning_policy"]["primary_metric"]

    print("Run ID:", run.run_id)
    print("Run directory:", run.run_dir)
    print("Target split:", TARGET_SPLIT)
    print("Test access: PROHIBITED")

    for model_name, outcome in winners.items():
        print(
            f"{model_name}: {outcome.experiment_id} "
            f"{metric_name}={_metric_value(outcome.result, metric_name):.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
