from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

import scripts.run_stability_ablation as stability


def _frozen_hp():
    return {
        "tuned_models": {
            "g-emb": {
                "hyperparameters": {
                    "batch_size": 2048,
                    "bias_l2": 1e-5,
                    "context_l2": 1e-5,
                    "context_normalization": "mean",
                    "device": "cpu",
                    "epochs": 50,
                    "learning_rate": 0.05,
                    "max_grad_norm": 5.0,
                    "n_factors": 64,
                    "target_l2": 1e-5,
                }
            },
            "ng-emb": {
                "hyperparameters": {
                    "batch_size": 2048,
                    "bias_l2": 0.0,
                    "context_l2": 0.0,
                    "context_normalization": "mean",
                    "device": "cpu",
                    "epochs": 50,
                    "learning_rate": 0.003,
                    "max_grad_norm": 5.0,
                    "n_factors": 32,
                    "target_l2": 0.0,
                }
            },
        }
    }


def test_frozen_protocol_constants_are_exact():
    assert stability.STABILITY_PROTOCOL_SHA256 == (
        "1e27f0b23684b83cccb2dd625110c597"
        "579e0d6f42489d191f06b45264360664"
    )

    assert stability.FINAL_HYPERPARAMETERS_SHA256 == (
        "9aae84dd59d7c34c82a67563139bdb272"
        "56874ba79c645c749cbcf788d72c981"
    )

    assert stability.RUN_ID == (
        "stability-ml1m-v1-1e27f0b2"
    )


def test_frozen_matrix_is_exactly_18():
    matrix = stability._planned_matrix(
        model_filter="all",
        normalization_filter="all",
        seed_filter=None,
    )

    assert len(matrix) == 18
    assert len(set(matrix)) == 18

    assert {
        model
        for model, _, _
        in matrix
    } == {
        "g-emb",
        "ng-emb",
    }

    assert {
        normalization
        for _, normalization, _
        in matrix
    } == {
        "raw",
        "sqrt",
        "mean",
    }

    assert {
        seed
        for _, _, seed
        in matrix
    } == {
        2026,
        2027,
        2028,
    }


@pytest.mark.parametrize(
    "model_name",
    ["g-emb", "ng-emb"],
)
@pytest.mark.parametrize(
    "normalization",
    ["raw", "sqrt", "mean"],
)
@pytest.mark.parametrize(
    "seed",
    [2026, 2027, 2028],
)
def test_hyperparameter_override_changes_only_normalization_and_seed(
    model_name,
    normalization,
    seed,
):
    frozen = _frozen_hp()

    source = deepcopy(
        frozen[
            "tuned_models"
        ][
            model_name
        ][
            "hyperparameters"
        ]
    )

    result = stability._build_hyperparameters(
        frozen_hp=frozen,
        model_name=model_name,
        normalization=normalization,
        seed=seed,
    )

    assert result["context_normalization"] == normalization
    assert result["seed"] == seed

    for key, value in source.items():
        if key == "context_normalization":
            continue
        assert result[key] == value

    assert "seed" not in source
    assert source["context_normalization"] == "mean"


@pytest.mark.parametrize(
    "normalization",
    ["raw", "sqrt", "mean"],
)
def test_spec_is_explicitly_stability_variant(normalization):
    hp = stability._build_hyperparameters(
        frozen_hp=_frozen_hp(),
        model_name="g-emb",
        normalization=normalization,
        seed=2026,
    )

    spec = stability._build_spec(
        model_name="g-emb",
        normalization=normalization,
        seed=2026,
        hyperparameters=hp,
    )

    assert spec.model_name == "g-emb"
    assert spec.family == "gaussian"
    assert spec.variant == f"stability-{normalization}"
    assert spec.seed == 2026
    assert spec.hyperparameters == hp


def test_distribution_summary_schema_and_values():
    values = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dtype=np.float64,
    )

    summary = stability._distribution_summary(
        values
    )

    assert tuple(summary.keys()) == (
        stability.DISTRIBUTION_KEYS
    )

    assert summary["count"] == 5
    assert summary["min"] == 1.0
    assert summary["mean"] == 3.0
    assert summary["p50"] == 3.0
    assert summary["max"] == 5.0


def test_numerical_failure_classifier_is_narrow():
    assert stability._is_numerical_stability_failure(
        RuntimeError(
            "g-emb produced non-finite predictions"
        )
    )

    assert stability._is_numerical_stability_failure(
        RuntimeError(
            "gradient norm became NaN"
        )
    )

    assert not stability._is_numerical_stability_failure(
        FileNotFoundError(
            "missing publication artifact"
        )
    )

    assert not stability._is_numerical_stability_failure(
        ValueError(
            "wrong experiment identity"
        )
    )


def test_protocol_manifest_forbids_test_and_retuning():
    payload = stability._protocol_manifest_payload()

    assert payload["evaluation_partition"] == "validation"
    assert payload["test_access_allowed"] is False
    assert payload["primary_results_affected"] is False
    assert payload["retuning_allowed"] is False
    assert payload["planned_experiments"] == 18


def test_failure_marker_validation_accepts_matching_payload(tmp_path):
    experiment_id = "g-emb-stability-raw-s2026-validation-example"

    path = tmp_path / "failure.json"

    payload = {
        "schema_version": "1.0",
        "freeze_id": "paper2-efemb-gaussian-stability-v1",
        "stability_protocol_sha256": (
            stability.STABILITY_PROTOCOL_SHA256
        ),
        "final_hyperparameters_sha256": (
            stability.FINAL_HYPERPARAMETERS_SHA256
        ),
        "run_id": stability.RUN_ID,
        "experiment_id": experiment_id,
        "model_name": "g-emb",
        "variant": "stability-raw",
        "context_normalization": "raw",
        "seed": 2026,
        "target_split": "validation",
        "test_metrics_computed": False,
        "status": "failed_numerical_stability",
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    loaded = stability._verified_failure_marker(
        path,
        experiment_id=experiment_id,
        model_name="g-emb",
        normalization="raw",
        seed=2026,
    )

    assert loaded is not None
    assert loaded["status"] == "failed_numerical_stability"


def test_failure_marker_rejects_test_claim(tmp_path):
    experiment_id = "g-emb-stability-raw-s2026-validation-example"

    path = tmp_path / "failure.json"

    payload = {
        "schema_version": "1.0",
        "freeze_id": "paper2-efemb-gaussian-stability-v1",
        "stability_protocol_sha256": (
            stability.STABILITY_PROTOCOL_SHA256
        ),
        "final_hyperparameters_sha256": (
            stability.FINAL_HYPERPARAMETERS_SHA256
        ),
        "run_id": stability.RUN_ID,
        "experiment_id": experiment_id,
        "model_name": "g-emb",
        "variant": "stability-raw",
        "context_normalization": "raw",
        "seed": 2026,
        "target_split": "validation",
        "test_metrics_computed": True,
        "status": "failed_numerical_stability",
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="test access",
    ):
        stability._verified_failure_marker(
            path,
            experiment_id=experiment_id,
            model_name="g-emb",
            normalization="raw",
            seed=2026,
        )


def test_cli_has_no_test_execution_option():
    parser = stability.build_argument_parser()

    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--target-split" not in option_strings
    assert "--allow-test" not in option_strings
    assert "--execute-final-test" not in option_strings


def test_default_cli_is_preflight_only():
    parser = stability.build_argument_parser()
    args = parser.parse_args([])

    assert args.execute is False
    assert args.model == "all"
    assert args.normalization == "all"
    assert args.seed is None

def _valid_persisted_distribution_summary():
    # Intentionally use JSON-style sorted-key order rather than
    # DISTRIBUTION_KEYS insertion order. This reproduces the historical
    # Step 57 resume-validation failure mode.
    return {
        "count": 100,
        "max": 5.0,
        "mean": 3.0,
        "min": 1.0,
        "p50": 3.0,
        "p75": 4.0,
        "p90": 4.5,
        "p95": 4.75,
        "p99": 4.95,
        "std": 1.0,
    }


def test_persisted_distribution_schema_accepts_reordered_valid_keys(tmp_path):
    summary = _valid_persisted_distribution_summary()

    assert tuple(summary.keys()) != stability.DISTRIBUTION_KEYS

    stability._validate_distribution_summary_payload(
        summary,
        path=tmp_path / "diagnostic.json",
        quantity="positive_training_context_norm",
    )


def test_persisted_distribution_schema_rejects_missing_key(tmp_path):
    summary = _valid_persisted_distribution_summary()
    summary.pop("p99")

    with pytest.raises(
        ValueError,
        match="schema mismatch",
    ):
        stability._validate_distribution_summary_payload(
            summary,
            path=tmp_path / "diagnostic.json",
            quantity="positive_training_context_norm",
        )


def test_persisted_distribution_schema_rejects_extra_key(tmp_path):
    summary = _valid_persisted_distribution_summary()
    summary["unexpected"] = 1.0

    with pytest.raises(
        ValueError,
        match="schema mismatch",
    ):
        stability._validate_distribution_summary_payload(
            summary,
            path=tmp_path / "diagnostic.json",
            quantity="positive_training_context_norm",
        )


def test_persisted_distribution_schema_rejects_nonfinite_value(tmp_path):
    summary = _valid_persisted_distribution_summary()
    summary["p99"] = float("nan")

    with pytest.raises(
        ValueError,
        match="invalid numeric value",
    ):
        stability._validate_distribution_summary_payload(
            summary,
            path=tmp_path / "diagnostic.json",
            quantity="positive_training_context_norm",
        )


def test_all_18_existing_stability_diagnostics_resume_validate_read_only():
    diagnostics_root = (
        stability.PROJECT_ROOT
        / "results"
        / stability.RUN_ID
        / "stability_diagnostics"
    )

    paths = sorted(diagnostics_root.glob("*.json"))

    assert len(paths) == 18

    before = {
        path: stability._sha256_file(path)
        for path in paths
    }

    for path in paths:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        loaded = stability._verified_diagnostic_sidecar(
            path,
            experiment_id=payload["experiment_id"],
            model_name=payload["model_name"],
            normalization=payload["context_normalization"],
            seed=int(payload["seed"]),
        )

        assert loaded == payload

    after = {
        path: stability._sha256_file(path)
        for path in paths
    }

    assert after == before

