from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.tune_models as tune
from src.experiment import ExperimentSpec


TUNING_PATH = PROJECT_ROOT / "config" / "tuning_spaces_v1.yaml"


def _valid_config() -> dict:
    return {
        "evaluation": {
            "test_access_during_tuning": False,
            "primary_selection_metric": "ndcg_at_10",
        },
        "runtime": {
            "device": "cpu",
        },
        "paths": {
            "results_root": "results",
        },
    }


def _valid_ground_truth() -> dict:
    return {
        "dataset": {
            "id": "ml-1m",
            "artifact_version": "v1",
        },
        "publication_rules": {
            "test_locked_during_tuning": True,
        },
        "evaluation": {
            "test_access_during_tuning": False,
        },
    }


def _minimal_tuning() -> dict:
    return {
        "tuning_policy": {
            "dataset_id": "ml-1m",
            "artifact_version": "v1",
            "selection_split": "validation",
            "selection_seed": 2026,
            "primary_metric": "ndcg_at_10",
            "test_access_allowed": False,
        },
        "models": {
            "PCA/SVD": {
                "stage_1": {
                    "n_components": [32],
                    "n_iter": [7],
                },
                "fixed": {},
            },
        },
    }


def _load_frozen_tuning() -> dict:
    with TUNING_PATH.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    assert isinstance(payload, dict)
    return payload


def _fake_outcome(
    *,
    experiment_id: str,
    metric: float,
    hyperparameters: dict,
    model_name: str = "BernoulliMF",
) -> tune.CandidateOutcome:
    spec = ExperimentSpec(
        model_name=model_name,
        family="bernoulli",
        variant="primary",
        seed=2026,
        hyperparameters=dict(hyperparameters),
    )
    result = SimpleNamespace(
        model_name=model_name,
        ranking_metrics={
            "ndcg_at_10": metric,
        },
    )
    return tune.CandidateOutcome(
        stage="stage_1",
        spec=spec,
        experiment_id=experiment_id,
        result=result,
        resumed=False,
    )


def test_cli_has_no_test_surface():
    parser = tune.build_argument_parser()

    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--target-split" not in option_strings
    assert "--allow-test" not in option_strings

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model",
                "PCA/SVD",
                "--allow-test",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model",
                "PCA/SVD",
                "--target-split",
                "test",
            ]
        )


def test_target_split_constant_is_validation():
    assert tune.TARGET_SPLIT == "validation"


def test_model_inventory_is_exact_and_frozen():
    assert tune.MODEL_ORDER == [
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

    assert tune._model_selection("all") == tune.MODEL_ORDER


@pytest.mark.parametrize(
    "mutation, exception_type",
    [
        (
            lambda config, tuning_cfg, gt: tuning_cfg["tuning_policy"].update(
                {"selection_split": "test"}
            ),
            PermissionError,
        ),
        (
            lambda config, tuning_cfg, gt: tuning_cfg["tuning_policy"].update(
                {"test_access_allowed": True}
            ),
            PermissionError,
        ),
        (
            lambda config, tuning_cfg, gt: config["evaluation"].update(
                {"test_access_during_tuning": True}
            ),
            PermissionError,
        ),
        (
            lambda config, tuning_cfg, gt: gt["publication_rules"].update(
                {"test_locked_during_tuning": False}
            ),
            PermissionError,
        ),
        (
            lambda config, tuning_cfg, gt: gt["evaluation"].update(
                {"test_access_during_tuning": True}
            ),
            PermissionError,
        ),
        (
            lambda config, tuning_cfg, gt: tuning_cfg["tuning_policy"].update(
                {"dataset_id": "ml-latest-small"}
            ),
            ValueError,
        ),
    ],
)
def test_scientific_contract_fails_closed(
    mutation,
    exception_type,
):
    config = _valid_config()
    tuning_cfg = _minimal_tuning()
    ground_truth = _valid_ground_truth()

    mutation(config, tuning_cfg, ground_truth)

    with pytest.raises(exception_type):
        tune._validate_scientific_contract(
            config=config,
            tuning=tuning_cfg,
            ground_truth=ground_truth,
            version="v1",
        )


def test_scientific_contract_returns_frozen_selection_controls():
    seed, metric = tune._validate_scientific_contract(
        config=_valid_config(),
        tuning=_minimal_tuning(),
        ground_truth=_valid_ground_truth(),
        version="v1",
    )

    assert seed == 2026
    assert metric == "ndcg_at_10"


def test_hard_gate_runs_before_split_access(monkeypatch):
    config = _valid_config()
    tuning_cfg = _minimal_tuning()
    tuning_cfg["tuning_policy"]["test_access_allowed"] = True
    ground_truth = _valid_ground_truth()

    monkeypatch.setattr(
        tune,
        "load_yaml",
        lambda path: config,
    )

    def fake_load_yaml_mapping(path):
        text = str(path)
        if "tuning" in text:
            return tuning_cfg
        return ground_truth

    monkeypatch.setattr(
        tune,
        "_load_yaml_mapping",
        fake_load_yaml_mapping,
    )

    def forbidden_split_access(*args, **kwargs):
        pytest.fail("split was accessed before the hard tuning gate")

    monkeypatch.setattr(
        tune,
        "load_frozen_split",
        forbidden_split_access,
    )

    with pytest.raises(PermissionError):
        tune.run_tuning(
            config_path="config/publication.yaml",
            tuning_path="config/tuning_spaces_v1.yaml",
            ground_truth_path="docs/publication_ground_truth_v1.yaml",
            version="v1",
            model_selection="PCA/SVD",
            stage="all",
            run_id=None,
            results_root=None,
            dry_run=False,
        )


def test_dry_run_never_loads_split_creates_run_or_builds_model(monkeypatch):
    config = _valid_config()
    tuning_cfg = _minimal_tuning()
    ground_truth = _valid_ground_truth()

    monkeypatch.setattr(
        tune,
        "load_yaml",
        lambda path: config,
    )

    def fake_load_yaml_mapping(path):
        text = str(path)
        if "tuning" in text:
            return tuning_cfg
        return ground_truth

    monkeypatch.setattr(
        tune,
        "_load_yaml_mapping",
        fake_load_yaml_mapping,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("dry run touched split/model/result infrastructure")

    monkeypatch.setattr(tune, "load_frozen_split", forbidden)
    monkeypatch.setattr(tune, "prepare_run_context", forbidden)
    monkeypatch.setattr(tune, "build_tuning_model", forbidden)
    monkeypatch.setattr(tune, "run_experiment", forbidden)
    monkeypatch.setattr(tune, "write_experiment_result", forbidden)

    run, winners = tune.run_tuning(
        config_path="config/publication.yaml",
        tuning_path="config/tuning_spaces_v1.yaml",
        ground_truth_path="docs/publication_ground_truth_v1.yaml",
        version="v1",
        model_selection="PCA/SVD",
        stage="all",
        run_id=None,
        results_root=None,
        dry_run=True,
    )

    assert run is None
    assert winners == {}


def test_frozen_tuning_budget_is_exactly_102():
    tuning_cfg = _load_frozen_tuning()

    total_stage_1 = 0
    total_stage_2 = 0

    for model_name in tune.MODEL_ORDER:
        space = tuning_cfg["models"][model_name]
        total_stage_1 += len(
            tune._cartesian_candidates(
                space["stage_1"]
            )
        )
        total_stage_2 += len(
            space.get(
                "stage_2_at_best_stage_1",
                [],
            )
        )
        total_stage_2 += len(
            space.get(
                "additional_capacity_checks",
                [],
            )
        )

    assert total_stage_1 == 64
    assert total_stage_2 == 38
    assert total_stage_1 + total_stage_2 == 102


def test_stage_1_hyperparameters_include_seed_device_and_fixed_budget():
    tuning_cfg = _load_frozen_tuning()

    candidates = tune._stage_1_hyperparameters(
        model_name="BernoulliMF",
        model_space=tuning_cfg["models"]["BernoulliMF"],
        seed=2026,
        config={
            "runtime": {
                "device": "cpu",
            }
        },
    )

    assert len(candidates) == 6

    for candidate in candidates:
        assert candidate["seed"] == 2026
        assert candidate["device"] == "cpu"
        assert candidate["epochs"] == 50
        assert candidate["batch_size"] == 1024
        assert candidate["negative_strategy"] == "uniform"
        assert candidate["negative_replace"] is True


def test_pca_stage_1_does_not_receive_device_parameter():
    tuning_cfg = _load_frozen_tuning()

    candidates = tune._stage_1_hyperparameters(
        model_name="PCA/SVD",
        model_space=tuning_cfg["models"]["PCA/SVD"],
        seed=2026,
        config={
            "runtime": {
                "device": "cpu",
            }
        },
    )

    assert len(candidates) == 4
    assert all(candidate["seed"] == 2026 for candidate in candidates)
    assert all("device" not in candidate for candidate in candidates)


def test_ap_emb_stage_2_preserves_winner_and_required_regularization_coverage():
    tuning_cfg = _load_frozen_tuning()
    model_space = tuning_cfg["models"]["ap-emb"]

    base = dict(model_space["fixed"])
    base.update(
        {
            "n_factors": 32,
            "learning_rate": 0.01,
            "seed": 2026,
            "device": "cpu",
        }
    )

    winner = _fake_outcome(
        experiment_id="ap-winner",
        metric=0.2,
        hyperparameters=base,
        model_name="ap-emb",
    )

    stage_2 = tune._stage_2_hyperparameters(
        model_space=model_space,
        stage_1_winner=winner,
    )

    assert len(stage_2) == 4

    for candidate in stage_2:
        assert candidate["n_factors"] == 32
        assert candidate["learning_rate"] == 0.01
        assert candidate["seed"] == 2026
        assert candidate["device"] == "cpu"

    covered = {
        float(base["target_l2"]),
        *{
            float(candidate["target_l2"])
            for candidate in stage_2
        },
    }

    assert {
        0.0,
        1e-8,
        1e-7,
        1e-6,
    }.issubset(covered)


def test_best_candidate_prefers_highest_validation_metric():
    weaker = _fake_outcome(
        experiment_id="a",
        metric=0.20,
        hyperparameters={
            "n_factors": 32,
        },
    )
    stronger = _fake_outcome(
        experiment_id="b",
        metric=0.25,
        hyperparameters={
            "n_factors": 64,
        },
    )

    winner = tune.select_best_candidate(
        [weaker, stronger],
        primary_metric="ndcg_at_10",
    )

    assert winner is stronger


def test_best_candidate_tie_prefers_lower_latent_dimension():
    small = _fake_outcome(
        experiment_id="z",
        metric=0.25,
        hyperparameters={
            "n_factors": 32,
        },
    )
    large = _fake_outcome(
        experiment_id="a",
        metric=0.25,
        hyperparameters={
            "n_factors": 64,
        },
    )

    winner = tune.select_best_candidate(
        [large, small],
        primary_metric="ndcg_at_10",
    )

    assert winner is small


def test_best_candidate_exact_tie_prefers_lexicographically_smaller_id():
    first = _fake_outcome(
        experiment_id="aaa",
        metric=0.25,
        hyperparameters={
            "n_factors": 32,
        },
    )
    second = _fake_outcome(
        experiment_id="bbb",
        metric=0.25,
        hyperparameters={
            "n_factors": 32,
        },
    )

    winner = tune.select_best_candidate(
        [second, first],
        primary_metric="ndcg_at_10",
    )

    assert winner is first


def test_resume_candidate_skips_model_fit_and_write(monkeypatch):
    result = SimpleNamespace(
        status="completed",
        error_message=None,
        model_name="PCA/SVD",
        ranking_metrics={
            "ndcg_at_10": 0.2,
        },
    )

    monkeypatch.setattr(
        tune,
        "experiment_id_for_spec",
        lambda spec, target_split: "exp-resumed",
    )
    monkeypatch.setattr(
        tune,
        "load_completed_experiment",
        lambda **kwargs: (
            result,
            SimpleNamespace(
                experiment_id="exp-resumed"
            ),
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("resumed candidate was rebuilt, refit, or rewritten")

    monkeypatch.setattr(tune, "build_tuning_model", forbidden)
    monkeypatch.setattr(tune, "run_experiment", forbidden)
    monkeypatch.setattr(tune, "write_experiment_result", forbidden)

    outcome = tune.run_or_resume_candidate(
        run=SimpleNamespace(),
        split=SimpleNamespace(),
        config={},
        model_name="PCA/SVD",
        seed=2026,
        hyperparameters={
            "n_components": 32,
            "n_iter": 7,
            "seed": 2026,
        },
        stage="stage_1",
        primary_metric="ndcg_at_10",
    )

    assert outcome.resumed is True
    assert outcome.experiment_id == "exp-resumed"
    assert outcome.result is result


def test_fresh_candidate_is_validation_only_and_written_once(monkeypatch):
    result = SimpleNamespace(
        status="completed",
        error_message=None,
        model_name="PCA/SVD",
        ranking_metrics={
            "ndcg_at_10": 0.21,
        },
    )

    monkeypatch.setattr(
        tune,
        "experiment_id_for_spec",
        lambda spec, target_split: "exp-fresh",
    )
    monkeypatch.setattr(
        tune,
        "load_completed_experiment",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        tune,
        "build_tuning_model",
        lambda **kwargs: SimpleNamespace(),
    )

    captured = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(
        tune,
        "run_experiment",
        fake_run_experiment,
    )

    writes = []

    def fake_write_experiment_result(**kwargs):
        writes.append(kwargs)
        return SimpleNamespace(
            experiment_id="exp-fresh"
        )

    monkeypatch.setattr(
        tune,
        "write_experiment_result",
        fake_write_experiment_result,
    )

    outcome = tune.run_or_resume_candidate(
        run=SimpleNamespace(),
        split=SimpleNamespace(),
        config={
            "runtime": {
                "device": "cpu",
            }
        },
        model_name="PCA/SVD",
        seed=2026,
        hyperparameters={
            "n_components": 32,
            "n_iter": 7,
            "seed": 2026,
        },
        stage="stage_1",
        primary_metric="ndcg_at_10",
    )

    assert outcome.resumed is False
    assert captured["target_split"] == "validation"
    assert captured["allow_test"] is False
    assert len(writes) == 1


def test_noncompleted_candidate_fails_closed(monkeypatch):
    result = SimpleNamespace(
        status="failed",
        error_message="synthetic failure",
        model_name="PCA/SVD",
        ranking_metrics={
            "ndcg_at_10": 0.0,
        },
    )

    monkeypatch.setattr(
        tune,
        "experiment_id_for_spec",
        lambda spec, target_split: "exp-failed",
    )
    monkeypatch.setattr(
        tune,
        "load_completed_experiment",
        lambda **kwargs: (
            result,
            SimpleNamespace(
                experiment_id="exp-failed"
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="did not complete successfully"):
        tune.run_or_resume_candidate(
            run=SimpleNamespace(),
            split=SimpleNamespace(),
            config={},
            model_name="PCA/SVD",
            seed=2026,
            hyperparameters={
                "n_components": 32,
                "n_iter": 7,
                "seed": 2026,
            },
            stage="stage_1",
            primary_metric="ndcg_at_10",
        )


def test_json_consistency_is_idempotent_and_fails_on_mismatch(tmp_path):
    path = tmp_path / "selection.json"

    tune._write_json_consistent(
        path,
        {
            "a": 1,
            "b": [2, 3],
        },
    )

    tune._write_json_consistent(
        path,
        {
            "b": [2, 3],
            "a": 1,
        },
    )

    with pytest.raises(ValueError, match="differs from requested payload"):
        tune._write_json_consistent(
            path,
            {
                "a": 999,
            },
        )
