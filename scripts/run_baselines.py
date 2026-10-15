"""Run Paper 2 simple baseline experiments.

This runner owns exactly two models:

- Popularity
- Random

PCA/SVD is intentionally excluded because it is owned by
scripts/run_mf.py.

Pipeline
--------
config
    -> frozen canonical split artifacts
    -> baseline construction
    -> ExperimentSpec
    -> verified completed-checkpoint lookup
    -> run_experiment() only when needed
    -> structured result storage

Validation is the default target. Test evaluation requires explicit
authorization through both:

    --target-split test
    --allow-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


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


from scripts.run_mf import (
    load_frozen_split,
    prepare_run_context,
    resolve_seed,
)

from src.data import load_yaml
from src.experiment import (
    ExperimentResult,
    ExperimentSpec,
    run_experiment,
)
from src.models.baselines import (
    PopularityBaseline,
    RandomBaseline,
)
from src.results import (
    RunContext,
    load_completed_experiment,
    write_experiment_result,
)
from src.split import SplitResult


# ============================================================
# Frozen baseline inventory
# ============================================================


BASELINE_ORDER = [
    "popularity",
    "random",
]


BASELINE_CHOICES = (
    BASELINE_ORDER
    +
    [
        "all",
    ]
)


BASELINE_METADATA = {
    "popularity": {
        "model_name": "Popularity",
        "family": "baseline",
        "variant": "primary",
    },
    "random": {
        "model_name": "Random",
        "family": "baseline",
        "variant": "primary",
    },
}


# ============================================================
# Path helper
# ============================================================


def _resolve_project_path(
    path: str | Path,
) -> Path:
    """Resolve one path relative to the Paper 2 project root."""

    path = Path(
        str(
            path
        ).replace(
            "\\",
            "/",
        )
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        /
        path
    )


# ============================================================
# Baseline selection
# ============================================================


def resolve_baseline_keys(
    requested: str,
) -> list[str]:
    """Expand one CLI baseline selection."""

    if requested == "all":
        return list(
            BASELINE_ORDER
        )

    if requested not in BASELINE_ORDER:
        raise ValueError(
            f"Unknown baseline selection: {requested}"
        )

    return [
        requested
    ]


# ============================================================
# Model construction
# ============================================================


def build_baseline(
    baseline_key: str,
    *,
    split: SplitResult,
    config: dict[str, Any],
    seed: int,
):
    """Construct one simple baseline and explicit hyperparameters."""

    if baseline_key not in BASELINE_ORDER:
        raise ValueError(
            f"Unknown baseline key: {baseline_key}"
        )

    if not isinstance(
        split,
        SplitResult,
    ):
        raise TypeError(
            "split must be a SplitResult"
        )

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "config must be a dictionary"
        )

    if (
        not isinstance(
            seed,
            int,
        )
        or isinstance(
            seed,
            bool,
        )
    ):
        raise ValueError(
            "seed must be an integer"
        )

    # --------------------------------------------------------
    # Popularity
    # --------------------------------------------------------

    if baseline_key == "popularity":

        hyperparameters: dict[
            str,
            Any,
        ] = {}

        model = (
            PopularityBaseline()
        )

        return (
            model,
            hyperparameters,
        )

    # --------------------------------------------------------
    # Random
    # --------------------------------------------------------

    if baseline_key == "random":

        hyperparameters = {
            "seed": seed,
            "namespace": (
                "random_baseline"
            ),
        }

        model = RandomBaseline(
            **hyperparameters
        )

        return (
            model,
            hyperparameters,
        )

    raise AssertionError(
        "unreachable baseline construction path"
    )


# ============================================================
# Experiment specification
# ============================================================


def build_experiment_spec(
    baseline_key: str,
    *,
    seed: int,
    hyperparameters: dict[str, Any],
) -> ExperimentSpec:
    """Build one Step-43 ExperimentSpec."""

    if baseline_key not in BASELINE_METADATA:
        raise ValueError(
            f"Unknown baseline key: {baseline_key}"
        )

    metadata = (
        BASELINE_METADATA[
            baseline_key
        ]
    )

    return ExperimentSpec(
        model_name=(
            metadata[
                "model_name"
            ]
        ),
        family=(
            metadata[
                "family"
            ]
        ),
        variant=(
            metadata[
                "variant"
            ]
        ),
        seed=seed,
        hyperparameters=(
            hyperparameters
        ),
    )


# ============================================================
# Runner
# ============================================================


def run_selected_baselines(
    *,
    config_path: str | Path,
    version: str,
    baseline_selection: str,
    seed: int | None = None,
    run_id: str | None = None,
    results_root: str | Path | None = None,
    target_split: str = "validation",
    allow_test: bool = False,
) -> tuple[
    RunContext,
    list[ExperimentResult],
]:
    """Execute selected simple baselines."""

    config_path = (
        _resolve_project_path(
            config_path
        )
    )

    config = load_yaml(
        config_path
    )

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "Loaded configuration must be a dictionary"
        )

    if target_split not in {
        "validation",
        "test",
    }:
        raise ValueError(
            "target_split must be "
            "'validation' or 'test'"
        )

    if not isinstance(
        allow_test,
        bool,
    ):
        raise TypeError(
            "allow_test must be boolean"
        )

    # --------------------------------------------------------
    # Hard test gate before split/model access.
    # --------------------------------------------------------

    if (
        target_split
        ==
        "test"
        and not allow_test
    ):
        raise PermissionError(
            "Test evaluation is disabled. "
            "Use --allow-test only for an "
            "explicitly authorized final evaluation."
        )

    split = load_frozen_split(
        config,
        version=version,
    )

    resolved_seed = (
        resolve_seed(
            config,
            seed,
        )
    )

    paths_config = config.get(
        "paths"
    )

    if not isinstance(
        paths_config,
        dict,
    ):
        raise ValueError(
            "config.paths must be a mapping"
        )

    if results_root is None:

        results_root = (
            paths_config.get(
                "results_root"
            )
        )

    if (
        not isinstance(
            results_root,
            (
                str,
                Path,
            ),
        )
    ):
        raise ValueError(
            "results_root must be configured "
            "or explicitly supplied"
        )

    run = prepare_run_context(
        config_path=config_path,
        config=config,
        version=version,
        results_root=results_root,
        run_id=run_id,
    )

    baseline_keys = (
        resolve_baseline_keys(
            baseline_selection
        )
    )

    results: list[
        ExperimentResult
    ] = []

    for baseline_key in baseline_keys:

        metadata = (
            BASELINE_METADATA[
                baseline_key
            ]
        )

        print()
        print(
            "="
            *
            72
        )

        print(
            "Running:",
            metadata[
                "model_name"
            ],
        )

        print(
            "Run ID:",
            run.run_id,
        )

        print(
            "Seed:",
            resolved_seed,
        )

        print(
            "Target split:",
            target_split,
        )

        print(
            "="
            *
            72
        )

        model, hyperparameters = (
            build_baseline(
                baseline_key,
                split=split,
                config=config,
                seed=resolved_seed,
            )
        )

        spec = (
            build_experiment_spec(
                baseline_key,
                seed=resolved_seed,
                hyperparameters=(
                    hyperparameters
                ),
            )
        )

        resumed = (
            load_completed_experiment(
                run=run,
                spec=spec,
                target_split=target_split,
            )
        )

        if resumed is not None:

            result, artifacts = (
                resumed
            )

            print(
                "Resume:",
                "SKIPPED - verified completed experiment",
            )

        else:

            result = run_experiment(
                model=model,
                split=split,
                config=config,
                spec=spec,
                target_split=target_split,
                allow_test=allow_test,
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

        primary_metric = (
            config.get(
                "evaluation",
                {},
            )
            .get(
                "primary_selection_metric"
            )
        )

        if (
            isinstance(
                primary_metric,
                str,
            )
            and primary_metric
            in result.ranking_metrics
        ):
            print(
                primary_metric,
                "=",
                f"{result.ranking_metrics[primary_metric]:.6f}",
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


# ============================================================
# CLI
# ============================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Paper 2 simple baselines "
            "on a frozen canonical split."
        )
    )

    parser.add_argument(
        "--config",
        default="config/dev.yaml",
        help=(
            "Experiment YAML configuration. "
            "Default: config/dev.yaml"
        ),
    )

    parser.add_argument(
        "--version",
        default="v1",
        help=(
            "Frozen artifact version. "
            "Default: v1"
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=BASELINE_CHOICES,
        help=(
            "Baseline to run, or 'all' "
            "for both baselines."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Experiment seed. "
            "Defaults to reproducibility.global_seed."
        ),
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional run ID. Existing matching "
            "runs may be safely reopened."
        ),
    )

    parser.add_argument(
        "--results-root",
        default=None,
        help=(
            "Optional results root override."
        ),
    )

    parser.add_argument(
        "--target-split",
        choices=[
            "validation",
            "test",
        ],
        default="validation",
        help=(
            "Evaluation target. "
            "Default: validation."
        ),
    )

    parser.add_argument(
        "--allow-test",
        action="store_true",
        help=(
            "Explicitly authorize final test evaluation. "
            "Never use during tuning."
        ),
    )

    return parser


def main() -> int:
    """CLI entry point."""

    parser = (
        build_argument_parser()
    )

    args = parser.parse_args()

    run, results = (
        run_selected_baselines(
            config_path=args.config,
            version=args.version,
            baseline_selection=(
                args.model
            ),
            seed=args.seed,
            run_id=args.run_id,
            results_root=(
                args.results_root
            ),
            target_split=(
                args.target_split
            ),
            allow_test=(
                args.allow_test
            ),
        )
    )

    print()
    print(
        "="
        *
        72
    )

    print(
        "BASELINE RUNNER COMPLETE"
    )

    print(
        "Run ID:",
        run.run_id,
    )

    print(
        "Experiments completed:",
        len(
            results
        ),
    )

    print(
        "Run directory:",
        run.run_dir,
    )

    print(
        "="
        *
        72
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )