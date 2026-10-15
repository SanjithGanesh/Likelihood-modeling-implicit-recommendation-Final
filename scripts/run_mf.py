"""Run Paper 2 matrix-factorization and literature baseline experiments.

This runner executes the following model group:

- PCA/SVD
- BernoulliMF
- PoissonMF
- GaussianMF
- LogisticPCA
- HPF

Pipeline
--------
config
    -> frozen canonical split artifacts
    -> leakage-safe model construction
    -> ExperimentSpec
    -> verified completed-checkpoint lookup
    -> run_experiment() only when needed
    -> structured result storage

The runner never creates a new train/validation/test split.

Validation is the default evaluation target. Test evaluation requires
both:

    --target-split test
    --allow-test

which preserves the Paper 2 validation-only tuning contract.

Completed experiments are safe checkpoint boundaries. When an existing
matching run is reopened, verified completed experiments are loaded and
skipped; incomplete/corrupt/mismatched checkpoints fail closed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


from src.data import load_yaml
from src.experiment import (
    ExperimentResult,
    ExperimentSpec,
    run_experiment,
)
from src.models.bernoulli_mf import BernoulliMF
from src.models.gaussian_mf import GaussianMF
from src.models.hpf import HPF
from src.models.logistic_pca import LogisticPCA
from src.models.pca_svd import PCASVDBaseline
from src.models.poisson_mf import PoissonMF
from src.results import (
    RunContext,
    build_run_provenance,
    create_run,
    load_completed_experiment,
    load_run_context,
    write_experiment_result,
)
from src.sampling import NegativeSampler
from src.split import SplitResult


# ============================================================
# Frozen runner inventory
# ============================================================


MODEL_ORDER = [
    "pca_svd",
    "bernoulli_mf",
    "poisson_mf",
    "gaussian_mf",
    "logistic_pca",
    "hpf",
]


MODEL_CHOICES = (
    MODEL_ORDER
    +
    [
        "all",
    ]
)


MODEL_METADATA = {
    "pca_svd": {
        "model_name": "PCA/SVD",
        "family": "baseline",
        "variant": "primary",
    },
    "bernoulli_mf": {
        "model_name": "BernoulliMF",
        "family": "bernoulli",
        "variant": "primary",
    },
    "poisson_mf": {
        "model_name": "PoissonMF",
        "family": "poisson",
        "variant": "primary",
    },
    "gaussian_mf": {
        "model_name": "GaussianMF",
        "family": "gaussian",
        "variant": "primary",
    },
    "logistic_pca": {
        "model_name": "LogisticPCA",
        "family": "bernoulli",
        "variant": "primary",
    },
    "hpf": {
        "model_name": "HPF",
        "family": "poisson",
        "variant": "primary",
    },
}


# ============================================================
# Path helpers
# ============================================================


def _resolve_project_path(
    path: str | Path,
) -> Path:
    """Resolve a config/manifest path relative to the project root."""

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


def _load_yaml_mapping(
    path: str | Path,
) -> dict[
    str,
    Any,
]:
    """Load one YAML mapping."""

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

        data = yaml.safe_load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            f"YAML root must be a mapping: {path}"
        )

    return data


# ============================================================
# Frozen artifact locations
# ============================================================


def artifact_paths(
    config: dict[
        str,
        Any,
    ],
    *,
    version: str,
) -> dict[
    str,
    Path,
]:
    """Resolve canonical processed/split artifact paths."""

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "config must be a dictionary"
        )

    if not isinstance(
        version,
        str,
    ):
        raise TypeError(
            "version must be a string"
        )

    if not version.strip():
        raise ValueError(
            "version must not be empty"
        )

    dataset = config.get(
        "dataset"
    )

    if not isinstance(
        dataset,
        dict,
    ):
        raise ValueError(
            "config.dataset must be a mapping"
        )

    dataset_id = dataset.get(
        "id"
    )

    processed_root = dataset.get(
        "processed_root"
    )

    split_root = dataset.get(
        "split_root"
    )

    for name, value in [
        (
            "dataset.id",
            dataset_id,
        ),
        (
            "dataset.processed_root",
            processed_root,
        ),
        (
            "dataset.split_root",
            split_root,
        ),
    ]:

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                f"{name} must be a nonempty string"
            )

    processed_dir = (
        _resolve_project_path(
            processed_root
        )
        /
        dataset_id
        /
        version
    )

    split_dir = (
        _resolve_project_path(
            split_root
        )
        /
        dataset_id
        /
        version
    )

    return {
        "processed_dir": (
            processed_dir
        ),
        "split_dir": (
            split_dir
        ),
        "preparation_manifest": (
            processed_dir
            /
            "preparation_manifest.yaml"
        ),
        "split_manifest": (
            split_dir
            /
            "split_manifest.yaml"
        ),
    }


# ============================================================
# Frozen split reconstruction
# ============================================================


def load_frozen_split(
    config: dict[
        str,
        Any,
    ],
    *,
    version: str,
) -> SplitResult:
    """Load the already-frozen canonical split from disk.

    This function never calls split_ratings() or split_from_config().
    """

    paths = artifact_paths(
        config,
        version=version,
    )

    manifest = (
        _load_yaml_mapping(
            paths[
                "split_manifest"
            ]
        )
    )

    artifact = manifest.get(
        "artifact"
    )

    protocol = manifest.get(
        "protocol"
    )

    statistics = manifest.get(
        "statistics"
    )

    files = manifest.get(
        "files"
    )

    for name, value in [
        (
            "artifact",
            artifact,
        ),
        (
            "protocol",
            protocol,
        ),
        (
            "statistics",
            statistics,
        ),
        (
            "files",
            files,
        ),
    ]:

        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                "split manifest "
                f"{name} must be a mapping"
            )

    if (
        artifact.get(
            "type"
        )
        !=
        "canonical_split"
    ):
        raise ValueError(
            "split manifest artifact.type "
            "must be 'canonical_split'"
        )

    dataset = config.get(
        "dataset"
    )

    if not isinstance(
        dataset,
        dict,
    ):
        raise ValueError(
            "config.dataset must be a mapping"
        )

    dataset_id = dataset.get(
        "id"
    )

    if (
        artifact.get(
            "dataset_id"
        )
        !=
        dataset_id
    ):
        raise ValueError(
            "split manifest dataset_id "
            "does not match configuration"
        )

    if (
        artifact.get(
            "version"
        )
        !=
        version
    ):
        raise ValueError(
            "split manifest version "
            "does not match requested version"
        )

    frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        file_entry = files.get(
            split_name
        )

        if not isinstance(
            file_entry,
            dict,
        ):
            raise ValueError(
                "split manifest missing "
                f"files.{split_name}"
            )

        raw_path = file_entry.get(
            "path"
        )

        if (
            not isinstance(
                raw_path,
                str,
            )
            or not raw_path.strip()
        ):
            raise ValueError(
                "split manifest "
                f"files.{split_name}.path "
                "must be a nonempty string"
            )

        path = _resolve_project_path(
            raw_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Frozen split file missing: {path}"
            )

        frame = pd.read_csv(
            path
        )

        required_columns = {
            "userId",
            "movieId",
            "rating",
        }

        missing = (
            required_columns
            -
            set(
                frame.columns
            )
        )

        if missing:
            raise ValueError(
                f"{split_name} split missing "
                f"required columns: {sorted(missing)}"
            )

        if len(
            frame
        ) == 0:
            raise ValueError(
                f"{split_name} split must not be empty"
            )

        frames[
            split_name
        ] = frame

    expected_rows = {
        "train": statistics.get(
            "train_rows"
        ),
        "validation": statistics.get(
            "validation_rows"
        ),
        "test": statistics.get(
            "test_rows"
        ),
    }

    for split_name, expected in expected_rows.items():

        if (
            not isinstance(
                expected,
                int,
            )
            or isinstance(
                expected,
                bool,
            )
        ):
            raise ValueError(
                "split manifest statistics."
                f"{split_name}_rows must be an integer"
            )

        actual = len(
            frames[
                split_name
            ]
        )

        if actual != expected:
            raise ValueError(
                f"{split_name} row count mismatch: "
                f"manifest={expected}, actual={actual}"
            )

    seed = protocol.get(
        "seed"
    )

    attempt = protocol.get(
        "constrained_attempt"
    )

    coverage = protocol.get(
        "enforce_train_item_coverage"
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
            "split manifest protocol.seed "
            "must be an integer"
        )

    if (
        not isinstance(
            attempt,
            int,
        )
        or isinstance(
            attempt,
            bool,
        )
        or attempt < 1
    ):
        raise ValueError(
            "split manifest protocol."
            "constrained_attempt must be "
            "a positive integer"
        )

    if not isinstance(
        coverage,
        bool,
    ):
        raise ValueError(
            "split manifest protocol."
            "enforce_train_item_coverage "
            "must be boolean"
        )

    return SplitResult(
        train=frames[
            "train"
        ],
        validation=frames[
            "validation"
        ],
        test=frames[
            "test"
        ],
        seed=seed,
        attempt=attempt,
        enforce_train_item_coverage=(
            coverage
        ),
    )


# ============================================================
# Model construction helpers
# ============================================================


def _all_observed(
    split: SplitResult,
) -> pd.DataFrame:
    """Return canonical observed interactions across all partitions."""

    return pd.concat(
        [
            split.train,
            split.validation,
            split.test,
        ],
        ignore_index=True,
    )


def _heldout_interactions(
    split: SplitResult,
) -> pd.DataFrame:
    """Return validation+test positives for leakage-prevention masks."""

    return pd.concat(
        [
            split.validation,
            split.test,
        ],
        ignore_index=True,
    )


def _shared_training_values(
    config: dict[
        str,
        Any,
    ],
    *,
    seed: int,
) -> dict[
    str,
    Any,
]:
    """Extract shared frozen development training settings."""

    training = config.get(
        "training"
    )

    runtime = config.get(
        "runtime"
    )

    if not isinstance(
        training,
        dict,
    ):
        raise ValueError(
            "config.training must be a mapping"
        )

    if not isinstance(
        runtime,
        dict,
    ):
        raise ValueError(
            "config.runtime must be a mapping"
        )

    batch_size = training.get(
        "batch_size"
    )

    epochs = training.get(
        "default_epochs"
    )

    learning_rate = training.get(
        "default_learning_rate"
    )

    device = runtime.get(
        "device"
    )

    if (
        not isinstance(
            batch_size,
            int,
        )
        or isinstance(
            batch_size,
            bool,
        )
        or batch_size <= 0
    ):
        raise ValueError(
            "training.batch_size must be "
            "a positive integer"
        )

    if (
        not isinstance(
            epochs,
            int,
        )
        or isinstance(
            epochs,
            bool,
        )
        or epochs <= 0
    ):
        raise ValueError(
            "training.default_epochs must be "
            "a positive integer"
        )

    if (
        not isinstance(
            learning_rate,
            (
                int,
                float,
            ),
        )
        or isinstance(
            learning_rate,
            bool,
        )
        or float(
            learning_rate
        ) <= 0.0
    ):
        raise ValueError(
            "training.default_learning_rate "
            "must be positive"
        )

    if (
        not isinstance(
            device,
            str,
        )
        or not device.strip()
    ):
        raise ValueError(
            "runtime.device must be "
            "a nonempty string"
        )

    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": float(
            learning_rate
        ),
        "device": device,
        "seed": seed,
    }


def build_model(
    model_key: str,
    *,
    split: SplitResult,
    config: dict[
        str,
        Any,
    ],
    seed: int,
):
    """Construct one frozen development MF/literature model.

    Returns
    -------
    tuple
        (model, explicit_hyperparameter_dictionary)
    """

    if model_key not in MODEL_ORDER:
        raise ValueError(
            f"Unknown model key: {model_key}"
        )

    shared = (
        _shared_training_values(
            config,
            seed=seed,
        )
    )

    # --------------------------------------------------------
    # PCA/SVD
    # --------------------------------------------------------

    if model_key == "pca_svd":

        hyperparameters = {
            "n_components": 32,
            "seed": seed,
            "n_iter": 7,
        }

        model = PCASVDBaseline(
            **hyperparameters
        )

        return (
            model,
            hyperparameters,
        )

    # --------------------------------------------------------
    # BernoulliMF
    # --------------------------------------------------------

    if model_key == "bernoulli_mf":

        sampler = NegativeSampler(
            train=split.train,
            all_observed=(
                _all_observed(
                    split
                )
            ),
            seed=seed,
            namespace=(
                "bernoulli_mf_negative_sampling"
            ),
            popularity_alpha=1.0,
        )

        hyperparameters = {
            "n_factors": 32,
            "learning_rate": (
                shared[
                    "learning_rate"
                ]
            ),
            "epochs": (
                shared[
                    "epochs"
                ]
            ),
            "batch_size": (
                shared[
                    "batch_size"
                ]
            ),
            "negative_ratio": 5,
            "negative_weight": 1.0,
            "factor_l2": 1e-6,
            "bias_l2": 1e-6,
            "negative_strategy": "uniform",
            "negative_replace": True,
            "max_grad_norm": 5.0,
            "seed": seed,
            "device": (
                shared[
                    "device"
                ]
            ),
        }

        model = BernoulliMF(
            negative_sampler=sampler,
            **hyperparameters,
        )

        return (
            model,
            hyperparameters,
        )

    # --------------------------------------------------------
    # PoissonMF
    # --------------------------------------------------------

    if model_key == "poisson_mf":

        sampler = NegativeSampler(
            train=split.train,
            all_observed=(
                _all_observed(
                    split
                )
            ),
            seed=seed,
            namespace=(
                "poisson_mf_zero_sampling"
            ),
            popularity_alpha=1.0,
        )

        hyperparameters = {
            "n_factors": 32,
            "learning_rate": (
                shared[
                    "learning_rate"
                ]
            ),
            "epochs": (
                shared[
                    "epochs"
                ]
            ),
            "batch_size": (
                shared[
                    "batch_size"
                ]
            ),
            "zero_ratio": 5,
            "factor_l2": 1e-6,
            "bias_l2": 1e-6,
            "zero_strategy": "uniform",
            "zero_replace": True,
            "log_rate_min": -20.0,
            "log_rate_max": 12.0,
            "max_grad_norm": 5.0,
            "seed": seed,
            "device": (
                shared[
                    "device"
                ]
            ),
        }

        model = PoissonMF(
            negative_sampler=sampler,
            **hyperparameters,
        )

        return (
            model,
            hyperparameters,
        )

    # --------------------------------------------------------
    # GaussianMF
    # --------------------------------------------------------

    if model_key == "gaussian_mf":

        hyperparameters = {
            "n_factors": 32,
            "learning_rate": (
                shared[
                    "learning_rate"
                ]
            ),
            "epochs": (
                shared[
                    "epochs"
                ]
            ),
            "batch_size": (
                shared[
                    "batch_size"
                ]
            ),
            "factor_l2": 1e-6,
            "bias_l2": 1e-6,
            "max_grad_norm": 5.0,
            "seed": seed,
            "device": (
                shared[
                    "device"
                ]
            ),
        }

        model = GaussianMF(
            **hyperparameters
        )

        return (
            model,
            hyperparameters,
        )

    # --------------------------------------------------------
    # LogisticPCA
    # --------------------------------------------------------

    if model_key == "logistic_pca":

        hyperparameters = {
            "n_factors": 32,
            "learning_rate": (
                shared[
                    "learning_rate"
                ]
            ),
            "epochs": (
                shared[
                    "epochs"
                ]
            ),
            "user_batch_size": 32,
            "max_grad_norm": 5.0,
            "intercept_epsilon": 1e-6,
            "seed": seed,
            "device": (
                shared[
                    "device"
                ]
            ),
        }

        model = LogisticPCA(
            heldout_interactions=(
                _heldout_interactions(
                    split
                )
            ),
            **hyperparameters,
        )

        return (
            model,
            hyperparameters,
        )

    # --------------------------------------------------------
    # HPF
    # --------------------------------------------------------

    if model_key == "hpf":

        hyperparameters = {
            "n_factors": 32,
            "max_iterations": 50,
            "tolerance": 1e-4,
            "a0": 0.3,
            "a": 0.3,
            "b0": 1.0,
            "c0": 0.3,
            "c": 0.3,
            "d0": 1.0,
            "initialization_scale": 0.1,
            "seed": seed,
        }

        model = HPF(
            heldout_interactions=(
                _heldout_interactions(
                    split
                )
            ),
            **hyperparameters,
        )

        return (
            model,
            hyperparameters,
        )

    raise AssertionError(
        "unreachable model construction path"
    )


# ============================================================
# Experiment metadata
# ============================================================


def build_experiment_spec(
    model_key: str,
    *,
    seed: int,
    hyperparameters: dict[
        str,
        Any,
    ],
) -> ExperimentSpec:
    """Build one explicit Step-43 ExperimentSpec."""

    if model_key not in MODEL_METADATA:
        raise ValueError(
            f"Unknown model key: {model_key}"
        )

    metadata = (
        MODEL_METADATA[
            model_key
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
# Run-context handling
# ============================================================


def prepare_run_context(
    *,
    config_path: str | Path,
    config: dict[
        str,
        Any,
    ],
    version: str,
    results_root: str | Path,
    run_id: str | None,
) -> RunContext:
    """Create a new run or safely reopen a matching existing run."""

    paths = artifact_paths(
        config,
        version=version,
    )

    provenance = (
        build_run_provenance(
            config_path=(
                _resolve_project_path(
                    config_path
                )
            ),
            preparation_manifest_path=(
                paths[
                    "preparation_manifest"
                ]
            ),
            split_manifest_path=(
                paths[
                    "split_manifest"
                ]
            ),
            project_root=PROJECT_ROOT,
            verify_files=True,
        )
    )

    results_root_path = (
        _resolve_project_path(
            results_root
        )
    )

    if run_id is None:

        return create_run(
            provenance=provenance,
            output_root=(
                results_root_path
            ),
            project_root=PROJECT_ROOT,
        )

    candidate = (
        results_root_path
        /
        run_id
    )

    if candidate.exists():

        existing = (
            load_run_context(
                candidate
            )
        )

        if (
            existing.provenance
            !=
            provenance
        ):
            raise ValueError(
                "Existing run provenance does not "
                "match the current config/data/"
                "split/code state"
            )

        return existing

    return create_run(
        provenance=provenance,
        output_root=(
            results_root_path
        ),
        run_id=run_id,
        project_root=PROJECT_ROOT,
    )


# ============================================================
# Runner
# ============================================================


def resolve_model_keys(
    requested: str,
) -> list[
    str,
]:
    """Expand one CLI model selection."""

    if requested == "all":
        return list(
            MODEL_ORDER
        )

    if requested not in MODEL_ORDER:
        raise ValueError(
            f"Unknown model selection: {requested}"
        )

    return [
        requested
    ]


def resolve_seed(
    config: dict[
        str,
        Any,
    ],
    requested_seed: int | None,
) -> int:
    """Resolve the explicit experiment seed."""

    reproducibility = config.get(
        "reproducibility"
    )

    if not isinstance(
        reproducibility,
        dict,
    ):
        raise ValueError(
            "config.reproducibility "
            "must be a mapping"
        )

    if requested_seed is None:

        seed = reproducibility.get(
            "global_seed"
        )

    else:
        seed = requested_seed

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
            "Experiment seed must be an integer"
        )

    return seed


def run_selected_models(
    *,
    config_path: str | Path,
    version: str,
    model_selection: str,
    seed: int | None = None,
    run_id: str | None = None,
    results_root: str | Path | None = None,
    target_split: str = "validation",
    allow_test: bool = False,
) -> tuple[
    RunContext,
    list[
        ExperimentResult
    ],
]:
    """Execute the selected MF/literature model experiments."""

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
    # Hard test gate BEFORE loading/fitting any models.
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

    model_keys = resolve_model_keys(
        model_selection
    )

    results: list[
        ExperimentResult
    ] = []

    for model_key in model_keys:

        print()
        print(
            "="
            *
            72
        )

        print(
            "Running:",
            MODEL_METADATA[
                model_key
            ][
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
            build_model(
                model_key,
                split=split,
                config=config,
                seed=resolved_seed,
            )
        )

        spec = (
            build_experiment_spec(
                model_key,
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
            "Run Paper 2 MF/literature "
            "experiments on a frozen split."
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
        choices=MODEL_CHOICES,
        help=(
            "Model to run, or 'all' for "
            "the complete MF/literature group."
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
            "Optional result root override. "
            "Defaults to config.paths.results_root."
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
        run_selected_models(
            config_path=args.config,
            version=args.version,
            model_selection=args.model,
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
        "MF RUNNER COMPLETE"
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