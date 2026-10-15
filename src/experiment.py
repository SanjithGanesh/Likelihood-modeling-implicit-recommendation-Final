"""Unified experiment orchestration for Paper 2.

This module provides the common experiment layer used by every Paper 2
model after model construction.

Responsibilities
----------------
1. validate experiment metadata;
2. enforce validation/test information boundaries;
3. fit a model on the canonical training partition;
4. run the shared full-catalog ranking evaluator;
5. collect model-native diagnostics when available;
6. collect training diagnostics;
7. return one structured ExperimentResult.

This module deliberately does NOT:
- construct models;
- perform hyperparameter search;
- save result files;
- save model checkpoints;
- aggregate multiple runs;
- generate figures.

Those responsibilities belong to later pipeline stages.

The default evaluation target is validation and test access is denied
unless explicitly enabled for a final-evaluation workflow.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from src.evaluation import (
    EvaluationResult,
    evaluate_from_config,
)
from src.split import SplitResult


TargetSplit = Literal[
    "validation",
    "test",
]

ExperimentStatus = Literal[
    "completed",
    "failed",
]


# ============================================================
# Experiment specification
# ============================================================


@dataclass(
    frozen=True,
)
class ExperimentSpec:
    """Immutable identifying metadata for one experiment run."""

    model_name: str

    family: str

    seed: int

    variant: str = "primary"

    hyperparameters: dict[
        str,
        Any,
    ] = field(
        default_factory=dict,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and defensively copy experiment metadata."""

        _validate_nonempty_string(
            self.model_name,
            name="model_name",
        )

        _validate_nonempty_string(
            self.family,
            name="family",
        )

        _validate_nonempty_string(
            self.variant,
            name="variant",
        )

        if (
            not isinstance(
                self.seed,
                int,
            )
            or isinstance(
                self.seed,
                bool,
            )
        ):
            raise TypeError(
                "seed must be an integer"
            )

        if not isinstance(
            self.hyperparameters,
            Mapping,
        ):
            raise TypeError(
                "hyperparameters must be a mapping"
            )

        copied_hyperparameters = (
            deepcopy(
                dict(
                    self.hyperparameters
                )
            )
        )

        object.__setattr__(
            self,
            "hyperparameters",
            copied_hyperparameters,
        )


# ============================================================
# Experiment result
# ============================================================


@dataclass(
    frozen=True,
)
class ExperimentResult:
    """Structured output of one completed experiment run."""

    model_name: str

    family: str

    variant: str

    seed: int

    target_split: TargetSplit

    ranking_metrics: dict[
        str,
        float,
    ]

    native_metrics: dict[
        str,
        float,
    ]

    training_history: list[
        dict[
            str,
            Any,
        ]
    ]

    runtime_seconds: float

    status: ExperimentStatus

    error_message: str | None

    metadata: dict[
        str,
        Any,
    ]

    def __post_init__(
        self,
    ) -> None:
        """Validate result-level invariants."""

        _validate_nonempty_string(
            self.model_name,
            name="model_name",
        )

        _validate_nonempty_string(
            self.family,
            name="family",
        )

        _validate_nonempty_string(
            self.variant,
            name="variant",
        )

        if (
            not isinstance(
                self.seed,
                int,
            )
            or isinstance(
                self.seed,
                bool,
            )
        ):
            raise TypeError(
                "seed must be an integer"
            )

        _validate_target_split(
            self.target_split
        )

        if self.status not in {
            "completed",
            "failed",
        }:
            raise ValueError(
                "status must be 'completed' or 'failed'"
            )

        if (
            not np.isfinite(
                self.runtime_seconds
            )
            or self.runtime_seconds < 0.0
        ):
            raise ValueError(
                "runtime_seconds must be finite and nonnegative"
            )

        _validate_metric_mapping(
            self.ranking_metrics,
            name="ranking_metrics",
        )

        _validate_metric_mapping(
            self.native_metrics,
            name="native_metrics",
        )

        if not isinstance(
            self.training_history,
            list,
        ):
            raise TypeError(
                "training_history must be a list"
            )

        for index, record in enumerate(
            self.training_history
        ):
            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    "training_history records must be dictionaries; "
                    f"record {index} is invalid"
                )

        if (
            self.status
            ==
            "completed"
            and self.error_message
            is not None
        ):
            raise ValueError(
                "completed experiments must have error_message=None"
            )

        if (
            self.status
            ==
            "failed"
            and (
                self.error_message
                is None
                or not str(
                    self.error_message
                ).strip()
            )
        ):
            raise ValueError(
                "failed experiments must include an error_message"
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary"
            )

        object.__setattr__(
            self,
            "ranking_metrics",
            deepcopy(
                self.ranking_metrics
            ),
        )

        object.__setattr__(
            self,
            "native_metrics",
            deepcopy(
                self.native_metrics
            ),
        )

        object.__setattr__(
            self,
            "training_history",
            deepcopy(
                self.training_history
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            deepcopy(
                self.metadata
            ),
        )

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        """Return a defensive dictionary representation.

        Actual JSON/CSV persistence is intentionally deferred to the
        result-storage layer.
        """

        return {
            "model_name": self.model_name,
            "family": self.family,
            "variant": self.variant,
            "seed": self.seed,
            "target_split": (
                self.target_split
            ),
            "ranking_metrics": deepcopy(
                self.ranking_metrics
            ),
            "native_metrics": deepcopy(
                self.native_metrics
            ),
            "training_history": deepcopy(
                self.training_history
            ),
            "runtime_seconds": float(
                self.runtime_seconds
            ),
            "status": self.status,
            "error_message": (
                self.error_message
            ),
            "metadata": deepcopy(
                self.metadata
            ),
        }


# ============================================================
# Validation helpers
# ============================================================


def _validate_nonempty_string(
    value: Any,
    *,
    name: str,
) -> None:
    """Require a nonempty string."""

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string"
        )

    if not value.strip():
        raise ValueError(
            f"{name} must not be empty"
        )


def _validate_target_split(
    target_split: str,
) -> None:
    """Validate supported experiment evaluation partitions."""

    if target_split not in {
        "validation",
        "test",
    }:
        raise ValueError(
            "target_split must be "
            "'validation' or 'test'"
        )


def _validate_metric_mapping(
    metrics: Mapping[
        str,
        Any,
    ],
    *,
    name: str,
) -> None:
    """Require finite numeric metric values."""

    if not isinstance(
        metrics,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping"
        )

    for key, value in metrics.items():

        if not isinstance(
            key,
            str,
        ):
            raise TypeError(
                f"{name} keys must be strings"
            )

        if isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"{name}[{key!r}] must be numeric"
            )

        try:
            numeric_value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                f"{name}[{key!r}] must be numeric"
            ) from exc

        if not np.isfinite(
            numeric_value
        ):
            raise ValueError(
                f"{name}[{key!r}] must be finite"
            )


def _validate_model_contract(
    model: Any,
) -> None:
    """Validate the portion of the common model contract used here."""

    for method_name in [
        "fit",
        "score",
    ]:

        method = getattr(
            model,
            method_name,
            None,
        )

        if not callable(
            method
        ):
            raise TypeError(
                "model must implement callable "
                f"{method_name}()"
            )


def _validate_model_identity(
    model: Any,
    spec: ExperimentSpec,
) -> None:
    """Prevent accidentally mislabeled model-result rows."""

    model_name = getattr(
        model,
        "model_name",
        None,
    )

    if model_name is None:
        raise AttributeError(
            "model must expose model_name"
        )

    if str(
        model_name
    ) != spec.model_name:
        raise ValueError(
            "ExperimentSpec model_name does not match "
            "the model implementation: "
            f"spec={spec.model_name!r}, "
            f"model={model_name!r}"
        )

    model_seed = getattr(
        model,
        "seed",
        None,
    )

    if model_seed is not None:

        if (
            isinstance(
                model_seed,
                bool,
            )
            or not isinstance(
                model_seed,
                (
                    int,
                    np.integer,
                ),
            )
        ):
            raise TypeError(
                "model.seed must be an integer when exposed"
            )

        if int(
            model_seed
        ) != spec.seed:
            raise ValueError(
                "ExperimentSpec seed does not match "
                "the model seed: "
                f"spec={spec.seed}, "
                f"model={int(model_seed)}"
            )


def _validate_split(
    split: SplitResult,
) -> None:
    """Validate the experiment split container at the orchestration edge."""

    if not isinstance(
        split,
        SplitResult,
    ):
        raise TypeError(
            "split must be a SplitResult"
        )

    for name in [
        "train",
        "validation",
        "test",
    ]:

        frame = getattr(
            split,
            name,
        )

        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"split.{name} must be a pandas DataFrame"
            )

        if len(
            frame
        ) == 0:
            raise ValueError(
                f"split.{name} must not be empty"
            )


# ============================================================
# Extraction helpers
# ============================================================


def _float_metric_mapping(
    metrics: Mapping[
        str,
        Any,
    ],
    *,
    name: str,
) -> dict[
    str,
    float,
]:
    """Convert a metric mapping to plain finite Python floats."""

    _validate_metric_mapping(
        metrics,
        name=name,
    )

    return {
        str(
            key
        ): float(
            value
        )
        for key, value
        in metrics.items()
    }


def _collect_training_history(
    model: Any,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Collect model training diagnostics if exposed."""

    if not hasattr(
        model,
        "training_history",
    ):
        return []

    history = getattr(
        model,
        "training_history"
    )

    if history is None:
        return []

    if not isinstance(
        history,
        list,
    ):
        raise TypeError(
            "model.training_history must be a list"
        )

    copied: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, record in enumerate(
        history
    ):

        if not isinstance(
            record,
            Mapping,
        ):
            raise TypeError(
                "training_history records must be mappings; "
                f"record {index} is invalid"
            )

        copied.append(
            deepcopy(
                dict(
                    record
                )
            )
        )

    return copied


def _collect_native_metrics(
    model: Any,
    evaluation_frame: pd.DataFrame,
) -> dict[
    str,
    float,
]:
    """Collect optional family-native diagnostics.

    Models without a native_metrics() method simply return an empty
    mapping. Gaussian-family models currently use this hook for RMSE/MAE.
    """

    native_method = getattr(
        model,
        "native_metrics",
        None,
    )

    if native_method is None:
        return {}

    if not callable(
        native_method
    ):
        raise TypeError(
            "model.native_metrics must be callable when exposed"
        )

    metrics = native_method(
        evaluation_frame
    )

    if not isinstance(
        metrics,
        Mapping,
    ):
        raise TypeError(
            "model.native_metrics() must return a mapping"
        )

    return _float_metric_mapping(
        metrics,
        name="native_metrics",
    )


def _evaluation_frame(
    split: SplitResult,
    target_split: TargetSplit,
) -> pd.DataFrame:
    """Return the frame corresponding to the requested target."""

    if target_split == "validation":
        return split.validation

    if target_split == "test":
        return split.test

    raise AssertionError(
        "unreachable target_split"
    )


def _experiment_metadata(
    *,
    model: Any,
    split: SplitResult,
    evaluation: EvaluationResult,
    spec: ExperimentSpec,
) -> dict[
    str,
    Any,
]:
    """Construct structured non-metric run metadata."""

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
            split.seed
        ),
        "split_attempt": int(
            split.attempt
        ),
        "enforce_train_item_coverage": bool(
            split.enforce_train_item_coverage
        ),
        "train_rows": int(
            len(
                split.train
            )
        ),
        "validation_rows": int(
            len(
                split.validation
            )
        ),
        "test_rows": int(
            len(
                split.test
            )
        ),
    }


# ============================================================
# Main experiment entry point
# ============================================================


def run_experiment(
    model: Any,
    split: SplitResult,
    config: dict[
        str,
        Any,
    ],
    spec: ExperimentSpec,
    target_split: TargetSplit = "validation",
    allow_test: bool = False,
) -> ExperimentResult:
    """Run one Paper 2 experiment through the common evaluation layer.

    Parameters
    ----------
    model:
        A constructed Paper 2 model implementing fit() and score().

    split:
        The canonical frozen train/validation/test SplitResult.

    config:
        Experiment configuration consumed by the shared evaluator.

    spec:
        Structured identifying metadata for this run.

    target_split:
        Evaluation target. Defaults to validation.

    allow_test:
        Explicit final-evaluation gate. Test evaluation is rejected unless
        both target_split="test" and allow_test=True are deliberately used.

    Returns
    -------
    ExperimentResult
        Structured result for one successfully completed run.

    Notes
    -----
    This function intentionally does not catch model/evaluation exceptions.
    Invalid runs must fail loudly rather than silently entering scientific
    result tables.
    """

    _validate_target_split(
        target_split
    )

    if not isinstance(
        allow_test,
        bool,
    ):
        raise TypeError(
            "allow_test must be a boolean"
        )

    # --------------------------------------------------------
    # Test-access gate.
    #
    # This occurs BEFORE model fitting so a tuning-time test
    # request cannot perform any experiment work at all.
    # --------------------------------------------------------

    if (
        target_split
        ==
        "test"
        and not allow_test
    ):
        raise PermissionError(
            "Test evaluation is disabled. "
            "Use target_split='test' with allow_test=True "
            "only for an explicitly authorized final evaluation."
        )

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "config must be a dictionary"
        )

    if not isinstance(
        spec,
        ExperimentSpec,
    ):
        raise TypeError(
            "spec must be an ExperimentSpec"
        )

    _validate_split(
        split
    )

    _validate_model_contract(
        model
    )

    _validate_model_identity(
        model,
        spec,
    )

    start_time = perf_counter()

    # --------------------------------------------------------
    # Fit strictly on the canonical training partition.
    # --------------------------------------------------------

    model.fit(
        split.train
    )

    # --------------------------------------------------------
    # Common full-catalog ranking evaluation.
    # --------------------------------------------------------

    evaluation = evaluate_from_config(
        split=split,
        config=config,
        score_function=model.score,
        target_split=target_split,
        allow_test=allow_test,
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
        target_split
    ):
        raise RuntimeError(
            "Evaluator returned a different target_split "
            "than requested"
        )

    ranking_metrics = (
        _float_metric_mapping(
            evaluation.aggregate,
            name="ranking_metrics",
        )
    )

    # --------------------------------------------------------
    # Optional family-native diagnostics.
    #
    # Validation remains the default. The test DataFrame can
    # reach this path only after the explicit gate above.
    # --------------------------------------------------------

    native_metrics = (
        _collect_native_metrics(
            model,
            _evaluation_frame(
                split,
                target_split,
            ),
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
        _experiment_metadata(
            model=model,
            split=split,
            evaluation=evaluation,
            spec=spec,
        )
    )

    return ExperimentResult(
        model_name=spec.model_name,
        family=spec.family,
        variant=spec.variant,
        seed=spec.seed,
        target_split=target_split,
        ranking_metrics=ranking_metrics,
        native_metrics=native_metrics,
        training_history=training_history,
        runtime_seconds=float(
            runtime_seconds
        ),
        status="completed",
        error_message=None,
        metadata=metadata,
    )