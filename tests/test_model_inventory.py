"""Permanent model inventory and interface tests for Paper 2."""

from __future__ import annotations

import importlib
import inspect

import pytest


MODEL_MODULES = {
    "src.models.baselines": 2,
    "src.models.pca_svd": 1,
    "src.models.bernoulli_mf": 1,
    "src.models.poisson_mf": 1,
    "src.models.gaussian_mf": 1,
    "src.models.logistic_pca": 1,
    "src.models.hpf": 1,
    "src.models.b_emb": 1,
    "src.models.p_emb": 1,
    "src.models.ap_emb": 1,
    "src.models.g_emb": 1,
    "src.models.ng_emb": 1,
}


EXPECTED_TOTAL_MODELS = 13


def _model_classes_from_module(
    module_name: str,
) -> list[type]:
    """Return public model classes defined by a model module."""

    module = importlib.import_module(
        module_name
    )

    discovered: list[type] = []

    for _, obj in inspect.getmembers(
        module,
        inspect.isclass,
    ):
        # Ignore classes merely imported into the module.
        if obj.__module__ != module_name:
            continue

        # Private implementation modules such as _PEmbModule are
        # intentionally not part of the public model inventory.
        if obj.__name__.startswith("_"):
            continue

        # Every primary recommender model exposes the frozen
        # fit / score / save / load contract.
        required_methods = (
            "fit",
            "score",
            "save",
            "load",
        )

        if all(
            callable(
                getattr(
                    obj,
                    method,
                    None,
                )
            )
            for method in required_methods
        ):
            discovered.append(
                obj
            )

    return discovered


@pytest.mark.parametrize(
    (
        "module_name",
        "expected_count",
    ),
    MODEL_MODULES.items(),
)
def test_model_module_imports_and_public_model_count(
    module_name: str,
    expected_count: int,
) -> None:
    """Every frozen model module imports and exposes expected models."""

    classes = _model_classes_from_module(
        module_name
    )

    assert len(
        classes
    ) == expected_count, (
        f"{module_name} exposed "
        f"{len(classes)} public model classes; "
        f"expected {expected_count}. "
        f"Found: {[cls.__name__ for cls in classes]}"
    )


def test_total_primary_model_inventory_is_thirteen() -> None:
    """The corrected Paper 2 inventory contains 13 models."""

    all_classes: list[
        tuple[
            str,
            type,
        ]
    ] = []

    for module_name in MODEL_MODULES:
        for model_class in (
            _model_classes_from_module(
                module_name
            )
        ):
            all_classes.append(
                (
                    module_name,
                    model_class,
                )
            )

    assert len(
        all_classes
    ) == EXPECTED_TOTAL_MODELS

    identities = {
        (
            module_name,
            model_class.__name__,
        )
        for (
            module_name,
            model_class
        )
        in all_classes
    }

    assert len(
        identities
    ) == EXPECTED_TOTAL_MODELS


def test_every_primary_model_exposes_frozen_interface() -> None:
    """All 13 models preserve the common model-output contract."""

    for module_name in MODEL_MODULES:

        classes = (
            _model_classes_from_module(
                module_name
            )
        )

        for model_class in classes:

            for method in (
                "fit",
                "score",
                "save",
                "load",
            ):
                attribute = getattr(
                    model_class,
                    method,
                    None,
                )

                assert callable(
                    attribute
                ), (
                    f"{module_name}."
                    f"{model_class.__name__} "
                    f"is missing callable {method}()"
                )


def test_primary_model_classes_are_not_aliases() -> None:
    """No two inventory entries resolve to the same class object."""

    discovered_classes: list[
        type
    ] = []

    for module_name in MODEL_MODULES:
        discovered_classes.extend(
            _model_classes_from_module(
                module_name
            )
        )

    object_ids = {
        id(model_class)
        for model_class
        in discovered_classes
    }

    assert len(
        object_ids
    ) == EXPECTED_TOTAL_MODELS


def test_efemb_modules_have_distinct_public_classes() -> None:
    """The five EF-EMB variants must remain separate implementations."""

    ef_modules = [
        "src.models.b_emb",
        "src.models.p_emb",
        "src.models.ap_emb",
        "src.models.g_emb",
        "src.models.ng_emb",
    ]

    classes = []

    for module_name in ef_modules:
        module_classes = (
            _model_classes_from_module(
                module_name
            )
        )

        assert len(
            module_classes
        ) == 1

        classes.append(
            module_classes[
                0
            ]
        )

    assert len(
        {
            id(model_class)
            for model_class
            in classes
        }
    ) == 5


def test_key_semantically_distinct_models_are_not_aliases() -> None:
    """Historically confused model pairs must remain distinct."""

    bernoulli_mf = (
        _model_classes_from_module(
            "src.models.bernoulli_mf"
        )[0]
    )

    logistic_pca = (
        _model_classes_from_module(
            "src.models.logistic_pca"
        )[0]
    )

    poisson_mf = (
        _model_classes_from_module(
            "src.models.poisson_mf"
        )[0]
    )

    hpf = (
        _model_classes_from_module(
            "src.models.hpf"
        )[0]
    )

    p_emb = (
        _model_classes_from_module(
            "src.models.p_emb"
        )[0]
    )

    ap_emb = (
        _model_classes_from_module(
            "src.models.ap_emb"
        )[0]
    )

    g_emb = (
        _model_classes_from_module(
            "src.models.g_emb"
        )[0]
    )

    ng_emb = (
        _model_classes_from_module(
            "src.models.ng_emb"
        )[0]
    )

    assert bernoulli_mf is not logistic_pca
    assert poisson_mf is not hpf
    assert p_emb is not ap_emb
    assert g_emb is not ng_emb