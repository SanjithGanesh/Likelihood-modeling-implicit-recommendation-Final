"""Reproducibility utilities for the Paper 2 experiment pipeline.

This module centralizes random seeding and deterministic execution settings.

All experiment entry points should call ``set_seed(seed)`` before constructing
models, samplers, or other stochastic objects.

Independent random-number-generator streams should be created with
``make_python_rng``, ``make_numpy_rng``, or ``make_torch_generator`` rather
than repeatedly mutating the global random state.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

import numpy as np
import torch


MAX_SEED = 2**32 - 1


def _validate_seed(seed: int) -> int:
    """Validate and return a seed suitable for project-wide use."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if seed < 0 or seed > MAX_SEED:
        raise ValueError(
            f"seed must be between 0 and {MAX_SEED}, inclusive"
        )

    return seed


def set_seed(
    seed: int,
    deterministic: bool = True,
    warn_only: bool = False,
) -> None:
    """Seed Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Master random seed.

    deterministic:
        If True, request deterministic PyTorch behavior wherever supported.

    warn_only:
        Passed to ``torch.use_deterministic_algorithms``. If False,
        nondeterministic operations raise an error. If True, PyTorch warns
        instead.
    """
    seed = _validate_seed(seed)

    # Useful for child processes launched after this call.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Required by some deterministic CUDA operations if GPU execution is used
    # in the future.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(
            True,
            warn_only=warn_only,
        )

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False


def derive_seed(base_seed: int, namespace: str) -> int:
    """Derive a deterministic independent seed from a base seed.

    The namespace should describe the component using the random stream,
    for example ``"split"``, ``"negative_sampling"``, or
    ``"model_initialization"``.
    """
    base_seed = _validate_seed(base_seed)

    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")

    if not namespace.strip():
        raise ValueError("namespace must not be empty")

    payload = f"{base_seed}:{namespace}".encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    derived = int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )

    return derived


def make_python_rng(
    base_seed: int,
    namespace: str,
) -> random.Random:
    """Create an independent deterministic Python RNG."""
    seed = derive_seed(base_seed, namespace)
    return random.Random(seed)


def make_numpy_rng(
    base_seed: int,
    namespace: str,
) -> np.random.Generator:
    """Create an independent deterministic NumPy RNG."""
    seed = derive_seed(base_seed, namespace)
    return np.random.default_rng(seed)


def make_torch_generator(
    base_seed: int,
    namespace: str,
    device: Optional[str] = None,
) -> torch.Generator:
    """Create an independent deterministic PyTorch generator."""
    seed = derive_seed(base_seed, namespace)

    if device is None:
        generator = torch.Generator()
    else:
        generator = torch.Generator(device=device)

    generator.manual_seed(seed)

    return generator