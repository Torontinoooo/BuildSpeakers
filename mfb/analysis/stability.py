"""Stability utilities for closed-loop checks."""

from __future__ import annotations

import numpy as np

from mfb.core.transfer import Transfer


def poles(system: Transfer) -> np.ndarray:
    """Return closed-loop poles."""
    return np.roots(system.den)


def stable(system: Transfer) -> bool:
    """Return ``True`` when all poles have negative real part."""
    return bool(np.max(np.real(poles(system))) < 0.0)
