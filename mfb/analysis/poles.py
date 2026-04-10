"""Pole formatting helpers used by studies."""

from __future__ import annotations

import numpy as np


def max_real_part(poles_array: np.ndarray) -> float:
    """Return the largest real pole part as a scalar float."""
    return float(np.max(np.real(poles_array)))
