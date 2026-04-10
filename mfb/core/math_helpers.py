"""Small math helper utilities for the MFB educational project.

These helpers keep tiny numeric conversions in one place so the other modules
stay readable.
"""

from __future__ import annotations

import numpy as np


def as_float_array(values) -> np.ndarray:
    """Return *values* as a one-dimensional float NumPy array."""
    return np.atleast_1d(np.asarray(values, dtype=float))


def hz_to_rad_s(freq_hz):
    """Convert frequency in Hz to angular frequency in rad/s."""
    return 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
