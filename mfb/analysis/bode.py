"""Bode helpers for analysis and tests."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from mfb.core.transfer import Transfer, magnitude_db, phase_deg


def bode_data(system: Transfer, freq_hz: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return magnitude (dB) and phase (deg) arrays."""
    return magnitude_db(system, freq_hz), phase_deg(system, freq_hz)
