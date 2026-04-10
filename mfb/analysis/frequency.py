from __future__ import annotations

import numpy as np

from mfb.core.transfer import Transfer, freq_response, magnitude_db


def normalized_magnitude_db(system: Transfer, freq_hz: np.ndarray, ref_hz: float = 200.0) -> np.ndarray:
    mag = magnitude_db(system, freq_hz)
    ref = float(np.interp(np.log10(ref_hz), np.log10(freq_hz), mag))
    return mag - ref


def spl_db(system_pressure: Transfer, freq_hz: np.ndarray, p_ref: float = 20e-6) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(freq_response(system_pressure, freq_hz)) / p_ref, 1e-20))
