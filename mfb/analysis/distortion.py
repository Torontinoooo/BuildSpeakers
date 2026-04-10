from __future__ import annotations

import numpy as np


def thd_estimate(samples: np.ndarray) -> float:
    """Small placeholder THD estimate using FFT bins."""
    spec = np.fft.rfft(samples)
    mags = np.abs(spec)
    if len(mags) < 3 or mags[1] == 0:
        return 0.0
    fundamental = mags[1]
    harmonics = np.sqrt(np.sum(mags[2:6] ** 2))
    return float(harmonics / fundamental)
