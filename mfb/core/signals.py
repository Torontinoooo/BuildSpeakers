from __future__ import annotations

import numpy as np


def sine(freq_hz: float, duration_s: float, fs_hz: float, amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration_s, 1.0 / fs_hz)
    return t, amplitude * np.sin(2 * np.pi * freq_hz * t)
