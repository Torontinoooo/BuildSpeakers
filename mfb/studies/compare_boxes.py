"""Quick study comparing open-loop acoustic response for two box sizes."""

from __future__ import annotations

import numpy as np

from mfb.core.transfer import magnitude_db
from mfb.systems.mfb_loop import build_mfb_loop


def run() -> dict[str, np.ndarray]:
    """Return simple comparison curves for free-air and small box."""
    freq_hz = np.logspace(0, 4, 400)
    free = build_mfb_loop(None)
    small = build_mfb_loop(3.0)
    return {
        "freq_hz": freq_hz,
        "free_open_db": magnitude_db(free.h_open_acoustic, freq_hz),
        "small_open_db": magnitude_db(small.h_open_acoustic, freq_hz),
    }
