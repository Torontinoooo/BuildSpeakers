from __future__ import annotations

import numpy as np
from scipy import signal

from mfb.core.transfer import Transfer


def simulate(system: Transfer, t: np.ndarray, u: np.ndarray) -> np.ndarray:
    _, y, _ = signal.lsim(system.tf, U=u, T=t)
    return y
