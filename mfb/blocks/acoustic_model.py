"""Low-frequency acoustic radiation model block."""

import numpy as np

from mfb.core.transfer import Transfer, gain
from mfb.params.speaker_params import SpeakerParams


def acoustic_pressure_tf(spk: SpeakerParams, distance_m: float = 1.0) -> Transfer:
    """Approximate pressure/acceleration gain for low-frequency piston behavior."""
    rho = 1.20
    k = rho * spk.sd_m2 / (4.0 * np.pi * distance_m)
    return gain(k)
