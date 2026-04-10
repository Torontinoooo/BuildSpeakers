"""Amplifier transfer-function block."""

import numpy as np

from mfb.core.transfer import Transfer, gain
from mfb.params.amp_params import AmpParams


def amplifier_tf(params: AmpParams) -> Transfer:
    """Return a constant-gain amplifier block."""
    linear_gain = 10.0 ** (params.gain_db / 20.0)
    return gain(linear_gain)
