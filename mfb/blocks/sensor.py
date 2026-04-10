"""Accelerometer transfer-function block."""

import numpy as np

from mfb.core.transfer import Transfer, make_tf, series
from mfb.params.sensor_params import SensorParams


def sensor_dc_v_per_mps2(params: SensorParams) -> float:
    """Convert mV/g to V/(m/s^2)."""
    return (params.sensitivity_mv_per_g * 1e-3) / 9.81


def sensor_tf(params: SensorParams) -> Transfer:
    """2nd-order sensor resonance plus first-order output amplifier roll-off."""
    s0 = sensor_dc_v_per_mps2(params)
    wn = 2.0 * np.pi * params.f_res_hz
    wa = 2.0 * np.pi * params.f_output_amp_hz
    mech = make_tf([s0 * wn**2], [1.0, 2.0 * params.zeta * wn, wn**2])
    out = make_tf([wa], [1.0, wa])
    return series(mech, out)
