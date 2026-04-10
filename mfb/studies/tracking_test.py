"""Minimal tracking study for closed-loop acceleration command following."""

from __future__ import annotations

import numpy as np
from scipy import signal

from mfb.blocks.sensor import sensor_dc_v_per_mps2
from mfb.params.sensor_params import SensorParams
from mfb.systems.mfb_loop import build_mfb_loop


def run(freq_hz: float = 40.0, duration_s: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a sinusoidal acceleration reference and return (t, ref, out)."""
    system = build_mfb_loop(3.0)
    t = np.linspace(0.0, duration_s, 3000)
    ref = np.sin(2.0 * np.pi * freq_hz * t)
    r_v = sensor_dc_v_per_mps2(SensorParams()) * ref
    _, out, _ = signal.lsim(system.t.tf, U=r_v, T=t)
    return t, ref, out
