"""Accelerometer model parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorParams:
    """Simple MEMS acceleration sensor model data."""

    sensitivity_mv_per_g: float = 10.0
    f_res_hz: float = 28_000.0
    f_output_amp_hz: float = 70_000.0
    zeta: float = 0.35
