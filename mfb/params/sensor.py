from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorParams:
    sensitivity_mv_per_g: float = 10.0
    f_res_hz: float = 28e3
    f_output_amp_hz: float = 70e3
    zeta: float = 0.35

    @property
    def dc_v_per_mps2(self) -> float:
        return (self.sensitivity_mv_per_g * 1e-3) / 9.81
