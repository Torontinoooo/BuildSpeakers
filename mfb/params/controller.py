from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PIDParams:
    kp: float = 1.4
    ki: float = 0.05
    kd: float = 0.0001
    f_d_hz: float = 700.0
