from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoxParams:
    volume_l: float | None = None


@dataclass(frozen=True)
class AcousticParams:
    rho: float = 1.2
    c: float = 343.0
    distance_m: float = 1.0
    p_ref: float = 20e-6
