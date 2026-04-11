from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class LimiterParams:
    enabled: bool = False
    peak_v: float = 2.0


class SignalLimiter:
    def __init__(self, params: LimiterParams) -> None:
        self.params = params

    def process(self, u: np.ndarray) -> np.ndarray:
        if not self.params.enabled:
            return u
        return np.clip(u, -self.params.peak_v, self.params.peak_v)
