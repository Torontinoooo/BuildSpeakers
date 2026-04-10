from __future__ import annotations

from dataclasses import dataclass

from .base import NonlinearCurve


@dataclass
class CmsCurve(NonlinearCurve):
    k: float = 0.0

    def value(self, x: float) -> float:
        if not self.enabled:
            return x
        return x / (1.0 + self.k * x * x)
