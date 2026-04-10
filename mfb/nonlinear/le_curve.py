from __future__ import annotations

from dataclasses import dataclass

from .base import NonlinearCurve


@dataclass
class LeCurve(NonlinearCurve):
    alpha: float = 0.0

    def value(self, x: float) -> float:
        if not self.enabled:
            return x
        return x * (1.0 + self.alpha * abs(x))
