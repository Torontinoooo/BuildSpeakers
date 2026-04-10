from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardLimiter:
    threshold: float = 1.0

    def apply(self, x: float) -> float:
        return max(min(x, self.threshold), -self.threshold)
