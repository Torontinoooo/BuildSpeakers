from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NonlinearCurve:
    enabled: bool = False

    def value(self, x: float) -> float:
        return x
