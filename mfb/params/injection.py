from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionParams:
    gain: float = 1.0


@dataclass(frozen=True)
class AmplifierParams:
    gain_db: float = 26.3

    @property
    def gain_vv(self) -> float:
        return 10 ** (self.gain_db / 20.0)
