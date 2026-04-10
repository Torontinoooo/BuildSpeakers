from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerParams:
    re: float = 6.08
    le: float = 0.90e-3
    fs: float = 31.0
    qms: float = 2.15
    mms: float = 14.51e-3
    cms: float = 1.82e-3
    sd: float = 104e-4
    bl: float = 6.9
    vas_l: float = 27.5

    @property
    def rms(self) -> float:
        from math import pi

        return 2 * pi * self.fs * self.mms / self.qms
