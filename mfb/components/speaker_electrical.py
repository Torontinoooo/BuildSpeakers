from __future__ import annotations

from dataclasses import dataclass

from mfb.params.speaker import SpeakerParams


@dataclass
class SpeakerElectrical:
    params: SpeakerParams

    @property
    def re(self) -> float:
        return self.params.re

    @property
    def le(self) -> float:
        return self.params.le
