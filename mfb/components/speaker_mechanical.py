from __future__ import annotations

from dataclasses import dataclass

from mfb.params.speaker import SpeakerParams


@dataclass
class SpeakerMechanical:
    params: SpeakerParams

    @property
    def mms(self) -> float:
        return self.params.mms

    @property
    def rms(self) -> float:
        return self.params.rms

    @property
    def cms(self) -> float:
        return self.params.cms
