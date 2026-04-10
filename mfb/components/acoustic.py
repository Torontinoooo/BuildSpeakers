from __future__ import annotations

from dataclasses import dataclass
from math import pi

from mfb.core.transfer import Transfer, gain
from mfb.params.box import AcousticParams
from mfb.params.speaker import SpeakerParams


@dataclass
class MonopoleAcoustic:
    speaker: SpeakerParams
    acoustic: AcousticParams

    def transfer(self) -> Transfer:
        k = self.acoustic.rho * self.speaker.sd / (4 * pi * self.acoustic.distance_m)
        return gain(k)
