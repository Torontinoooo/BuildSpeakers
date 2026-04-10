from __future__ import annotations

from dataclasses import dataclass

from mfb.params.box import BoxParams
from mfb.params.speaker import SpeakerParams


@dataclass
class SealedBox:
    params: BoxParams

    def total_compliance(self, speaker: SpeakerParams) -> float:
        if self.params.volume_l is None:
            return speaker.cms
        alpha = speaker.vas_l / self.params.volume_l
        return speaker.cms / (1.0 + alpha)
