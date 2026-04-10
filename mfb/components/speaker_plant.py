from __future__ import annotations

from dataclasses import dataclass

from mfb.components.box import SealedBox
from mfb.params.injection import AmplifierParams
from mfb.params.speaker import SpeakerParams
from mfb.core.transfer import Transfer, gain, make_tf, series


@dataclass
class Amplifier:
    """Flat amplifier model.

    Transfer function:
        G_amp(s) = K_amp,  K_amp = 10^(gain_db/20)
    """

    params: AmplifierParams

    def transfer(self) -> Transfer:
        return gain(self.params.gain_vv)


@dataclass
class SpeakerPlant:
    speaker: SpeakerParams
    box: SealedBox

    def transfer(self) -> Transfer:
        c_tot = self.box.total_compliance(self.speaker)
        num = [self.speaker.bl, 0.0, 0.0]
        den = [
            self.speaker.le * self.speaker.mms,
            self.speaker.le * self.speaker.rms + self.speaker.re * self.speaker.mms,
            self.speaker.le / c_tot + self.speaker.re * self.speaker.rms + self.speaker.bl**2,
            self.speaker.re / c_tot,
        ]
        return make_tf(num, den)


def amplified_speaker(amplifier: Amplifier, speaker: SpeakerPlant) -> Transfer:
    return series(amplifier.transfer(), speaker.transfer())
