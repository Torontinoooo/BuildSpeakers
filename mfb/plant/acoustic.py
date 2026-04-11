from __future__ import annotations

import math

from mfb.config.params import SpeakerParams
from mfb.utils.transfer_helpers import TFMath, Transfer, TransferBlock


class AcousticRadiationModel:
    """Very-low-frequency monopole approximation from acceleration to pressure."""

    def __init__(self, speaker: SpeakerParams, rho_air: float, reference_distance_m: float) -> None:
        self.speaker = speaker
        self.rho_air = rho_air
        self.reference_distance_m = reference_distance_m

    def transfer_function(self) -> Transfer:
        gain = self.rho_air * self.speaker.Sd / (4.0 * math.pi * self.reference_distance_m)
        return TFMath.gain(gain)

    def block(self) -> TransferBlock:
        return TransferBlock("Aco(s)", self.transfer_function())
