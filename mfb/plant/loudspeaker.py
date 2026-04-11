from __future__ import annotations

from mfb.config.params import SpeakerParams
from mfb.utils.transfer_helpers import TFMath, Transfer, TransferBlock


class LoudspeakerModel:
    """SPK(s): drive voltage to cone acceleration for a sealed box."""

    def __init__(self, speaker: SpeakerParams, box_volume_L: float | None = None) -> None:
        self.speaker = speaker
        self.box_volume_L = speaker.actual_box_volume_L if box_volume_L is None else box_volume_L

    def transfer_function(self) -> Transfer:
        numerator, denominator = self.speaker.voltage_to_acceleration_coefficients(self.box_volume_L)
        return TFMath.tf(numerator, denominator)

    def block(self) -> TransferBlock:
        return TransferBlock("SPK(s)", self.transfer_function())
