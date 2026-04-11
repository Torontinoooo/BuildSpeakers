from __future__ import annotations

"""Linear plant transfer functions.

Implemented equation (voltage to acceleration):
G_a/v(s) = Bl s^2 / [Bl^2 s + (s Le + Re)(s^2 Mms + s Rms + 1/Cms)]
"""

from mfb.config.params import SpeakerParams


def voltage_to_acceleration_coefficients(speaker: SpeakerParams, box_volume_L: float | None = None) -> tuple[list[float], list[float]]:
    return speaker.voltage_to_acceleration_coefficients(box_volume_L)
