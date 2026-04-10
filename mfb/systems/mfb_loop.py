"""System assembly for the accelerometer-based MFB loudspeaker loop."""

from __future__ import annotations

from dataclasses import dataclass

from mfb.blocks.acoustic_model import acoustic_pressure_tf
from mfb.blocks.amplifier import amplifier_tf
from mfb.blocks.controller import controller_tf
from mfb.blocks.sensor import sensor_tf
from mfb.blocks.speaker import speaker_tf
from mfb.core.transfer import Transfer, feedback_negative, sensitivity, series
from mfb.params.amp_params import AmpParams
from mfb.params.speaker_params import SpeakerParams
from mfb.params.sensor_params import SensorParams


@dataclass(frozen=True)
class MFBLoop:
    """Container with all key transfer functions for one box setting."""

    g_amp: Transfer
    g_spk: Transfer
    g_sensor: Transfer
    g_acoustic: Transfer
    c: Transfer
    g: Transfer
    l: Transfer
    t: Transfer
    s: Transfer
    h_open_acoustic: Transfer
    h_closed_acoustic: Transfer


def build_mfb_loop(box_volume_l: float | None, controller_mode: str = "pid") -> MFBLoop:
    """Build and return all loop transfer functions.

    The notation follows the paper convention:
    ``L = C * G * M``, ``T = C*G / (1 + C*G*M)``, ``S = 1/(1+L)``.
    """
    spk = SpeakerParams()
    g_amp = amplifier_tf(AmpParams())
    g_spk = speaker_tf(spk, box_volume_l)
    g_sensor = sensor_tf(SensorParams())
    g_acoustic = acoustic_pressure_tf(spk)
    c = controller_tf(controller_mode)

    g = series(g_amp, g_spk)
    l = series(c, g, g_sensor)
    t = feedback_negative(series(c, g), g_sensor)
    s = sensitivity(l)

    return MFBLoop(
        g_amp=g_amp,
        g_spk=g_spk,
        g_sensor=g_sensor,
        g_acoustic=g_acoustic,
        c=c,
        g=g,
        l=l,
        t=t,
        s=s,
        h_open_acoustic=series(g_acoustic, g),
        h_closed_acoustic=series(g_acoustic, t),
    )
