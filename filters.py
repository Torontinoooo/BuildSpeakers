from __future__ import annotations

"""Analogue-style control and shaping filters.

This module follows the signal decomposition used in the current project:
- F(s): measurement cleanup in the feedback path
- C(s): controller
- R(s): command / repair path

The controller-shaping philosophy also follows the practical loop-shaping notes in
Schneider et al. (extra low-frequency gain, then high-frequency roll-off for noise
and breakup) and the RMS design paper sections on controller design.
"""

import math

from .params import FeedbackFilterParams, PIDParams, LinkwitzTransformParams, OutputFilterParams, SpeakerParams
from .transfer_helpers import TFMath, Transfer, TransferBlock


class OffsetHighPass:
    def __init__(self, params: FeedbackFilterParams) -> None:
        self.params = params

    def transfer_function(self) -> Transfer:
        if not self.params.offset_hp_enabled:
            return TFMath.gain(1.0)
        if self.params.offset_hp_order <= 1:
            return TFMath.first_order_highpass(self.params.offset_hp_hz)
        family = self.params.offset_hp_family
        if self.params.offset_hp_order == 2:
            if family == "butterworth":
                return TFMath.butter2_highpass(self.params.offset_hp_hz)
            return TFMath.analog_bessel_highpass(self.params.offset_hp_hz, 2)
        if family == "butterworth":
            return TFMath.analog_butter_highpass(self.params.offset_hp_hz, self.params.offset_hp_order)
        return TFMath.analog_bessel_highpass(self.params.offset_hp_hz, self.params.offset_hp_order)

    def block(self) -> TransferBlock:
        return TransferBlock("OFFHP(s)", self.transfer_function())


class NoiseLowPass:
    def __init__(self, params: FeedbackFilterParams) -> None:
        self.params = params

    def transfer_function(self) -> Transfer:
        if not self.params.noise_lp_enabled:
            return TFMath.gain(1.0)
        if self.params.noise_lp_order <= 1:
            return TFMath.first_order_lowpass(self.params.noise_lp_hz)
        family = self.params.noise_lp_family
        if self.params.noise_lp_order == 2:
            if family == "butterworth":
                return TFMath.butter2_lowpass(self.params.noise_lp_hz)
            return TFMath.analog_bessel_lowpass(self.params.noise_lp_hz, 2)
        if family == "butterworth":
            return TFMath.analog_butter_lowpass(self.params.noise_lp_hz, self.params.noise_lp_order)
        return TFMath.analog_bessel_lowpass(self.params.noise_lp_hz, self.params.noise_lp_order)

    def block(self) -> TransferBlock:
        return TransferBlock("NoiseLP(s)", self.transfer_function())


class PIDController:
    """Continuous-time PID controller with practical derivative roll-off.

    C(s) = Kp + Ki / (s + wi) + Kd*s / (1 + s/wd)
    """

    def __init__(self, params: PIDParams) -> None:
        self.params = params

    def transfer_function(self) -> Transfer:
        p_term = TFMath.gain(self.params.resolved_kp)
        terms: list[Transfer] = [p_term]

        if self.params.controller_type in {"PI", "PID"} and self.params.resolved_ki > 0.0:
            wi = 2.0 * math.pi * self.params.resolved_integral_leak_hz
            terms.append(TFMath.tf([self.params.resolved_ki], [1.0, wi]))

        if self.params.controller_type == "PID" and self.params.resolved_kd > 0.0:
            terms.append(TFMath.differentiator_with_rolloff(self.params.resolved_derivative_hz, self.params.resolved_kd))

        return TFMath.parallel(*terms)

    def block(self) -> TransferBlock:
        return TransferBlock("PID(s)", self.transfer_function())


class LinkwitzTransform:
    """Command-path Linkwitz transform for closed-box retuning."""

    def __init__(self, speaker: SpeakerParams, params: LinkwitzTransformParams, actual_box_L: float | None = None) -> None:
        self.speaker = speaker
        self.params = params
        self.actual_box_L = speaker.actual_box_volume_L if actual_box_L is None else actual_box_L

    def transfer_function(self) -> Transfer:
        if not self.params.enabled:
            return TFMath.gain(1.0)
        f_actual = self.speaker.sealed_box_resonance(self.actual_box_L)
        q_actual = self.speaker.sealed_box_q(self.actual_box_L)
        target_box_l = self.params.resolve_target_box_L(self.speaker)
        f_target = self.speaker.sealed_box_resonance(target_box_l)
        q_target = self.params.target_qtc
        wa = 2.0 * math.pi * f_actual
        wt = 2.0 * math.pi * f_target
        numerator = [1.0 / wa**2, 1.0 / (q_actual * wa), 1.0]
        denominator = [1.0 / wt**2, 1.0 / (q_target * wt), 1.0]
        return TFMath.tf(numerator, denominator)

    def block(self) -> TransferBlock:
        return TransferBlock("LinkT(s)", self.transfer_function())


class OutputLowPass:
    def __init__(self, params: OutputFilterParams) -> None:
        self.params = params

    def transfer_function(self) -> Transfer:
        if not self.params.enabled:
            return TFMath.gain(1.0)
        if self.params.lowpass_order <= 1:
            return TFMath.first_order_lowpass(self.params.lowpass_hz)
        if self.params.lowpass_order == 2:
            return TFMath.butter2_lowpass(self.params.lowpass_hz, q=self.params.lowpass_q)
        return TFMath.analog_butter_lowpass(self.params.lowpass_hz, self.params.lowpass_order)

    def block(self) -> TransferBlock:
        return TransferBlock("LP(s)", self.transfer_function())


class FeedbackMeasurementChain:
    def __init__(self, params: FeedbackFilterParams) -> None:
        self.offset_hp = OffsetHighPass(params)
        self.noise_lp = NoiseLowPass(params)

    def transfer_function(self) -> Transfer:
        return TFMath.series(self.offset_hp.transfer_function(), self.noise_lp.transfer_function())

    def blocks(self) -> list[TransferBlock]:
        return [self.offset_hp.block(), self.noise_lp.block()]

    def block(self) -> TransferBlock:
        return TransferBlock("F(s)", self.transfer_function())


class RepairFilterChain:
    def __init__(self, speaker: SpeakerParams, linkwitz: LinkwitzTransformParams, output_filter: OutputFilterParams, actual_box_l: float | None = None) -> None:
        self.linkwitz = LinkwitzTransform(speaker, linkwitz, actual_box_l)
        self.output_lowpass = OutputLowPass(output_filter)

    def transfer_function(self) -> Transfer:
        return TFMath.series(self.linkwitz.transfer_function(), self.output_lowpass.transfer_function())

    def blocks(self) -> list[TransferBlock]:
        return [self.linkwitz.block(), self.output_lowpass.block()]

    def block(self) -> TransferBlock:
        return TransferBlock("R(s)", self.transfer_function())
