from __future__ import annotations

from mfb.config.params import LinkwitzTransformParams, OutputFilterParams, SpeakerParams
from mfb.utils.transfer_helpers import TFMath, Transfer


class LinkwitzTransform:
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
        wa = 2.0 * 3.141592653589793 * f_actual
        wt = 2.0 * 3.141592653589793 * f_target
        numerator = [1.0 / wa**2, 1.0 / (q_actual * wa), 1.0]
        denominator = [1.0 / wt**2, 1.0 / (q_target * wt), 1.0]
        return TFMath.tf(numerator, denominator)


class OutputLowPass:
    def __init__(self, params: OutputFilterParams) -> None:
        self.params = params

    def transfer_function(self) -> Transfer:
        if not self.params.enabled:
            return TFMath.gain(1.0)
        if self.params.lowpass_order <= 1:
            return TFMath.first_order_lowpass(self.params.lowpass_hz)
        if self.params.family == "linkwitz_riley":
            return TFMath.linkwitz_riley_lowpass(self.params.lowpass_hz, self.params.lowpass_order)
        if self.params.family == "bessel":
            return TFMath.analog_bessel_lowpass(self.params.lowpass_hz, self.params.lowpass_order)
        return TFMath.analog_butter_lowpass(self.params.lowpass_hz, self.params.lowpass_order)


class RepairFilterChain:
    """R(s): command shaping path."""

    def __init__(self, speaker: SpeakerParams, linkwitz: LinkwitzTransformParams, output_filter: OutputFilterParams) -> None:
        self.speaker = speaker
        self.linkwitz = linkwitz
        self.output_filter = output_filter

    def transfer_function(self) -> Transfer:
        return TFMath.series(
            LinkwitzTransform(self.speaker, self.linkwitz).transfer_function(),
            OutputLowPass(self.output_filter).transfer_function(),
        )
