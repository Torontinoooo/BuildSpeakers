from __future__ import annotations

from mfb.config.params import FeedbackFilterParams
from mfb.utils.transfer_helpers import TFMath, Transfer


class OFFHPFilter:
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


class FeedbackMeasurementChain:
    """F(s): feedback-path correction filter chain.

    For now this is CorrHP(s) only. Additional filters can be appended later.
    """

    def __init__(self, params: FeedbackFilterParams) -> None:
        self.params = params

    def transfer_function(self) -> Transfer:
        return OFFHPFilter(self.params).transfer_function()


# Backward-compatible alias.
OffsetHighPass = OFFHPFilter
