from __future__ import annotations

from mfb.config.params import AmplifierParams
from mfb.utils.transfer_helpers import TFMath, Transfer, TransferBlock


class AmplifierModel:
    """AMP(s): gain, optional analogue/DSP low-pass, optional delay."""

    def __init__(self, params: AmplifierParams) -> None:
        self.params = params

    def _lowpass_transfer(self) -> Transfer:
        if not self.params.lowpass_enabled:
            return TFMath.gain(1.0)
        if self.params.lowpass_family == "linkwitz_riley":
            return TFMath.linkwitz_riley_lowpass(self.params.lowpass_hz, self.params.lowpass_order)
        if self.params.lowpass_family == "butterworth":
            return TFMath.analog_butter_lowpass(self.params.lowpass_hz, self.params.lowpass_order)
        return TFMath.analog_bessel_lowpass(self.params.lowpass_hz, self.params.lowpass_order)

    def _delay_transfer(self) -> Transfer:
        if not self.params.dsp_delay_enabled:
            return TFMath.gain(1.0)
        total_delay = self.params.dsp_fixed_latency_s + self.params.output_alignment_delay_s
        return TFMath.pade_delay(total_delay, self.params.pade_order)

    def transfer_function(self) -> Transfer:
        linear_gain = 10.0 ** (self.params.gain_db / 20.0)
        return TFMath.series(TFMath.gain(linear_gain), self._lowpass_transfer(), self._delay_transfer())

    def block(self) -> TransferBlock:
        return TransferBlock("AMP(s)", self.transfer_function())
