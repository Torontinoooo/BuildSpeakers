from __future__ import annotations

import math

from mfb.config.params import PIDParams
from mfb.utils.transfer_helpers import TFMath, Transfer


class PIDController:
    """C(s) = Kp + Ki/(s+wi) + Kd*s/(1+s/wd)."""

    def __init__(self, params: PIDParams) -> None:
        self.params = params

    def pid_core_transfer(self) -> Transfer:
        terms: list[Transfer] = [TFMath.gain(self.params.resolved_kp)]
        if self.params.controller_type in {"PI", "PID"} and self.params.resolved_ki > 0.0:
            wi = 2.0 * math.pi * self.params.resolved_integral_leak_hz
            terms.append(TFMath.tf([self.params.resolved_ki], [1.0, wi]))
        if self.params.controller_type == "PID" and self.params.resolved_kd > 0.0:
            terms.append(TFMath.differentiator_with_rolloff(self.params.resolved_derivative_hz, self.params.resolved_kd))
        return TFMath.parallel(*terms)

    def corr_lp_transfer(self) -> Transfer:
        if not self.params.corr_lp_enabled:
            return TFMath.gain(1.0)
        if self.params.corr_lp_order <= 1:
            return TFMath.first_order_lowpass(self.params.corr_lp_hz)
        if self.params.corr_lp_family == "linkwitz_riley":
            return TFMath.linkwitz_riley_lowpass(self.params.corr_lp_hz, self.params.corr_lp_order)
        if self.params.corr_lp_family == "butterworth":
            return TFMath.analog_butter_lowpass(self.params.corr_lp_hz, self.params.corr_lp_order)
        return TFMath.analog_bessel_lowpass(self.params.corr_lp_hz, self.params.corr_lp_order)

    def transfer_function(self) -> Transfer:
        return TFMath.series(self.pid_core_transfer(), self.corr_lp_transfer())
