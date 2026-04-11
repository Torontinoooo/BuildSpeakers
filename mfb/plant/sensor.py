from __future__ import annotations

from mfb.config.params import SensorParams
from mfb.utils.transfer_helpers import TFMath, Transfer, TransferBlock


class AccelerometerModel:
    """AccV(s): cone acceleration to analogue sensor voltage."""

    def __init__(self, params: SensorParams) -> None:
        self.params = params

    def transfer_function(self) -> Transfer:
        base = TFMath.gain(self.params.dc_sensitivity_v_per_mps2)
        if self.params.sensor_model in {"adxl327", "adxl206"}:
            return TFMath.series(base, TFMath.first_order_lowpass(float(self.params.bandwidth_hz)))
        system = TFMath.series(base, TFMath.first_order_lowpass(float(self.params.bandwidth_hz)))
        if self.params.output_amplifier_bandwidth_hz is not None:
            system = TFMath.series(system, TFMath.first_order_lowpass(self.params.output_amplifier_bandwidth_hz))
        return system

    def block(self) -> TransferBlock:
        return TransferBlock("AccV(s)", self.transfer_function())
