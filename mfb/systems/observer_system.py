from __future__ import annotations

from dataclasses import dataclass

from mfb.systems.injected_voltage_loop import InjectedVoltageLoop


@dataclass
class ObserverSystem:
    """Placeholder for future state observer integration."""

    loop: InjectedVoltageLoop

    def estimated_acceleration_from_error(self, error_v: float) -> float:
        sensor_dc = self.loop.sensor.params.dc_v_per_mps2
        return error_v / sensor_dc
