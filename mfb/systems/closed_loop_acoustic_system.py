from __future__ import annotations

from dataclasses import dataclass

from mfb.components.acoustic import MonopoleAcoustic
from mfb.systems.injected_voltage_loop import InjectedVoltageLoop
from mfb.core.transfer import Transfer, series


@dataclass
class ClosedLoopAcousticSystem:
    loop: InjectedVoltageLoop
    acoustic: MonopoleAcoustic

    def open_acoustic(self) -> Transfer:
        return series(self.acoustic.transfer(), self.loop.plant_transfer())

    def closed_acoustic(self) -> Transfer:
        return series(self.acoustic.transfer(), self.loop.closed_loop())
