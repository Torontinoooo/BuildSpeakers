from __future__ import annotations

from dataclasses import dataclass

from mfb.components.controller import PIDController
from mfb.components.injector import ErrorInjector
from mfb.components.sensor import Accelerometer
from mfb.components.speaker_plant import Amplifier, SpeakerPlant, amplified_speaker
from mfb.core.transfer import Transfer, feedback_negative, sensitivity, series


@dataclass
class InjectedVoltageLoop:
    amplifier: Amplifier
    plant: SpeakerPlant
    sensor: Accelerometer
    controller: PIDController
    injector: ErrorInjector

    def plant_transfer(self) -> Transfer:
        return amplified_speaker(self.amplifier, self.plant)

    def open_loop(self) -> Transfer:
        return series(
            self.injector.transfer(),
            self.controller.transfer(),
            self.plant_transfer(),
            self.sensor.transfer(),
        )

    def closed_loop(self) -> Transfer:
        return feedback_negative(series(self.controller.transfer(), self.plant_transfer()), self.sensor.transfer())

    def sensitivity(self) -> Transfer:
        return sensitivity(self.open_loop())
