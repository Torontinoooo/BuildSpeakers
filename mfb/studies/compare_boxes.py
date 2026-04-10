from __future__ import annotations

import numpy as np

from mfb.analysis.frequency import normalized_magnitude_db
from mfb.components.acoustic import MonopoleAcoustic
from mfb.components.box import SealedBox
from mfb.components.controller import PIDController
from mfb.components.injector import ErrorInjector
from mfb.components.sensor import Accelerometer
from mfb.components.speaker_plant import Amplifier, SpeakerPlant
from mfb.params.box import AcousticParams, BoxParams
from mfb.params.controller import PIDParams
from mfb.params.injection import AmplifierParams, InjectionParams
from mfb.params.sensor import SensorParams
from mfb.params.speaker import SpeakerParams
from mfb.systems.closed_loop_acoustic_system import ClosedLoopAcousticSystem
from mfb.systems.injected_voltage_loop import InjectedVoltageLoop


def build_acoustic(volume_l: float | None) -> ClosedLoopAcousticSystem:
    spk = SpeakerParams()
    loop = InjectedVoltageLoop(
        amplifier=Amplifier(AmplifierParams()),
        plant=SpeakerPlant(spk, SealedBox(BoxParams(volume_l))),
        sensor=Accelerometer(SensorParams()),
        controller=PIDController(PIDParams()),
        injector=ErrorInjector(InjectionParams()),
    )
    return ClosedLoopAcousticSystem(loop=loop, acoustic=MonopoleAcoustic(spk, AcousticParams()))


def run() -> dict[str, np.ndarray]:
    f = np.logspace(0, 4, 2000)
    free_sys = build_acoustic(None)
    small_sys = build_acoustic(3.0)
    return {
        "f_hz": f,
        "free_open": normalized_magnitude_db(free_sys.open_acoustic(), f),
        "free_closed": normalized_magnitude_db(free_sys.closed_acoustic(), f),
        "small_open": normalized_magnitude_db(small_sys.open_acoustic(), f),
        "small_closed": normalized_magnitude_db(small_sys.closed_acoustic(), f),
    }


if __name__ == "__main__":
    result = run()
    print("Computed comparison arrays:", ", ".join(result.keys()))
