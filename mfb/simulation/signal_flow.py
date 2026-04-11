from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal

from mfb.control.limiter import SignalLimiter
from mfb.control.prefilter import PrefilterModel
from mfb.control.loop_assembler import MFBLoop
from mfb.plant.source import SourceModel
from mfb.plant.summing_amp import SummingAmplifierModel


@dataclass(slots=True)
class TimeDomainTrace:
    t: np.ndarray
    u_ref: np.ndarray
    u_cmd: np.ndarray
    u_sum: np.ndarray
    u_ctrl: np.ndarray
    v_spk: np.ndarray
    a_cone: np.ndarray
    u_sens: np.ndarray
    u_meas: np.ndarray
    u_fb: np.ndarray


class ClosedLoopSignalFlow:
    """Discrete-time experiment flow aligned with the paper block chain."""

    def __init__(
        self,
        loop: MFBLoop,
        source: SourceModel,
        prefilter: PrefilterModel,
        limiter: SignalLimiter,
        summer: SummingAmplifierModel,
    ) -> None:
        self.loop = loop
        self.source = source
        self.prefilter = prefilter
        self.limiter = limiter
        self.summer = summer

    def run(self) -> TimeDomainTrace:
        t, u_ref = self.source.generate_voltage()
        u_cmd = self.limiter.process(self.prefilter.process(u_ref))

        u_fb = np.zeros_like(u_cmd)
        u_sum = self.summer.mix(u_cmd, u_fb)
        _, u_ctrl, _ = signal.lsim(self.loop.blocks.control.controller, U=u_sum, T=t)
        _, v_spk, _ = signal.lsim(self.loop.blocks.plant.amplifier, U=u_ctrl, T=t)
        _, a_cone, _ = signal.lsim(self.loop.blocks.plant.loudspeaker_small, U=v_spk, T=t)
        _, u_sens, _ = signal.lsim(self.loop.blocks.plant.accelerometer, U=a_cone, T=t)
        _, u_meas, _ = signal.lsim(self.loop.blocks.control.feedback_filter, U=u_sens, T=t)
        u_fb = u_meas

        u_sum = self.summer.mix(u_cmd, u_fb)
        return TimeDomainTrace(t, u_ref, u_cmd, u_sum, u_ctrl, v_spk, a_cone, u_sens, u_meas, u_fb)
