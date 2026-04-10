from __future__ import annotations

from dataclasses import dataclass
from math import pi

from mfb.core.transfer import Transfer, gain, make_tf, parallel
from mfb.params.controller import PIDParams


@dataclass
class PIDController:
    params: PIDParams

    def transfer(self) -> Transfer:
        wd = 2 * pi * self.params.f_d_hz
        cp = gain(self.params.kp)
        ci = make_tf([self.params.ki], [1.0, 0.0])
        cd = make_tf([self.params.kd, 0.0], [1.0 / wd, 1.0])
        return parallel(cp, ci, cd)
