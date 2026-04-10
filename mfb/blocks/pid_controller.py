"""PID controller block with derivative filtering."""

from dataclasses import dataclass

import numpy as np

from mfb.core.transfer import Transfer, gain, make_tf, parallel


@dataclass(frozen=True)
class PIDParams:
    """Continuous-time PID settings."""

    kp: float = 1.4
    ki: float = 0.05
    kd: float = 1e-4
    f_d_hz: float = 700.0


def pid_controller_tf(params: PIDParams) -> Transfer:
    """Return ``Kp + Ki/s + Kd*s/(1+s/wd)``."""
    wd = 2.0 * np.pi * params.f_d_hz
    cp = gain(params.kp)
    ci = make_tf([params.ki], [1.0, 0.0])
    cd = make_tf([params.kd, 0.0], [1.0 / wd, 1.0])
    return parallel(cp, ci, cd)
