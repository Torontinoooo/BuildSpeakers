from __future__ import annotations

from dataclasses import dataclass
from math import pi

from mfb.core.transfer import Transfer, make_tf, series
from mfb.params.sensor import SensorParams


@dataclass
class Accelerometer:
    params: SensorParams

    def transfer(self) -> Transfer:
        wn = 2 * pi * self.params.f_res_hz
        wa = 2 * pi * self.params.f_output_amp_hz
        mech = make_tf([self.params.dc_v_per_mps2 * wn**2], [1.0, 2 * self.params.zeta * wn, wn**2])
        out = make_tf([wa], [1.0, wa])
        return series(mech, out)
