from __future__ import annotations

from mfb.nonlinear.bl_curve import BlCurve


if __name__ == "__main__":
    curve = BlCurve(enabled=True, slope=0.1)
    print([curve.value(x) for x in (-2.0, -1.0, 0.0, 1.0, 2.0)])
