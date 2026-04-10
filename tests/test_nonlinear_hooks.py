import unittest

from mfb.nonlinear.bl_curve import BlCurve
from mfb.nonlinear.limiter import HardLimiter


class NonlinearHookTests(unittest.TestCase):
    def test_disabled_curve_passthrough(self):
        self.assertEqual(BlCurve(enabled=False).value(2.0), 2.0)

    def test_limiter_clamps(self):
        self.assertEqual(HardLimiter(1.0).apply(3.0), 1.0)


if __name__ == "__main__":
    unittest.main()
