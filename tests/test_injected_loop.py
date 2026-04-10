import unittest

from mfb.analysis.stability import stable
from mfb.studies.compare_boxes import build_acoustic


class InjectedLoopTests(unittest.TestCase):
    def test_closed_loop_stable(self):
        sys = build_acoustic(3.0).loop.closed_loop()
        self.assertTrue(stable(sys))


if __name__ == "__main__":
    unittest.main()
