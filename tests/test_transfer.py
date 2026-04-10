import unittest

import numpy as np

from mfb.core.transfer import feedback_negative, gain, magnitude_db, series


class TransferTests(unittest.TestCase):
    def test_series_gain(self):
        sys = series(gain(2.0), gain(3.0))
        self.assertAlmostEqual(sys.num[-1] / sys.den[-1], 6.0)

    def test_feedback_gain(self):
        closed = feedback_negative(gain(10), 1.0)
        self.assertAlmostEqual(closed.num[-1] / closed.den[-1], 10 / 11)

    def test_mag_db(self):
        vals = magnitude_db(gain(2.0), np.array([10.0]))
        self.assertGreater(vals[0], 5.9)


if __name__ == "__main__":
    unittest.main()
