import unittest

from mfb.components.sensor import Accelerometer
from mfb.params.sensor import SensorParams


class SensorTests(unittest.TestCase):
    def test_dc_sensitivity_positive(self):
        self.assertGreater(SensorParams().dc_v_per_mps2, 0)

    def test_sensor_tf_exists(self):
        tf = Accelerometer(SensorParams()).transfer()
        self.assertGreater(len(tf.den), 1)


if __name__ == "__main__":
    unittest.main()
