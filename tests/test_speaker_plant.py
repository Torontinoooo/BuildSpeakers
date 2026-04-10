import unittest

from mfb.components.box import SealedBox
from mfb.components.speaker_plant import SpeakerPlant
from mfb.params.box import BoxParams
from mfb.params.speaker import SpeakerParams


class SpeakerPlantTests(unittest.TestCase):
    def test_box_reduces_compliance(self):
        spk = SpeakerParams()
        free = SealedBox(BoxParams(None)).total_compliance(spk)
        small = SealedBox(BoxParams(3.0)).total_compliance(spk)
        self.assertLess(small, free)

    def test_transfer_has_expected_order(self):
        sys = SpeakerPlant(SpeakerParams(), SealedBox(BoxParams(3.0))).transfer()
        self.assertEqual(len(sys.den), 4)


if __name__ == "__main__":
    unittest.main()
