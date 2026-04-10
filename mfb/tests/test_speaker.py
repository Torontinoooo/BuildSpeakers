"""Tests for speaker/box behavior."""

from mfb.blocks.box import sealed_box_fc
from mfb.blocks.speaker import speaker_tf
from mfb.params.speaker_params import SpeakerParams


def test_small_box_raises_resonance():
    """A small sealed box should raise resonance above free-air Fs."""
    spk = SpeakerParams()
    assert sealed_box_fc(spk, 3.0) > spk.fs_hz


def test_speaker_tf_is_proper():
    """Speaker block denominator order is greater than numerator order."""
    spk = SpeakerParams()
    sys = speaker_tf(spk, 3.0)
    assert len(sys.den) > len(sys.num)
