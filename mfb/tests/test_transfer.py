"""Tests for core transfer-function helpers."""

import numpy as np

from mfb.core.transfer import feedback_negative, gain, magnitude_db, series


def test_series_gain_block():
    """Two gains in series should multiply."""
    sys = series(gain(2.0), gain(3.0))
    assert np.isclose(sys.num[-1] / sys.den[-1], 6.0)


def test_feedback_gain_block():
    """Closed-loop gain for k with unity feedback is k/(1+k)."""
    closed = feedback_negative(gain(10.0), 1.0)
    assert np.isclose(closed.num[-1] / closed.den[-1], 10.0 / 11.0)


def test_gain_magnitude_db():
    """Magnitude of gain(2) should be about +6 dB."""
    vals = magnitude_db(gain(2.0), np.array([10.0]))
    assert vals[0] > 5.9
