"""Sensitivity helpers for loop analysis."""

from mfb.core.transfer import Transfer, magnitude_db


def peak_sensitivity_db(s: Transfer, freq_hz) -> float:
    """Return the largest |S| value in dB over a frequency grid."""
    return float(magnitude_db(s, freq_hz).max())
