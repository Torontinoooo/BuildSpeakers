"""Tests for assembled MFB loop."""

import numpy as np

from mfb.analysis.stability import stable
from mfb.core.transfer import magnitude_db
from mfb.systems.mfb_loop import build_mfb_loop


def test_build_loop_returns_stable_default():
    """Default PID setup for the small box should be stable in this model."""
    loop = build_mfb_loop(3.0, controller_mode="pid")
    assert stable(loop.t)


def test_sensitivity_exists_on_grid():
    """Sensitivity should produce finite magnitudes on a test grid."""
    loop = build_mfb_loop(3.0)
    f = np.logspace(0, 4, 100)
    s_db = magnitude_db(loop.s, f)
    assert np.isfinite(s_db).all()
