"""Simple transfer-function helpers used across the MFB package.

The goal is to keep control math easy to read for this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import signal

from mfb.core.math_helpers import as_float_array, hz_to_rad_s


@dataclass(frozen=True)
class Transfer:
    """Light wrapper around ``scipy.signal.TransferFunction``."""

    tf: signal.TransferFunction

    @property
    def num(self) -> np.ndarray:
        """Numerator polynomial in descending powers of *s*."""
        return as_float_array(self.tf.num)

    @property
    def den(self) -> np.ndarray:
        """Denominator polynomial in descending powers of *s*."""
        return as_float_array(self.tf.den)


def make_tf(num: Sequence[float], den: Sequence[float]) -> Transfer:
    """Build a continuous-time transfer function from polynomial coefficients."""
    return Transfer(signal.TransferFunction(as_float_array(num), as_float_array(den)))


def gain(k: float) -> Transfer:
    """Return a static gain block ``k``."""
    return make_tf([k], [1.0])


def first_order_pole(freq_hz: float) -> Transfer:
    """Return ``1 / (1 + s/wp)`` with ``wp = 2*pi*freq_hz``."""
    wp = hz_to_rad_s(freq_hz)
    return make_tf([1.0], [1.0 / wp, 1.0])


def first_order_zero(freq_hz: float) -> Transfer:
    """Return ``1 + s/wz`` with ``wz = 2*pi*freq_hz``."""
    wz = hz_to_rad_s(freq_hz)
    return make_tf([1.0 / wz, 1.0], [1.0])


def series(*systems: Transfer) -> Transfer:
    """Connect transfer functions in series."""
    num = np.array([1.0])
    den = np.array([1.0])
    for sys in systems:
        num = np.polymul(num, sys.num)
        den = np.polymul(den, sys.den)
    return make_tf(num, den)


def parallel(*systems: Transfer) -> Transfer:
    """Connect transfer functions in parallel (sum)."""
    if not systems:
        raise ValueError("parallel() needs at least one system")
    out = systems[0]
    for sys in systems[1:]:
        num = np.polyadd(np.polymul(out.num, sys.den), np.polymul(sys.num, out.den))
        den = np.polymul(out.den, sys.den)
        out = make_tf(num, den)
    return out


def feedback_negative(forward: Transfer, feedback: Transfer | float = 1.0) -> Transfer:
    """Return closed loop ``forward / (1 + forward*feedback)``."""
    fb = gain(float(feedback)) if isinstance(feedback, (int, float)) else feedback
    num = np.polymul(forward.num, fb.den)
    den = np.polyadd(np.polymul(forward.den, fb.den), np.polymul(forward.num, fb.num))
    return make_tf(num, den)


def sensitivity(open_loop: Transfer) -> Transfer:
    """Return sensitivity function ``S = 1/(1+L)`` for open loop ``L``."""
    return make_tf(open_loop.den, np.polyadd(open_loop.den, open_loop.num))


def freq_response(system: Transfer, freq_hz: Iterable[float]) -> np.ndarray:
    """Complex frequency response evaluated on a Hz grid."""
    w = hz_to_rad_s(as_float_array(list(freq_hz)))
    _, h = signal.freqresp(system.tf, w)
    return h


def magnitude_db(system: Transfer, freq_hz: Iterable[float]) -> np.ndarray:
    """Magnitude in dB on a Hz grid."""
    return 20.0 * np.log10(np.maximum(np.abs(freq_response(system, freq_hz)), 1e-20))


def phase_deg(system: Transfer, freq_hz: Iterable[float]) -> np.ndarray:
    """Unwrapped phase in degrees on a Hz grid."""
    return np.unwrap(np.angle(freq_response(system, freq_hz))) * 180.0 / np.pi
