from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Transfer:
    """Small wrapper around scipy continuous-time transfer functions."""

    tf: signal.TransferFunction

    @property
    def num(self) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.tf.num, dtype=float))

    @property
    def den(self) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.tf.den, dtype=float))


def make_tf(num: Sequence[float], den: Sequence[float]) -> Transfer:
    return Transfer(signal.TransferFunction(np.asarray(num, dtype=float), np.asarray(den, dtype=float)))


def gain(k: float) -> Transfer:
    return make_tf([k], [1.0])


def series(*systems: Transfer) -> Transfer:
    num = np.array([1.0])
    den = np.array([1.0])
    for sys in systems:
        num = np.polymul(num, sys.num)
        den = np.polymul(den, sys.den)
    return make_tf(num, den)


def parallel(*systems: Transfer) -> Transfer:
    result = systems[0]
    for sys in systems[1:]:
        num = np.polyadd(np.polymul(result.num, sys.den), np.polymul(sys.num, result.den))
        den = np.polymul(result.den, sys.den)
        result = make_tf(num, den)
    return result


def feedback_negative(forward: Transfer, feedback: Transfer | float = 1.0) -> Transfer:
    fb = gain(float(feedback)) if isinstance(feedback, (int, float)) else feedback
    num = np.polymul(forward.num, fb.den)
    den = np.polyadd(np.polymul(forward.den, fb.den), np.polymul(forward.num, fb.num))
    return make_tf(num, den)


def sensitivity(open_loop: Transfer) -> Transfer:
    return make_tf(open_loop.den, np.polyadd(open_loop.den, open_loop.num))


def freq_response(system: Transfer, freq_hz: Iterable[float]) -> np.ndarray:
    w = 2 * np.pi * np.asarray(list(freq_hz), dtype=float)
    _, h = signal.freqresp(system.tf, w)
    return h


def magnitude_db(system: Transfer, freq_hz: Iterable[float]) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(freq_response(system, freq_hz)), 1e-20))


def phase_deg(system: Transfer, freq_hz: Iterable[float]) -> np.ndarray:
    return np.unwrap(np.angle(freq_response(system, freq_hz))) * 180.0 / np.pi
