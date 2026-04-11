from __future__ import annotations

"""Transfer-function helpers used across the MFB study package.

The package stays in continuous time on purpose. That keeps the signal-flow close to
classical analogue MFB block diagrams and to the plant/controller development used
in the AES and RMS references.
"""

from dataclasses import dataclass
from typing import Sequence
import math
import warnings

import numpy as np
from scipy import interpolate, signal
from scipy.linalg import LinAlgWarning


Transfer = signal.TransferFunction


@dataclass(slots=True)
class TransferBlock:
    name: str
    system: Transfer

    def bode(self, freq_hz: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        response = frequency_response(self.system, freq_hz)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-20))
        phase_deg = np.unwrap(np.angle(response)) * 180.0 / np.pi
        return magnitude_db, phase_deg

    def time_response(
        self,
        time_s: Sequence[float],
        input_signal: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        time = np.asarray(time_s, dtype=float)
        u = np.asarray(input_signal, dtype=float)
        tout, yout, _ = signal.lsim(self.system, U=u, T=time)
        return tout, yout


class TFMath:
    @staticmethod
    def poly1(values: np.ndarray | Sequence[float]) -> np.ndarray:
        return np.atleast_1d(np.asarray(values, dtype=float))

    @staticmethod
    def tf(num: Sequence[float], den: Sequence[float]) -> Transfer:
        return signal.TransferFunction(np.asarray(num, dtype=float), np.asarray(den, dtype=float))

    @staticmethod
    def gain(value: float) -> Transfer:
        return TFMath.tf([value], [1.0])

    @staticmethod
    def series(*systems: Transfer) -> Transfer:
        numerator = np.array([1.0])
        denominator = np.array([1.0])
        for system in systems:
            numerator = np.polymul(numerator, TFMath.poly1(system.num))
            denominator = np.polymul(denominator, TFMath.poly1(system.den))
        return TFMath.tf(numerator, denominator)

    @staticmethod
    def parallel(*systems: Transfer) -> Transfer:
        if not systems:
            return TFMath.gain(0.0)
        output = systems[0]
        for system in systems[1:]:
            numerator = np.polyadd(
                np.polymul(TFMath.poly1(output.num), TFMath.poly1(system.den)),
                np.polymul(TFMath.poly1(system.num), TFMath.poly1(output.den)),
            )
            denominator = np.polymul(TFMath.poly1(output.den), TFMath.poly1(system.den))
            output = TFMath.tf(numerator, denominator)
        return output

    @staticmethod
    def negative_feedback(forward: Transfer, feedback: Transfer | float = 1.0) -> Transfer:
        feedback_tf = TFMath.gain(float(feedback)) if isinstance(feedback, (int, float)) else feedback
        numerator = np.polymul(TFMath.poly1(forward.num), TFMath.poly1(feedback_tf.den))
        denominator = np.polyadd(
            np.polymul(TFMath.poly1(forward.den), TFMath.poly1(feedback_tf.den)),
            np.polymul(TFMath.poly1(forward.num), TFMath.poly1(feedback_tf.num)),
        )
        return TFMath.tf(numerator, denominator)

    @staticmethod
    def one_over_one_plus(system: Transfer) -> Transfer:
        return TFMath.tf(TFMath.poly1(system.den), np.polyadd(TFMath.poly1(system.den), TFMath.poly1(system.num)))

    @staticmethod
    def first_order_highpass(fc_hz: float) -> Transfer:
        wc = 2.0 * np.pi * fc_hz
        return TFMath.tf([1.0 / wc, 0.0], [1.0 / wc, 1.0])

    @staticmethod
    def first_order_lowpass(fc_hz: float) -> Transfer:
        wc = 2.0 * np.pi * fc_hz
        return TFMath.tf([1.0], [1.0 / wc, 1.0])

    @staticmethod
    def butter2_lowpass(fc_hz: float, q: float = 1.0 / np.sqrt(2.0)) -> Transfer:
        wc = 2.0 * np.pi * fc_hz
        return TFMath.tf([wc**2], [1.0, wc / q, wc**2])

    @staticmethod
    def butter2_highpass(fc_hz: float, q: float = 1.0 / np.sqrt(2.0)) -> Transfer:
        wc = 2.0 * np.pi * fc_hz
        return TFMath.tf([1.0, 0.0, 0.0], [1.0, wc / q, wc**2])

    @staticmethod
    def differentiator_with_rolloff(fc_hz: float, kd: float) -> Transfer:
        wc = 2.0 * np.pi * fc_hz
        return TFMath.tf([kd, 0.0], [1.0 / wc, 1.0])

    @staticmethod
    def analog_butter_lowpass(fc_hz: float, order: int) -> Transfer:
        num, den = signal.butter(order, 2.0 * np.pi * fc_hz, btype="low", analog=True, output="ba")
        return TFMath.tf(num, den)

    @staticmethod
    def analog_butter_highpass(fc_hz: float, order: int) -> Transfer:
        num, den = signal.butter(order, 2.0 * np.pi * fc_hz, btype="high", analog=True, output="ba")
        return TFMath.tf(num, den)

    @staticmethod
    def analog_bessel_lowpass(fc_hz: float, order: int) -> Transfer:
        num, den = signal.bessel(order, 2.0 * np.pi * fc_hz, btype="low", analog=True, norm="mag", output="ba")
        return TFMath.tf(num, den)

    @staticmethod
    def analog_bessel_highpass(fc_hz: float, order: int) -> Transfer:
        num, den = signal.bessel(order, 2.0 * np.pi * fc_hz, btype="high", analog=True, norm="mag", output="ba")
        return TFMath.tf(num, den)

    @staticmethod
    def linkwitz_riley_lowpass(fc_hz: float, order: int) -> Transfer:
        if order % 2 != 0:
            raise ValueError("Linkwitz-Riley order must be even")
        half_order = order // 2
        butter = TFMath.analog_butter_lowpass(fc_hz, half_order)
        return TFMath.series(butter, butter)

    @staticmethod
    def pade_delay(delay_s: float, order: int = 3) -> Transfer:
        if delay_s <= 0.0:
            return TFMath.gain(1.0)
        coeffs = np.array([(-delay_s) ** index / math.factorial(index) for index in range(2 * order + 1)], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            num_poly, den_poly = interpolate.pade(coeffs, order)
        return TFMath.tf(num_poly.c, den_poly.c)


def frequency_response(system: Transfer, freq_hz: Sequence[float]) -> np.ndarray:
    omega = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    _, response = signal.freqresp(system, omega)
    return response


def normalized_magnitude_db(system: Transfer, freq_hz: Sequence[float], normalize_at_hz: float) -> np.ndarray:
    response = frequency_response(system, freq_hz)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-20))
    reference = float(np.interp(np.log10(normalize_at_hz), np.log10(np.asarray(freq_hz, dtype=float)), magnitude_db))
    return magnitude_db - reference


def poles(system: Transfer) -> np.ndarray:
    return np.roots(TFMath.poly1(system.den))


def zeros(system: Transfer) -> np.ndarray:
    return np.roots(TFMath.poly1(system.num))


def phase_margin_and_gain_margin(system: Transfer, freq_hz: Sequence[float]) -> tuple[list[float], list[float], list[float], list[float]]:
    frequency = np.asarray(freq_hz, dtype=float)
    response = frequency_response(system, frequency)
    magnitude = np.abs(response)
    phase = np.unwrap(np.angle(response)) * 180.0 / np.pi
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-20))

    unity_crossings: list[float] = []
    phase_margins: list[float] = []
    phase_crossings: list[float] = []
    gain_margins: list[float] = []

    unity_indices = np.where(np.diff(np.sign(magnitude - 1.0)) != 0)[0]
    for index in unity_indices:
        x1, x2 = np.log10(frequency[index]), np.log10(frequency[index + 1])
        y1, y2 = np.log10(magnitude[index]), np.log10(magnitude[index + 1])
        xc = x1 if y1 == y2 else x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
        fc = 10.0**xc
        phase_here = float(np.interp(xc, [x1, x2], [phase[index], phase[index + 1]]))
        unity_crossings.append(fc)
        phase_margins.append(180.0 + phase_here)

    phase_indices = np.where(np.diff(np.sign(phase + 180.0)) != 0)[0]
    for index in phase_indices:
        x1, x2 = np.log10(frequency[index]), np.log10(frequency[index + 1])
        p1, p2 = phase[index], phase[index + 1]
        xc = x1 if p1 == p2 else x1 + (-180.0 - p1) * (x2 - x1) / (p2 - p1)
        fc = 10.0**xc
        magnitude_here = float(np.interp(xc, [x1, x2], [magnitude_db[index], magnitude_db[index + 1]]))
        phase_crossings.append(fc)
        gain_margins.append(-magnitude_here)

    return unity_crossings, phase_margins, phase_crossings, gain_margins
