from __future__ import annotations

import numpy as np

from mfb.core.transfer import Transfer, freq_response


def closed_loop_poles(system: Transfer) -> np.ndarray:
    return np.roots(system.den)


def max_real_part(system: Transfer) -> float:
    poles = closed_loop_poles(system)
    return float(np.max(np.real(poles)))


def stable(system: Transfer) -> bool:
    return max_real_part(system) < 0


def approximate_phase_margins(open_loop: Transfer, freq_hz: np.ndarray) -> list[tuple[float, float]]:
    h = freq_response(open_loop, freq_hz)
    mag = np.abs(h)
    ph = np.unwrap(np.angle(h)) * 180 / np.pi
    idx = np.where(np.diff(np.sign(mag - 1.0)) != 0)[0]
    out: list[tuple[float, float]] = []
    for i in idx:
        f1, f2 = freq_hz[i], freq_hz[i + 1]
        x1, x2 = np.log10(f1), np.log10(f2)
        y1, y2 = np.log10(mag[i]), np.log10(mag[i + 1])
        xc = x1 if y1 == y2 else x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
        phase = float(np.interp(xc, [x1, x2], [ph[i], ph[i + 1]]))
        out.append((10**xc, 180 + phase))
    return out
