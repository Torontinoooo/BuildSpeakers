from __future__ import annotations

from dataclasses import dataclass
from scipy import signal
import numpy as np


@dataclass(frozen=True, slots=True)
class PrefilterParams:
    enabled: bool = False
    cutoff_hz: float = 20.0
    order: int = 2


class PrefilterModel:
    def __init__(self, params: PrefilterParams, sample_rate_hz: float) -> None:
        self.params = params
        self.sample_rate_hz = sample_rate_hz

    def process(self, u: np.ndarray) -> np.ndarray:
        if not self.params.enabled:
            return u
        sos = signal.butter(self.params.order, self.params.cutoff_hz, btype="highpass", fs=self.sample_rate_hz, output="sos")
        return signal.sosfilt(sos, u)
