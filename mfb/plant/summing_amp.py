from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class SummingAmplifierParams:
    reference_gain: float = 1.0
    feedback_gain: float = 1.0
    bias_v: float = 0.0
    rail_pos_v: float = 12.0
    rail_neg_v: float = -12.0


class SummingAmplifierModel:
    """u_sum(t) = k_r*u_ref(t) - k_f*u_fb(t) + V_bias with rail clipping."""

    def __init__(self, params: SummingAmplifierParams) -> None:
        self.params = params

    def mix(self, u_ref: np.ndarray, u_fb: np.ndarray) -> np.ndarray:
        u_sum = self.params.reference_gain * u_ref - self.params.feedback_gain * u_fb + self.params.bias_v
        return np.clip(u_sum, self.params.rail_neg_v, self.params.rail_pos_v)
