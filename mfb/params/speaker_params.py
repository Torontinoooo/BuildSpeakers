"""Speaker constants used by the educational MFB model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerParams:
    """Small-signal loudspeaker parameters."""

    re_ohm: float = 6.08
    le_h: float = 0.90e-3
    fs_hz: float = 31.0
    qms: float = 2.15
    mms_kg: float = 14.51e-3
    cms_m_per_n: float = 1.82e-3
    sd_m2: float = 104e-4
    bl_tm: float = 6.9
    vas_l: float = 27.5

    @property
    def rms_n_s_per_m(self) -> float:
        """Compute mechanical resistance from Fs, Mms, and Qms."""
        import numpy as np

        return 2.0 * np.pi * self.fs_hz * self.mms_kg / self.qms
