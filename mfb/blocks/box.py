"""Box helpers for compliance and resonance shift."""

import numpy as np

from mfb.params.speaker_params import SpeakerParams


def total_compliance(spk: SpeakerParams, box_volume_l: float | None) -> float:
    """Return compliance for free-air or sealed-box operation."""
    if box_volume_l is None:
        return spk.cms_m_per_n
    alpha = spk.vas_l / box_volume_l
    return spk.cms_m_per_n / (1.0 + alpha)


def sealed_box_fc(spk: SpeakerParams, box_volume_l: float) -> float:
    """Classic sealed-box resonance approximation."""
    alpha = spk.vas_l / box_volume_l
    return spk.fs_hz * np.sqrt(1.0 + alpha)
