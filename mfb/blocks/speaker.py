"""Speaker electromechanical transfer-function block."""

from mfb.blocks.box import total_compliance
from mfb.core.transfer import Transfer, make_tf
from mfb.params.speaker_params import SpeakerParams


def speaker_tf(spk: SpeakerParams, box_volume_l: float | None) -> Transfer:
    """Voltage-to-acceleration loudspeaker model for educational loop studies."""
    c_tot = total_compliance(spk, box_volume_l)
    num = [spk.bl_tm, 0.0, 0.0]
    den = [
        spk.le_h * spk.mms_kg,
        spk.le_h * spk.rms_n_s_per_m + spk.re_ohm * spk.mms_kg,
        spk.le_h / c_tot + spk.re_ohm * spk.rms_n_s_per_m + spk.bl_tm**2,
        spk.re_ohm / c_tot,
    ]
    return make_tf(num, den)
