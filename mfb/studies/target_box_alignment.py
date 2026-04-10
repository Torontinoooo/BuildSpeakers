from __future__ import annotations

from mfb.params.speaker import SpeakerParams


def sealed_fc(volume_l: float, speaker: SpeakerParams | None = None) -> float:
    import math

    spk = speaker or SpeakerParams()
    return spk.fs * math.sqrt(1.0 + spk.vas_l / volume_l)


if __name__ == "__main__":
    print("fc@3L:", round(sealed_fc(3.0), 2), "Hz")
