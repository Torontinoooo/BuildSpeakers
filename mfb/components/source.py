from __future__ import annotations

from dataclasses import dataclass

from mfb.core.transfer import Transfer, gain
from mfb.params.source import SourceParams


@dataclass
class VoltageSource:
    params: SourceParams

    def transfer(self) -> Transfer:
        return gain(self.params.amplitude_v)
