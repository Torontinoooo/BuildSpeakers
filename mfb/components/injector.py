from __future__ import annotations

from dataclasses import dataclass

from mfb.core.transfer import Transfer, gain
from mfb.params.injection import InjectionParams


@dataclass
class ErrorInjector:
    params: InjectionParams

    def transfer(self) -> Transfer:
        return gain(self.params.gain)
