from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceParams:
    amplitude_v: float = 1.0
