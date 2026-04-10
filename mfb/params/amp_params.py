"""Amplifier parameter container."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AmpParams:
    """Flat-gain amplifier model."""

    gain_db: float = 26.3
