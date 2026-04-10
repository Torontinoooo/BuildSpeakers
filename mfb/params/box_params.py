"""Box volume settings used for comparison studies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoxParams:
    """Sealed-box settings in liters."""

    small_box_l: float = 3.0
    expected_box_l: float = 27.5
