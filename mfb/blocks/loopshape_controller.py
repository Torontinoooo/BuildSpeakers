"""Loop-shaping controller block."""

from dataclasses import dataclass

from mfb.core.transfer import Transfer, first_order_pole, make_tf, series, gain


@dataclass(frozen=True)
class LoopshapeParams:
    """Simple lead/lag and high-frequency roll-off configuration."""

    k: float = 200.0
    f_zero_hz: float = 2000.0
    f_pole_hz: float = 400.0
    f_hf1_hz: float = 100.0
    f_hf2_hz: float = 1400.0


def loopshape_controller_tf(params: LoopshapeParams) -> Transfer:
    """Return the educational loop-shaping controller from the study."""
    lf_shape = make_tf([1.0 / (2.0 * 3.141592653589793 * params.f_zero_hz), 1.0],
                       [1.0 / (2.0 * 3.141592653589793 * params.f_pole_hz), 1.0])
    return series(
        gain(params.k),
        lf_shape,
        lf_shape,
        first_order_pole(params.f_hf1_hz),
        first_order_pole(params.f_hf2_hz),
    )
