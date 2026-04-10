"""Minimal tuning helper showing stability for both controller families."""

from mfb.analysis.stability import stable
from mfb.systems.mfb_loop import build_mfb_loop


def run() -> dict[str, bool]:
    """Return stability flags for PID and loopshape defaults."""
    pid_sys = build_mfb_loop(3.0, controller_mode="pid")
    loop_sys = build_mfb_loop(3.0, controller_mode="loopshape")
    return {"pid_stable": stable(pid_sys.t), "loopshape_stable": stable(loop_sys.t)}
