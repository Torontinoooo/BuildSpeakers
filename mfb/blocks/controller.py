"""Controller chooser to keep system assembly simple."""

from mfb.blocks.loopshape_controller import LoopshapeParams, loopshape_controller_tf
from mfb.blocks.pid_controller import PIDParams, pid_controller_tf
from mfb.core.transfer import Transfer


def controller_tf(mode: str = "pid") -> Transfer:
    """Return the selected controller transfer function."""
    if mode == "pid":
        return pid_controller_tf(PIDParams())
    if mode == "loopshape":
        return loopshape_controller_tf(LoopshapeParams())
    raise ValueError(f"Unknown controller mode: {mode}")
