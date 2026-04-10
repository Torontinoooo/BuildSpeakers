from __future__ import annotations

from scipy import signal

from .transfer import Transfer


def to_state_space(system: Transfer) -> signal.StateSpace:
    """Convert Transfer to a continuous state-space model."""
    return system.tf.to_ss()
