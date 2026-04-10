"""Prefilter block placeholder.

This project currently uses a unity prefilter to keep the architecture explicit
without adding unnecessary complexity.
"""

from mfb.core.transfer import Transfer, gain


def prefilter_tf() -> Transfer:
    """Return unity prefilter (no shaping)."""
    return gain(1.0)
