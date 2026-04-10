from __future__ import annotations

from dataclasses import dataclass

from .transfer import Transfer


@dataclass
class LTIComponent:
    """Base class for components that expose a transfer function."""

    name: str

    def transfer(self) -> Transfer:
        raise NotImplementedError
