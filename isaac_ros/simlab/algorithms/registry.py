"""Name -> controller lookup, so configs can select an algorithm by string."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Type

from simlab.algorithms.avoidance import SocialForceController
from simlab.algorithms.base import DriveController
from simlab.algorithms.patrol import SquareLoopController, StandStillController

CONTROLLERS: Dict[str, Type[DriveController]] = {
    SquareLoopController.name: SquareLoopController,
    StandStillController.name: StandStillController,
    SocialForceController.name: SocialForceController,
}


def build_controller(name: str, params: Mapping[str, Any] | None = None) -> DriveController:
    """Instantiate a registered controller.

    Raises:
        ValueError: if ``name`` is unknown or ``params`` has an unexpected key.
    """
    try:
        cls = CONTROLLERS[name]
    except KeyError:
        raise ValueError(
            f"unknown controller {name!r}; available: {sorted(CONTROLLERS)}"
        ) from None
    try:
        return cls(**dict(params or {}))
    except TypeError as exc:
        raise ValueError(f"bad params for controller {name!r}: {exc}") from None
