"""SimulationApp bootstrap.

This is the only module that may be imported *before* Isaac Sim exists, so all
``omni``/``isaacsim`` imports happen inside functions.
"""

from __future__ import annotations

from typing import Any, Sequence

from simlab.utils.logging import get_logger

log = get_logger("app")

#: omni.anim.people needs its whole animation stack up before the stage is built.
PEOPLE_EXTENSIONS: Sequence[str] = (
    "omni.anim.people",
    "omni.anim.navigation.bundle",
    "omni.anim.timeline",
    "omni.anim.graph.bundle",
    "omni.anim.graph.core",
    "omni.anim.graph.ui",
    "omni.anim.retarget.bundle",
    "omni.anim.retarget.core",
    "omni.anim.retarget.ui",
    "omni.kit.scripting",
)


#: Enabled on top of PEOPLE_EXTENSIONS when the ROS 2 bridge is on.
ROS2_EXTENSIONS: Sequence[str] = (
    "isaacsim.ros2.bridge",
    "isaacsim.robot.wheeled_robots",
    "omni.graph.action",
    "omni.graph.nodes",
)


def launch(headless: bool = False) -> Any:
    """Start Kit. Nothing from ``omni``/``isaacsim`` may be imported before this."""
    from isaacsim import SimulationApp

    return SimulationApp({"headless": headless})


def enable_extensions(app: Any, extensions: Sequence[str] = PEOPLE_EXTENSIONS) -> None:
    from isaacsim.core.utils.extensions import enable_extension

    for extension in extensions:
        enable_extension(extension)
        app.update()
    log(f"enabled {len(extensions)} extensions")


def fresh_stage(app: Any, updates: int = 3) -> None:
    """Re-create the stage so the animation runtime sees a stage-attach event.

    ``omni.anim.graph.core`` initializes its CharacterManager on stage attach.
    The stage Kit creates at startup is attached *before* the animation
    extensions are enabled, so without this the manager never initializes,
    ``get_character()`` returns nothing and every character stands still --
    with no error beyond a "Shutdown() called without a prior successful call
    to Initialize()" warning at exit.
    """
    import omni.usd

    omni.usd.get_context().new_stage()
    for _ in range(updates):
        app.update()
