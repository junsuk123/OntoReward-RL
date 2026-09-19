#!/usr/bin/env python3
"""Whether the diagnostic camera overlay is worth the main loop's time.

Extracted from landing_world.py so it can be tested: that module imports Isaac
Sim at module scope and cannot be loaded outside the simulator, which is the
same reason px4_sitl_parameters.py lives beside it.

The overlay draws projected landmarks over the camera frame for an operator.
Nothing scientific reads it -- not the actor observation, the reward, the R-GAT
input, the labels, the resets or any metric -- but drawing it costs a
GRAY->BGR convert, a dozen OpenCV primitives, a BGR->RGB convert and a ~491 KB
``tobytes`` per camera per rendered frame, on the single thread that also steps
physics and renders.
"""

from __future__ import annotations


def overlay_wanted(enabled: bool, subscription_count: int | None) -> bool:
    """Decide whether to draw and publish the operator overlay this frame.

    ``enabled`` is ``vision.operator_overlay``: a hard off for a run being
    timed, where even an attached RViz should not be paid for.

    ``subscription_count`` is how many subscribers the overlay topic has, or
    ``None`` when the middleware cannot say. Unknown means draw it: losing the
    operator view silently is worse than paying for it, and the count is only
    unavailable on an rclpy too old to introspect.
    """
    if not enabled:
        return False
    if subscription_count is None:
        return True
    return int(subscription_count) > 0
