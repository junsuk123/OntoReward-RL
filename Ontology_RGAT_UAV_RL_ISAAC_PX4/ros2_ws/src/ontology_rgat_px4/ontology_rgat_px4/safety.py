from __future__ import annotations

import os


HARDWARE_ARM_PHRASE = "I_ACCEPT_PROPELLER_RISK"


HARDWARE_OFFBOARD_PHRASE = "I_ACCEPT_FLIGHT_CONTROL"


class SafetyGate:
    def __init__(self, target: str, allow_arm: bool, allow_offboard: bool = False):
        if target not in {"sitl", "hardware"}:
            raise ValueError("target must be sitl or hardware")
        self.target = target
        self.allow_arm = bool(allow_arm)
        self.allow_offboard = bool(allow_offboard)

    def may_arm(self) -> bool:
        if not self.allow_arm:
            return False
        if self.target == "sitl":
            return True
        return os.environ.get("ONTOLOGY_RGAT_HARDWARE_ARM") == HARDWARE_ARM_PHRASE

    def require_arm(self) -> None:
        if not self.allow_arm:
            raise PermissionError("arming disabled; restart gateway with --allow-arm")
        if self.target == "hardware" and not self.may_arm():
            raise PermissionError(
                "hardware arming requires ONTOLOGY_RGAT_HARDWARE_ARM=" + HARDWARE_ARM_PHRASE
            )

    def require_reset(self) -> None:
        if self.target != "sitl":
            raise PermissionError("episode reset is unavailable for hardware targets")

    def require_autonomous_climb(self) -> None:
        """Gate the pre-episode climb flown by PX4's position controller.

        On a real vehicle the pilot flies the entry pose, so an unattended
        autonomous climb is refused outright rather than merely gated.
        """
        if self.target != "sitl":
            raise PermissionError(
                "autonomous climb is unavailable for hardware targets; "
                "the pilot flies the entry pose"
            )
        self.require_offboard()

    def may_enable_offboard(self) -> bool:
        if self.target == "sitl":
            return True
        return self.allow_offboard and os.environ.get(
            "ONTOLOGY_RGAT_HARDWARE_OFFBOARD"
        ) == HARDWARE_OFFBOARD_PHRASE

    def require_offboard(self) -> None:
        if not self.may_enable_offboard():
            raise PermissionError(
                "hardware offboard requires --allow-offboard and "
                "ONTOLOGY_RGAT_HARDWARE_OFFBOARD=" + HARDWARE_OFFBOARD_PHRASE
            )
