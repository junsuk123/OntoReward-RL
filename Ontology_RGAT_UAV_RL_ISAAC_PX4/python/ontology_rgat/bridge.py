"""Versioned UDP client for the external PX4 gateway.

Port of the retired ``matlab/src/+bridge/PX4Bridge.m``. The wire format is
unchanged, so the gateway, ``tools/fake_gateway.py`` and ``tools/protocol_probe.py``
did not have to move with it.
"""
from __future__ import annotations

import json
import socket
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .config import Config

__all__ = ["PX4Bridge", "BridgeError", "GatewayRejected", "GatewayTimeout",
           "pacing_anchor_us"]


def pacing_anchor_us(deadline_us: int, observed_us: int) -> int:
    """Re-anchor a fixed cadence when the consumer has already fallen behind."""
    return max(int(deadline_us), int(observed_us))


class BridgeError(RuntimeError):
    """Anything the gateway link can fail with."""


class GatewayRejected(BridgeError):
    """The gateway answered with an explicit error."""


class GatewayTimeout(BridgeError):
    """The gateway did not answer inside the configured budget."""


REQUIRED_STATE_FIELDS = (
    "position", "velocity", "quaternion_wxyz", "angular_velocity",
    "acceleration", "wind", "aero_force", "marker_quality", "estimator_valid",
    "pad", "battery", "gnss",
)
NUMERIC_STATE_FIELDS = (
    "position", "velocity", "quaternion_wxyz", "angular_velocity",
    "acceleration", "wind", "aero_force", "marker_quality",
)


class PX4Bridge:
    """One episode's link to the gateway.

    Owns a UDP port, so it must be closed: a sweep that builds one bridge per
    episode and leaks them runs out of the local port on the second episode.
    Use it as a context manager, or call :meth:`close`.
    """

    def __init__(self, cfg: Config):
        if not cfg.get("external", {}).get("enabled", False):
            raise BridgeError("External simulation configuration is required.")
        self.cfg = cfg.external
        self.expected = {
            "collective_span": cfg.rl.collective_span,
            "max_roll_pitch_rad": cfg.rl.max_roll_pitch,
            "max_yaw_rate_rad_s": cfg.rl.max_yaw_rate,
        }
        self.control_period_us = int(round(cfg.sim.dt * 1e6))
        self.sequence = 0
        self.last_state: dict[str, Any] = {}
        self.last_px4_time_us: int | None = None
        self.last_reset_ack: dict[str, Any] = {}
        self._closed = False

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.cfg.local_host, int(self.cfg.local_port)))
        self.socket.settimeout(0.002)
        self._drain()
        hello = self.transact("hello", {}, ("state",))
        self.assert_control_mapping(hello)

    # ------------------------------------------------------------- lifecycle
    def __enter__(self) -> "PX4Bridge":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if str(self.cfg.target).lower() == "sitl":
                self.disarm()
        except BridgeError:
            pass
        try:
            self.socket.close()
        except OSError:
            pass

    def __del__(self) -> None:                # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:
            pass

    # -------------------------------------------------------------- protocol
    def _drain(self) -> None:
        while True:
            try:
                self.socket.recv(65535)
            except (socket.timeout, BlockingIOError, OSError):
                return

    def transact(self, kind: str, fields: dict[str, Any],
                 expected: Sequence[str]) -> dict[str, Any]:
        """Send one request and wait for the matching reply."""
        self.sequence += 1
        seq = self.sequence
        message = dict(fields)
        message.update({"v": int(self.cfg.protocol_version), "type": kind,
                        "seq": seq, "time_ns": time.time_ns()})
        payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.socket.sendto(payload, (self.cfg.gateway_host, int(self.cfg.gateway_port)))

        deadline = time.monotonic() + float(self.cfg.timeout)
        while time.monotonic() < deadline:
            try:
                raw = self.socket.recv(65535)
            except (socket.timeout, BlockingIOError):
                continue
            except OSError as exc:
                raise BridgeError(f"gateway socket failed: {exc}") from exc
            try:
                response = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if response.get("type") == "error":
                raise GatewayRejected(
                    f"PX4 gateway rejected request: {response.get('error')}")
            if response.get("ack_seq") == seq and response.get("type") in expected:
                return response
        raise GatewayTimeout(
            f"PX4 gateway timeout after {float(self.cfg.timeout):.2f} s "
            f"({kind} seq={seq}).")

    def assert_control_mapping(self, state: dict[str, Any]) -> None:
        """Refuse a gateway that scales actions differently.

        The policy emits normalised actions against ``cfg.rl.*``. If the
        gateway converts them with different limits the vehicle is silently
        under- or over-actuated and every result is invalid rather than merely
        degraded, so this is an error and not a warning.
        """
        mapping = (state.get("extra") or {}).get("control_mapping")
        if not isinstance(mapping, dict):
            print("WARNING: gateway does not report its control mapping; "
                  "cannot verify it.")
            return
        names = {"collective_span": "cfg.rl.collective_span",
                 "max_roll_pitch_rad": "cfg.rl.max_roll_pitch",
                 "max_yaw_rate_rad_s": "cfg.rl.max_yaw_rate"}
        for field, source in names.items():
            if field not in mapping:
                continue
            if abs(float(mapping[field]) - self.expected[field]) > 1e-6:
                raise BridgeError(
                    f"Gateway {field} is {float(mapping[field]):.4f} but {source} is "
                    f"{self.expected[field]:.4f}. Fix config/system.yaml so the "
                    "action scaling matches the policy.")

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int) -> dict[str, Any]:
        """Reseed the episode and hand back the first valid state.

        The entry pose is flown by PX4, never teleported: Pegasus cannot reset
        the PX4 estimator, so a jump would leave the policy reading a diverged
        EKF for the whole episode. It is an offset in the pad frame, and the
        gateway re-aims it at the live deck every control tick, so PX4 chases a
        moving entry point instead of holding a point the rover has already
        driven away from.
        """
        ack = self.transact("reset", {"seed": float(seed),
                                      "wind_scale": float(self.cfg.wind_scale),
                                      "pad_scale": float(self.cfg.pad_scale),
                                      "gnss_scale": float(self.cfg.gnss_scale)},
                            ("ack",))
        # Kept whether or not the entry pose is flown: it carries the seeded
        # entry offset and starting energy, which is what makes an episode
        # reproducible from its seed.
        self.last_reset_ack = ack
        time.sleep(float(self.cfg.reset_settle))
        state = self.wait_valid_state()
        if not self.cfg.auto_arm:
            return state
        entry = self.entry_pose(ack)
        self.transact("goto", {"position": list(entry["position"]),
                               "yaw": entry["yaw"], "frame": entry["frame"],
                               "hold_s": float(self.cfg.entry_timeout)}, ("ack",))
        # Let the setpoint stream establish offboard before arming.
        time.sleep(float(self.cfg.prestream_count) / float(self.cfg.control_hz))
        state = self.wait_at_entry(np.asarray(entry["position"], dtype=float))
        # The episode clock starts at handover, not at the reset.
        self.last_px4_time_us = int(state["px4_time_us"])
        return state

    def entry_pose(self, ack: dict[str, Any]) -> dict[str, Any]:
        """The seeded entry point, as an offset from the deck."""
        detail = ack.get("detail")
        if not isinstance(detail, dict):
            raise BridgeError("Reset acknowledgement carries no entry pose. "
                              "Restart Isaac with the current landing_world.py.")
        frame = str(self.cfg.entry_frame).lower()
        if frame == "pad":
            if "entry_offset_pad_m" not in detail:
                raise BridgeError(
                    "Reset acknowledgement carries no pad-relative entry offset. "
                    "Isaac is running a landing_world.py from before the pad was "
                    "put on a rover; restart it.")
            position = np.asarray(detail["entry_offset_pad_m"], dtype=float)
        else:
            if "entry_position_enu_m" not in detail:
                raise BridgeError("Reset acknowledgement carries no entry position.")
            position = np.asarray(detail["entry_position_enu_m"], dtype=float)
        if position.size != 3 or not np.isfinite(position).all():
            raise BridgeError("Reset acknowledgement carries a malformed entry position.")
        return {"position": position, "frame": frame,
                "yaw": float(detail.get("entry_yaw_enu_rad", 0.0))}

    def pad_in_view(self, state: dict[str, Any]) -> bool:
        """Is the deck actually in the camera frame right now?

        Every episode is meant to begin with the pad already seen, so the
        policy starts from a marker fix instead of opening on GNSS alone --
        which in this canyon is tens of metres out. Geometry alone cannot
        promise it: the entry pose is drawn inside the footprint, but the
        vehicle only has to hold that pose to within the handover tolerance,
        and the tolerance is the same size as the frame. So this asks the
        detector rather than the arithmetic.

        Off when the camera is not the pose source -- a run configured without
        vision has no frame to be in -- and it is a *gate*, not a measurement:
        the quality it reads is the same number the policy gets.
        """
        if not bool(self.cfg.require_pad_in_view):
            return True
        return float(state.get("marker_quality", 0.0)) > 0.0

    @staticmethod
    def entry_state(state: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Pad-relative pose and settling speed to decide handover on.

        The two halves deliberately come from different places, because they
        ask different questions.

        Both come from the simulator, because the climb is flown on the
        simulator's pose too (see the gateway's ``_goto_world_target``). A gate
        read from a different signal than the one being flown is the mistake
        this went through twice: judged on the measurement it timed out on the
        receiver's velocity error, and judged on truth while flown on the
        measurement it timed out on a position the vehicle had no way to reach.

        None of it is part of the experiment. It decides when an episode may
        begin and nothing else -- the policy, the reward and the log never see
        it, exactly as with re-seating the vehicle between episodes or grading
        it afterwards (``env.truth_state``). On hardware there is no truth
        block and this falls back to the measured state.
        """
        truth = state.get("truth") if isinstance(state.get("truth"), dict) else {}
        if truth.get("valid", False):
            position = np.asarray(truth["position"], dtype=float).reshape(-1)
            velocity = np.asarray(truth["velocity"], dtype=float).reshape(-1)
            if (position.size == 3 and velocity.size == 3
                    and np.isfinite(position).all() and np.isfinite(velocity).all()):
                return position, float(np.linalg.norm(velocity))
        return (np.asarray(state["position"], dtype=float),
                float(np.linalg.norm(state["velocity"])))

    def wait_at_entry(self, target: np.ndarray) -> dict[str, Any]:
        """Hand over only once PX4 holds the entry pose."""
        started = time.monotonic()
        settled_since: float | None = None
        state: dict[str, Any] | None = None
        last_arm = float("-inf")
        while time.monotonic() - started < float(self.cfg.entry_timeout):
            elapsed = time.monotonic() - started
            try:
                state = self.get_state()
                if not state["armed"] and elapsed - last_arm >= float(self.cfg.arm_retry):
                    # PX4 rejects arming in transient pre-flight states, so one
                    # request is not enough to start the climb.
                    last_arm = elapsed
                    self.transact("arm", {}, ("ack",))
            except BridgeError:
                # A brief estimator or link transient during the climb is not a
                # handover failure; only the deadline decides.
                settled_since = None
                time.sleep(0.05)
                continue
            here, speed = self.entry_state(state)
            at_target = (
                float(np.linalg.norm(here - target)) <= float(self.cfg.entry_tolerance)
                and speed <= float(self.cfg.entry_speed_tolerance)
                and self.pad_in_view(state))
            if not at_target:
                settled_since = None
            elif settled_since is None:
                settled_since = time.monotonic()
            elif time.monotonic() - settled_since >= float(self.cfg.entry_settle):
                return state
            time.sleep(0.02)
        if state is None:
            raise BridgeError("PX4 published no state while climbing to the entry pose.")
        here, speed = self.entry_state(state)
        raise BridgeError(
            f"PX4 did not hold the entry pose within {float(self.cfg.entry_timeout):.1f} s "
            f"(offset {float(np.linalg.norm(here - target)):.2f} m, "
            f"speed {speed:.2f} m/s, "
            f"marker quality {float(state.get('marker_quality', 0.0)):.2f}).")

    # ------------------------------------------------------------------- step
    def step(self, action: Iterable[float]) -> dict[str, Any]:
        a = np.asarray(list(action), dtype=float).reshape(-1)
        if a.size != 4 or not np.isfinite(a).all() or np.any(np.abs(a) > 1.0):
            raise BridgeError("Action must contain four finite values in [-1,1].")
        reply = self.transact("action", {"action": a.tolist()}, ("state",))
        state = self.pace_to_control_period(self.validate_state(reply))
        self.last_state = state
        return state

    def step_velocity(self, command: Iterable[float]) -> dict[str, Any]:
        """Send physical body-heading [vx, vy, vz, yaw-rate] to PX4."""
        value = np.asarray(list(command), dtype=float).reshape(-1)
        limits = np.array([10.0, 10.0, 5.0, np.deg2rad(180.0)])
        if (value.shape != (4,) or not np.isfinite(value).all()
                or np.any(np.abs(value) > limits)):
            raise BridgeError("Velocity command is malformed or outside safety bounds.")
        reply = self.transact(
            "velocity_action", {"command": value.tolist()}, ("state",))
        state = self.pace_to_control_period(self.validate_state(reply))
        self.last_state = state
        return state

    def pace_to_control_period(self, state: dict[str, Any]) -> dict[str, Any]:
        """Let one control period of *simulated* time pass.

        The gateway answers as soon as PX4 publishes odometry, which is several
        times faster than the control rate. Returning that sample straight away
        ran the policy far faster than ``cfg.sim.dt`` while the episode clock
        still advanced ``cfg.sim.dt`` per step, so an episode covered a fraction
        of its nominal duration and the vehicle ran out of steps before it could
        land. PX4's clock is the simulator's clock under lockstep, so pace on
        that and never on wall time.
        """
        if self.control_period_us <= 0 or "px4_time_us" not in state:
            return state
        if self.last_px4_time_us is None:
            self.last_px4_time_us = int(state["px4_time_us"])
            return state
        deadline = self.last_px4_time_us + self.control_period_us
        started = time.monotonic()
        while int(state["px4_time_us"]) < deadline:
            if time.monotonic() - started > float(self.cfg.timeout):
                advanced = (int(state["px4_time_us"]) - self.last_px4_time_us) / 1e3
                raise BridgeError(
                    f"PX4 simulated time advanced only {advanced:.1f} ms in "
                    f"{float(self.cfg.timeout):.2f} s of wall time; the simulator "
                    "has stalled.")
            state = self.validate_state(self.transact("state", {}, ("state",)))
        # Fixed cadence while on time, but re-anchor when an expensive monitor
        # or rendering callback let simulation advance past the deadline. If we
        # retain the missed deadline, subsequent actions run at odometry rate
        # until it catches up, under-counting a 20 s episode as only a few
        # seconds and starving PX4's setpoint stream again on the next callback.
        self.last_px4_time_us = pacing_anchor_us(deadline, int(state["px4_time_us"]))
        return state

    # ------------------------------------------------------------- commands
    def get_state(self) -> dict[str, Any]:
        state = self.validate_state(self.transact("state", {}, ("state",)))
        self.last_state = state
        return state

    def disarm(self) -> None:
        self.transact("disarm", {}, ("ack",))

    def stop_after_outcome(self, timeout: float | None = None) -> bool:
        """Stop control and confirm PX4 has landed before the next reset."""
        # A moving deck's physical contact latch is the authoritative landing
        # signal.  Keep it before disabling offboard: after the forced disarm,
        # PX4 lockstep can stop producing odometry before a newer
        # ``armed=false`` sample reaches the gateway.  Requiring that newer
        # sample downgraded a physically confirmed, safely disarmed landing to
        # ``unconfirmed_success`` even though PX4 logged the disarm.
        extra = ((self.last_state or {}).get("extra") or {})
        contact_confirmed = bool(
            (self.last_state or {}).get("landed", False)
            and extra.get("pad_contact", False)
            and extra.get("land_detector_authoritative", False))
        try:
            self.disable_offboard()
        except BridgeError:
            pass
        self.disarm()
        if contact_confirmed:
            return True
        deadline = time.monotonic() + float(
            timeout if timeout is not None else self.cfg.outcome_settle_timeout)
        while time.monotonic() < deadline:
            state = self.get_state()
            if bool(state.get("landed", False)) and not bool(state.get("armed", False)):
                return True
            time.sleep(0.05)
        return False

    def arm(self) -> None:
        self.transact("arm", {}, ("ack",))

    def enable_offboard(self) -> None:
        self.transact("enable_offboard", {}, ("ack",))

    def disable_offboard(self) -> None:
        self.transact("disable_offboard", {}, ("ack",))

    def wait_valid_state(self) -> dict[str, Any]:
        started = time.monotonic()
        last: BridgeError | None = None
        while time.monotonic() - started < float(self.cfg.estimator_warmup):
            try:
                return self.get_state()
            except BridgeError as exc:
                last = exc
                time.sleep(0.05)
        if last is None:
            raise BridgeError(
                f"PX4 estimator did not publish state within "
                f"{float(self.cfg.estimator_warmup):.1f} s.")
        raise last

    # ------------------------------------------------------------ validation
    @staticmethod
    def validate_state(state: dict[str, Any]) -> dict[str, Any]:
        for field in REQUIRED_STATE_FIELDS:
            if field not in state:
                raise BridgeError(
                    f"Gateway state is missing field {field}. A gateway from "
                    "before the moving pad, the energy budget and the urban "
                    "GNSS model cannot be used with this adapter.")
        # The target moves, so a gateway that still reports world-frame position
        # would silently be asking the policy to land on the origin.
        if "position_frame" not in state:
            raise BridgeError("Gateway does not declare its position frame. "
                              "Restart it from the current ros2_gateway.py.")
        if state["position_frame"] != "pad":
            raise BridgeError(f"Gateway reports {state['position_frame']}-frame "
                              "position; this adapter needs pad-frame.")
        if state.get("frame") != "ENU_FLU":
            raise BridgeError(f"Unsupported gateway frame: {state.get('frame')}")
        for field in NUMERIC_STATE_FIELDS:
            value = np.asarray(state[field], dtype=float).reshape(-1)
            if not np.isfinite(value).all():
                raise BridgeError(f"Gateway state contains non-finite {field}.")
        if not state["estimator_valid"]:
            raise BridgeError("PX4 estimator state is not valid yet.")
        out = dict(state)
        for field in ("position", "velocity", "quaternion_wxyz",
                      "angular_velocity", "acceleration", "wind", "aero_force"):
            out[field] = np.asarray(state[field], dtype=float).reshape(-1)
        out["marker_quality"] = float(state["marker_quality"])
        return out
