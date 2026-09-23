"""Versioned UDP client for the external PX4 gateway.

Port of the retired ``matlab/src/+bridge/PX4Bridge.m``. The wire format is
unchanged, so the gateway, ``tools/fake_gateway.py`` and ``tools/protocol_probe.py``
did not have to move with it.
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .config import Config
from .initialization import pad_view_margin
from .mathx import quat_to_euler_zyx

__all__ = ["PX4Bridge", "BridgeError", "EntryResetError", "GatewayRejected",
           "GatewayTimeout", "PX4Failsafe", "PX4EstimatorInvalid",
           "pacing_anchor_us"]


def pacing_anchor_us(deadline_us: int, observed_us: int) -> int:
    """Re-anchor a fixed cadence when the consumer has already fallen behind."""
    return max(int(deadline_us), int(observed_us))


class BridgeError(RuntimeError):
    """Anything the gateway link can fail with."""


# MAV_CMD_COMPONENT_ARM_DISARM, and the two results that are not a refusal.
ARM_COMMAND = 400
COMMAND_ACCEPTED = (0, 5)


class EntryResetError(BridgeError):
    """The SITL vehicle could not safely establish the episode entry hover.

    This is an infrastructure/reset outcome, never an RL transition.  Keeping
    it distinct lets the outer collector cycle a freshly owned simulator after
    the environment's local retries are exhausted without mislabelling the
    failed climb as policy experience.
    """


class GatewayRejected(BridgeError):
    """The gateway answered with an explicit error."""


class GatewayTimeout(BridgeError):
    """The gateway did not answer inside the configured budget."""


class ArmingRefused(EntryResetError):
    """PX4's preflight checks refuse to arm the vehicle.

    A separate type because the remedy is different from every other entry
    failure: no number of resets makes a vehicle whose EKF reports a high
    accelerometer bias or an attitude failure pass preflight. It needs a fresh
    simulator, and a run that cannot provide one has to say so instead of
    spending its bounded retries proving the point.
    """


class PX4Failsafe(BridgeError):
    """PX4 entered a failsafe while an episode was being measured.

    The gateway classifies OFFBOARD heartbeat loss and its benign autonomous
    SITL status-clear race as recoverable transport faults. Battery, estimator,
    geofence and multi-cause failure-detector events remain hard failures so a
    retry cannot hide a vehicle or policy problem.

    A lone ``fd_critical_failure`` is neither, and is not raised at all: see
    ``validate_state``. It is a tip-over, which is the task failing, and the
    environment scores it as a crash.
    """

    def __init__(self, reasons: Sequence[str] = (), *, recoverable: bool = False):
        self.reasons = tuple(str(reason) for reason in reasons)
        self.recoverable = bool(recoverable)
        detail = ", ".join(self.reasons) if self.reasons else "unknown"
        super().__init__(
            "PX4 reports an active failsafe "
            f"({detail}); refusing to record this as an RL step.")


class SimulatorFrameTimeout(BridgeError):
    """The camera topic stopped delivering frames within the step budget.

    Infrastructure, not task failure. It is what a rebuilt stack looks like
    from the learner's side: the Isaac process that published the image topic
    was replaced, and DDS has not finished rediscovering the new publisher when
    the next step asks for a frame.

    It exists because that arrived as a bare ``TimeoutError`` from the ROS
    buffer, which ``collect_episode_resilient`` -- catching ``BridgeError`` --
    let straight through. On 2026-09-23 that ended a 128-episode two-arm run
    outright, three stack restarts into a recovery the retry budget had not
    even spent.
    """


class PX4EstimatorInvalid(BridgeError):
    """PX4 temporarily stopped publishing a control-valid local estimate."""


# ``marker_quality`` stays in the wire contract for the legacy ArUco profile
# and recorded fixtures. Nothing on the primary two-pipeline path reads it:
# field-of-view truth is geometric (LiveShinEnvironment.geometric_pad_center_in_fov)
# and perception health comes from the keypoint encoder.
REQUIRED_STATE_FIELDS = (
    "position", "velocity", "quaternion_wxyz", "angular_velocity",
    "acceleration", "wind", "aero_force", "marker_quality", "estimator_valid",
    "pad", "battery", "gnss",
)
NUMERIC_STATE_FIELDS = (
    "position", "velocity", "quaternion_wxyz", "angular_velocity",
    "acceleration", "wind", "aero_force", "marker_quality",
)


def _wrap_to_pi(angle: float) -> float:
    """Signed angle in (-pi, pi], so a 359 deg error reads as -1 deg."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quaternion_wxyz(quaternion) -> float | None:
    """ENU heading from a w-x-y-z quaternion, or None if it is unusable."""
    try:
        w, x, y, z = (float(v) for v in np.asarray(
            quaternion, dtype=float).reshape(-1)[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (w, x, y, z)):
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# Local UDP ports with a live bridge in this process. The socket is bound
# with SO_REUSEADDR, so a second bridge on the same pair binds without any
# error and the kernel then hands every gateway reply to one of the two; the
# other sees nothing but timeouts, and a learner reads those as a dead
# simulator and rebuilds the shared stack. Two workers on one pair is a
# scheduling bug, and it has to fail at the second bind, not as that storm.
_LIVE_LOCAL_PORTS: dict[tuple[str, int], int] = {}
_LIVE_LOCAL_PORTS_LOCK = threading.Lock()


# The gateway will hold an entry setpoint autonomously for at most this long
# (``ontology_rgat_px4.protocol.GOTO_MAX_HOLD_S``; the two are pinned equal by
# a test). The learner's own wall-clock guard is a different quantity and may
# legitimately be longer on a slowly rendered multi-pair stage -- this profile
# asks for 1200 s, and 1320 s once the parallel scale is applied -- but asking
# the gateway for more hold than the protocol admits is rejected outright.
#
# Unclamped, that rejection failed *every* reset: 2026-09-21, the first run to
# finish its warm start died on the next stage's first episode. It had been
# invisible until then only because the demonstration stage shortens the guard
# to 120 s for its own flights.
GATEWAY_MAX_HOLD_S = 900.0
_HOLD_CLAMP_REPORTED = False


def entry_hold_seconds(entry_timeout: float, hold_margin_s: float) -> float:
    """How long to ask the gateway to hold the entry pose, within protocol.

    Clamping rather than failing is deliberate: the hold only has to outlast a
    real entry, which takes tens of seconds, while the wall guard exists to
    bound a stopped simulator.
    """
    global _HOLD_CLAMP_REPORTED
    requested = float(entry_timeout) + float(hold_margin_s)
    if not math.isfinite(requested) or requested <= 0.0:
        raise BridgeError(
            f"entry hold must be positive and finite, not {requested}")
    if requested <= GATEWAY_MAX_HOLD_S:
        return requested
    if not _HOLD_CLAMP_REPORTED:
        _HOLD_CLAMP_REPORTED = True
        print(
            f"WARNING: the entry wall-clock guard ({float(entry_timeout):.0f} s) "
            f"exceeds the gateway's maximum autonomous hold "
            f"({GATEWAY_MAX_HOLD_S:.0f} s); asking for the maximum. The guard "
            "itself is unchanged.")
    return GATEWAY_MAX_HOLD_S


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
        self._port_key = (str(self.cfg.local_host), int(self.cfg.local_port))

        with _LIVE_LOCAL_PORTS_LOCK:
            if self._port_key in _LIVE_LOCAL_PORTS:
                self._closed = True
                pair = getattr(self.cfg, "pair_index", None)
                where = "" if pair is None else f" (pair {int(pair)})"
                raise BridgeError(
                    f"local UDP port {self._port_key[1]} already carries a live "
                    f"bridge in this process{where}: two workers are flying the "
                    "same UAV/UGV pair, and the gateway would answer only one of "
                    "them. Give each concurrent worker its own pair.")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.socket.bind((self.cfg.local_host, int(self.cfg.local_port)))
            except OSError:
                self._closed = True
                self.socket.close()
                raise
            _LIVE_LOCAL_PORTS[self._port_key] = id(self)
        self.socket.settimeout(0.002)
        self._drain()
        # First contact after a boot or a restart: Isaac is still loading its
        # assets and the gateway has nothing to answer with yet. The short
        # control budget would fail the reconnect here and spend one of the
        # worker's bounded recovery attempts on a simulator that is merely
        # still starting.
        try:
            hello = self.transact("hello", {}, ("state",),
                                  timeout=self._setup_timeout())
            self.assert_control_mapping(hello)
        except BaseException:
            # Release the local UDP port so a retry can bind it again, instead
            # of leaving that to whenever the collector reaps this instance.
            self._release_socket()
            raise

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
        self._release_socket()

    def _release_socket(self) -> None:
        """Close the UDP socket and give up this process's claim on its port."""
        self._closed = True
        try:
            self.socket.close()
        except OSError:
            pass
        with _LIVE_LOCAL_PORTS_LOCK:
            key = getattr(self, "_port_key", None)
            if key is not None and _LIVE_LOCAL_PORTS.get(key) == id(self):
                del _LIVE_LOCAL_PORTS[key]

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

    def _setup_timeout(self) -> float:
        """Budget for a request that may be waiting on a simulator boot.

        Isaac takes minutes to load the city, the rover and the vehicle before
        the gateway can answer anything. Never shorter than the control
        budget, so a configuration cannot make setup stricter than flight.
        """
        # Looked up lazily: focused protocol tests and hardware adapters build
        # partial configurations, and asking for a budget must not be what
        # raises on them.
        control = float(getattr(self.cfg, "timeout", 2.0))
        configured = getattr(self.cfg, "setup_timeout", None)
        return control if configured is None else max(control, float(configured))

    def transact(self, kind: str, fields: dict[str, Any],
                 expected: Sequence[str], *,
                 timeout: float | None = None) -> dict[str, Any]:
        """Send one request and wait for the matching reply.

        ``timeout`` overrides the per-request budget for setup traffic that
        has to outlast a simulator boot. The default stays the short control
        budget: a step that waits minutes for a reply is a stalled simulator
        going unnoticed, and a gap that long inside an episode is not a
        trajectory PPO may learn from.
        """
        self.sequence += 1
        seq = self.sequence
        message = dict(fields)
        message.update({"v": int(self.cfg.protocol_version), "type": kind,
                        "seq": seq, "time_ns": time.time_ns()})
        payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.socket.sendto(payload, (self.cfg.gateway_host, int(self.cfg.gateway_port)))

        budget = float(self.cfg.timeout if timeout is None else timeout)
        deadline = time.monotonic() + budget
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
            f"PX4 gateway timeout after {budget:.2f} s ({kind} seq={seq}).")

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
    def reset(self, seed: int, scenario: str = "training_random_walk", *,
              initial_condition_scale: float | None = None) -> dict[str, Any]:
        """Reseed the episode and hand back the first valid state.

        The entry pose is flown by PX4, never teleported: Pegasus cannot reset
        the PX4 estimator, so a jump would leave the policy reading a diverged
        EKF for the whole episode. It is an offset in the pad frame, and the
        gateway re-aims it at the live deck every control tick, so PX4 chases a
        moving entry point instead of holding a point the rover has already
        driven away from.
        """
        reset_fields = {"seed": float(seed),
                        "wind_scale": float(self.cfg.wind_scale),
                        "pad_scale": float(self.cfg.pad_scale),
                        "gnss_scale": float(self.cfg.gnss_scale),
                        "scenario": str(scenario)}
        if initial_condition_scale is not None:
            scale = float(initial_condition_scale)
            if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
                raise BridgeError("initial-condition curriculum must be in [0, 1]")
            reset_fields["initial_condition_scale"] = scale
        ack = self.transact("reset", reset_fields, ("ack",),
                            timeout=self._setup_timeout())
        # Kept whether or not the entry pose is flown: it carries the seeded
        # entry offset and starting energy, which is what makes an episode
        # reproducible from its seed.
        self.last_reset_ack = ack
        time.sleep(float(self.cfg.reset_settle))
        state = self.wait_valid_state()
        if not self.cfg.auto_arm:
            return state
        entry = self.entry_pose(ack)
        # ``wait_at_entry`` starts only after the OFFBOARD pre-stream below.
        # Give the gateway's autonomous hold a little more time than that full
        # client-side window; otherwise it enters AUTO.LAND first and the final
        # diagnostic misleadingly reports the resulting descent as an unstable
        # hover.  This matters in a shared two-camera world whose wall clock
        # advances more slowly than a single-pair simulation.
        prestream_s = (float(self.cfg.prestream_count)
                       / float(self.cfg.control_hz))
        hold_margin_s = max(2.0, prestream_s + float(self.cfg.reset_settle))
        hold_s = entry_hold_seconds(self.cfg.entry_timeout, hold_margin_s)

        def send_goto(timeout: float | None = None) -> None:
            self.transact("goto", {"position": list(entry["position"]),
                                   "yaw": entry["yaw"], "frame": entry["frame"],
                                   "hold_s": hold_s}, ("ack",),
                          timeout=timeout)

        # The first setpoint may still be waiting on a booting stack. The
        # re-aims below happen inside the entry deadline and keep the short
        # budget, so one slow reply cannot eat the whole entry window.
        send_goto(self._setup_timeout())
        # Let the setpoint stream establish offboard before arming.
        time.sleep(prestream_s)

        def reissue(attempt: int) -> None:
            """Re-send the seeded entry setpoint, unchanged.

            Used only when the vehicle is already holding the commanded offset
            and speed but the deck is not in frame. The pose itself is not
            re-drawn: that offset is part of the episode's seeded initial
            condition and both arms have to receive the same one.
            """
            print(f"Entry hover: deck out of frame at the commanded offset; "
                  f"re-aiming the setpoint ({attempt}).")
            send_goto()

        state = self.wait_at_entry(np.asarray(entry["position"], dtype=float),
                                   reissue=reissue,
                                   target_yaw=float(entry["yaw"]))
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

    def entry_view_margin(self, state: dict[str, Any],
                          here: np.ndarray) -> float | None:
        """Normalised image position of the pad centre, from simulator geometry.

        Below 1 the pad centre is inside the landing camera's frame; ``None``
        when the geometry cannot be evaluated (no attitude in the sample, or a
        world-frame entry whose pad-relative pose is unknown). The camera model
        is ``cfg.external.landing_camera``, copied from the same
        ``vision.camera`` block Isaac builds the rendered camera from.

        This is the *one* definition of initial pad visibility used by the
        benchmark, and it is initialization-only simulator truth: it decides
        when an episode may begin and nothing else. The PPO actor never
        receives it.
        """
        if str(getattr(self.cfg, "entry_frame", "pad")).lower() != "pad":
            return None
        quaternion = state.get("quaternion_wxyz")
        if quaternion is None:
            return None
        camera = getattr(self.cfg, "landing_camera", None)
        params = dict(camera) if isinstance(camera, dict) else {}
        try:
            return float(pad_view_margin(
                np.asarray(here, dtype=float), quaternion,
                image_size=params.get("resolution", (512, 320)),
                horizontal_fov_deg=float(params.get("horizontal_fov_deg", 90.0)),
                pitch_down_deg=float(params.get("pitch_down_deg", 60.0)),
                mount_translation_flu_m=params.get(
                    "mount_translation_flu_m", (0.0, 0.0, -0.16))))
        except (TypeError, ValueError):
            return None

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

    def wait_at_entry(self, target: np.ndarray, *,
                      reissue=None, target_yaw: float | None = None) -> dict[str, Any]:
        """Hand over only once PX4 holds the entry pose.

        The outer timeout is deliberately wall-clock bounded so a stalled
        simulator cannot hang the runner. Settling is a physical-duration
        requirement, however, and therefore follows PX4's lockstep simulation
        clock. On a rendered multi-pair stage one second of simulation can
        take many wall seconds.

        "Pad in view" has exactly one meaning here: the simulator's own
        geometry places the pad centre inside the landing camera's frustum
        (``entry_view_margin``). The previous "recent ArUco detection OR
        geometry" gate mixed two incompatible definitions of visibility into
        one experiment, and from the upper half of the Table-I entry altitude
        range a small tag cannot be decoded at all even though the pad fills
        the centre of the image. A detector is in any case no longer part of
        the primary perception path.
        """
        started = time.monotonic()
        # Two clocks, two jobs.
        #
        # The budget the vehicle is judged against is *simulated* PX4 time,
        # because the settle streak below is. A wall-clock budget silently
        # shrinks the manoeuvre the gate asks for as the stage gets heavier:
        # on this two-camera city stage one simulated second costs several
        # wall seconds, so the same 99 s that was ample on a flat plane cut
        # arriving vehicles off mid-settle -- reported as "longest hold 0.98 s
        # of 1.00 s" with every tolerance met at the final sample. That is a
        # deadline, not an unstable hover, and restarting the simulator for it
        # throws away the other pair's episode as well.
        #
        # The wall clock stays only as a hang guard, for a simulator that has
        # stopped publishing time at all and would otherwise never return.
        sim_budget = float(getattr(self.cfg, "entry_sim_budget", float("inf")))
        wall_guard = float(self.cfg.entry_timeout)
        # Paid for once, from the first sample: the distance the vehicle still
        # has to cover before it can hold anything. See ``entry_travel_speed``.
        travel_allowance: float | None = None
        initial_offset = float("nan")
        failsafe_grace = float(getattr(self.cfg, "failsafe_grace", 10.0))
        failsafe_since: float | None = None
        failsafe_seconds = 0.0
        failsafe_reasons: tuple[str, ...] = ()
        sim_started: float | None = None
        sim_elapsed = 0.0
        expiry = "wall"
        settled_since: float | None = None
        view_margin: float | None = None
        state: dict[str, Any] | None = None
        last_arm = float("-inf")
        was_armed = False
        # The final sample alone is a misleading diagnosis: a vehicle whose
        # limit cycle straddles a tolerance can be inside every bound at the
        # instant the deadline expires and still never have held the streak.
        # Record what actually blocked the streak instead.
        samples = 0
        blocked = {"offset": 0, "speed": 0, "view": 0}
        worst = {"offset": 0.0, "speed": 0.0}
        longest_streak = 0.0
        # A vehicle that is neither closing the gap nor moving at all is not
        # settling slowly; it is not flying. Measured 2026-09-23: a pair sat on
        # the deck at 0.00 m/s, 5.13 m from a commanded pose 4.50 m above it,
        # and spent 10515 of 10515 samples -- 271 s of wall clock -- closing
        # -0.00 m of it before the budget expired and the stack was rebuilt
        # anyway. The rebuild was the only remedy either way, so the budget
        # bought nothing but delay.
        #
        # Bail as soon as that is unambiguous. ``entry_stall_speed`` is what
        # makes it safe to do: a vehicle that is chasing, overshooting or
        # circling has a relative speed far above it and can never trip this,
        # however long it takes to settle. Only a parked one does.
        best_offset = float("inf")
        stalled_since: float | None = None
        stalled_peak_speed = 0.0
        stall_seconds = float(getattr(self.cfg, "entry_stall_seconds", 20.0))
        stall_speed = float(getattr(self.cfg, "entry_stall_speed", 0.05))
        stall_progress = float(getattr(self.cfg, "entry_stall_progress_m", 0.25))
        # PX4 publishes the result of the last command it processed. A vehicle
        # that never arms cannot reach any pose, and reporting its frozen
        # position offset instead of the refusal sends the operator after the
        # wrong thing entirely.
        arm_refusal: tuple[int, int] | None = None
        # Bounded on its own terms: entry_timeout is now a wall-clock hang
        # guard, and a third of it would push the arming diagnosis minutes out.
        arm_grace = float(getattr(
            self.cfg, "entry_arm_grace",
            min(60.0, max(20.0, float(self.cfg.entry_timeout) / 3.0))))
        # Bounded re-aim for the case the vehicle is holding station correctly
        # but the deck is not in frame. Setup only, and identical for both arms.
        view_retries = 0
        view_retry_limit = int(getattr(self.cfg, "entry_view_retries", 2))
        view_retry_after = max(4.0, float(getattr(self.cfg, "entry_settle", 0.5)) * 4.0)
        out_of_view_since: float | None = None
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= wall_guard:
                expiry = "wall"
                break
            if sim_elapsed >= sim_budget:
                expiry = "simulated"
                break
            try:
                state = self.get_state()
                armed = bool(state["armed"])
                extra = state.get("extra") if isinstance(
                    state.get("extra"), dict) else {}
                # A deck strike during the unmeasured entry climb force-disarms
                # PX4.  The old loop then requested ARM every two seconds for
                # the rest of the 90 s deadline, turning one failed setup into
                # a velocity-estimator/failsafe storm.  It cannot become a
                # valid entry hover without rebuilding SITL, so fail fast and
                # let the owner restart it.  No transition has been collected.
                last_command = extra.get("last_command")
                if (isinstance(last_command, (list, tuple))
                        and len(last_command) == 2
                        and int(last_command[0]) == ARM_COMMAND
                        and int(last_command[1]) not in COMMAND_ACCEPTED):
                    arm_refusal = (int(last_command[0]), int(last_command[1]))
                elif armed:
                    arm_refusal = None
                if bool(extra.get("pad_contact", False)):
                    raise EntryResetError(
                        "PX4 contacted the pad while establishing the entry "
                        "hover; restarting SITL before collecting RL data.")
                if was_armed and not armed:
                    raise EntryResetError(
                        "PX4 disarmed while establishing the entry hover; "
                        "restarting SITL before collecting RL data.")
                was_armed = bool(was_armed or armed)
                if (not was_armed and arm_refusal is not None
                        and elapsed >= arm_grace):
                    # Waiting out the rest of the budget cannot change this:
                    # PX4 has been refusing to arm for the whole grace window.
                    # A degraded SITL instance needs a new simulator, not more
                    # seconds, and eight silent retries of 99 s each is a
                    # quarter of an hour spent proving that.
                    raise ArmingRefused(
                        "PX4 refused to arm for "
                        f"{elapsed:.0f} s (command {arm_refusal[0]} result "
                        f"{arm_refusal[1]}); the vehicle never left the "
                        "ground, so no entry pose is reachable. Check the "
                        "simulator log for 'Preflight Fail': a degraded SITL "
                        "EKF (high accelerometer bias, attitude failure) "
                        "needs a fresh simulator, not another reset.")
                if not armed and elapsed - last_arm >= float(self.cfg.arm_retry):
                    # PX4 rejects arming in transient pre-flight states, so one
                    # request is not enough to start the climb.
                    last_arm = elapsed
                    self.transact("arm", {}, ("ack",))
            except EntryResetError:
                # A latched contact or disarm is not a transient missing
                # sample. Waiting here only preserves a bad estimator; the
                # owned-stack recovery path must rebuild it.
                raise
            except PX4Failsafe as failsafe:
                # The gateway classifies an OFFBOARD heartbeat loss and its
                # benign status-clear race as recoverable transport faults,
                # and they clear in well under a second -- the gateway keeps
                # streaming setpoints and re-requests OFFBOARD as soon as the
                # flag drops. PX4 ignores the entry setpoints while it is set,
                # which is why the gate then reports a *stationary* vehicle
                # off target. Rebuilding the shared simulator for that costs
                # minutes and both pairs' progress, so wait it out first. A
                # hard failsafe (battery, estimator, geofence, failure
                # detector) still aborts immediately.
                if not failsafe.recoverable:
                    raise
                now = time.monotonic()
                if failsafe_since is None:
                    failsafe_since = now
                    failsafe_reasons = failsafe.reasons
                    print("Entry hover: waiting out a recoverable SITL link "
                          f"failsafe ({', '.join(failsafe.reasons) or 'unknown'}).")
                if now - failsafe_since >= failsafe_grace:
                    raise
                settled_since = None
                time.sleep(0.05)
                continue
            except BridgeError:
                # A brief estimator or link transient during the climb is not a
                # handover failure; only the deadline decides.
                settled_since = None
                time.sleep(0.05)
                continue
            if failsafe_since is not None:
                failsafe_seconds += time.monotonic() - failsafe_since
                failsafe_since = None
            here, speed = self.entry_state(state)
            wall_now = time.monotonic()
            try:
                sample_now = float(state["px4_time_us"]) * 1e-6
                if not np.isfinite(sample_now):
                    raise ValueError("non-finite PX4 timestamp")
            except (KeyError, TypeError, ValueError, OverflowError):
                # Hardware adapters and focused protocol tests may not expose
                # a PX4 timestamp. They retain the historical wall-clock
                # behaviour instead of losing the readiness gate entirely.
                sample_now = wall_now
            if sim_started is None:
                sim_started = sample_now
            # PX4's clock can be re-anchored by the timesync filter; a jump
            # backwards must not read as a budget that has already expired.
            if sample_now < sim_started:
                sim_started = sample_now
            sim_elapsed = float(sample_now - sim_started)
            view_margin = self.entry_view_margin(state, here)
            geometry_ready = (
                view_margin is not None
                and view_margin <= float(getattr(
                    self.cfg, "entry_view_margin", 0.85)))
            pad_ready = (
                not bool(self.cfg.require_pad_in_view) or geometry_ready)
            offset = float(np.linalg.norm(here - target))
            if travel_allowance is None:
                # A fixed budget pays for the manoeuvre, not for the trip. The
                # previous episode can leave the vehicle tens of metres from
                # the deck, and at PX4's own 2 m/s limit that is most of the
                # budget before the hold can even begin.
                initial_offset = offset
                cruise = max(0.05, float(getattr(
                    self.cfg, "entry_travel_speed", 1.2)))
                travel_allowance = min(
                    max(0.0, offset - float(self.cfg.entry_tolerance)) / cruise,
                    float(getattr(self.cfg, "entry_travel_budget_max", 60.0)))
                if travel_allowance > 1.0:
                    print(f"Entry hover: {offset:.1f} m to fly; allowing "
                          f"{travel_allowance:.0f} simulated s of travel on top "
                          f"of the {sim_budget:.0f} s hold budget.")
                sim_budget += travel_allowance
            offset_ready = offset <= float(self.cfg.entry_tolerance)
            speed_ready = speed <= float(self.cfg.entry_speed_tolerance)
            samples += 1
            worst["offset"] = max(worst["offset"], offset)
            worst["speed"] = max(worst["speed"], float(speed))
            if not offset_ready:
                blocked["offset"] += 1
            if not speed_ready:
                blocked["speed"] += 1
            if not pad_ready:
                blocked["view"] += 1
            # Progress, or the complete absence of it. Improvement resets the
            # window, so a vehicle that is slowly closing the gap is never cut
            # off; only one that has stopped both approaching and moving is.
            #
            # The window does not even open while the budget is still paying
            # for the trip. The travel allowance is this run's own statement
            # that the vehicle is expected to be under way rather than holding,
            # so a standstill inside it is not yet evidence of anything -- and
            # opening the window there would have it expire on arrival, which
            # is the moment the hold is supposed to begin.
            travelling = sim_elapsed < float(travel_allowance or 0.0)
            if offset < best_offset - stall_progress or travelling:
                best_offset = min(best_offset, offset)
                stalled_since = None
                stalled_peak_speed = 0.0
            elif not offset_ready:
                if stalled_since is None:
                    stalled_since = sample_now
                    stalled_peak_speed = float(speed)
                else:
                    stalled_peak_speed = max(stalled_peak_speed, float(speed))
                if (stall_seconds > 0.0
                        and sample_now - stalled_since >= stall_seconds
                        and stalled_peak_speed <= stall_speed):
                    raise EntryResetError(
                        f"PX4 stopped flying to the entry pose: {offset:.2f} m "
                        f"out and no closer for {sample_now - stalled_since:.0f} "
                        f"simulated s, with a peak speed of "
                        f"{stalled_peak_speed:.2f} m/s against a "
                        f"{stall_speed:.2f} m/s standstill threshold. The "
                        "vehicle is parked, not settling, so the remaining "
                        "entry budget cannot change the outcome; rebuilding "
                        "the simulator is the only remedy and is started now "
                        "instead of after it expires.")
            at_target = offset_ready and speed_ready and pad_ready
            if offset_ready and speed_ready and not pad_ready:
                # Stable at the commanded offset with the deck out of frame.
                if out_of_view_since is None:
                    out_of_view_since = sample_now
                elif (reissue is not None
                        and view_retries < view_retry_limit
                        and sample_now - out_of_view_since >= view_retry_after):
                    view_retries += 1
                    out_of_view_since = None
                    reissue(view_retries)
            else:
                out_of_view_since = None
            if not at_target:
                settled_since = None
            elif settled_since is None:
                settled_since = sample_now
            else:
                longest_streak = max(longest_streak, sample_now - settled_since)
                if sample_now - settled_since >= float(self.cfg.entry_settle):
                    return state
            time.sleep(0.02)
        if state is None:
            raise BridgeError("PX4 published no state while climbing to the entry pose.")
        here, speed = self.entry_state(state)
        limits = (f"offset<={float(self.cfg.entry_tolerance):.2f} m, "
                  f"speed<={float(self.cfg.entry_speed_tolerance):.2f} m/s, "
                  f"view<={float(getattr(self.cfg, 'entry_view_margin', 0.85)):.2f}")
        if not was_armed:
            raise ArmingRefused(
                f"PX4 never armed within {elapsed:.1f} s"
                + (f" (command {arm_refusal[0]} result {arm_refusal[1]})"
                   if arm_refusal is not None else "")
                + "; the reported entry offset is the parked vehicle's, not a "
                  "failure to hold station.")
        culprit = max(blocked, key=blocked.get) if samples else None
        if culprit is None or blocked[culprit] == 0:
            cause = ("every bound was met but never for the required "
                     f"{float(self.cfg.entry_settle):.2f} s in a row")
        else:
            cause = (f"{culprit} was out of tolerance on "
                     f"{blocked[culprit]}/{samples} samples "
                     f"(worst offset {worst['offset']:.2f} m, "
                     f"worst speed {worst['speed']:.2f} m/s)")
        # Name the clock that ran out. "Ran out of simulated time" is a vehicle
        # that never converged; "ran out of wall time" is a simulator that
        # stopped advancing, and the two need opposite responses.
        budget = (f"{sim_budget:.1f} simulated s ({elapsed:.0f} s wall)"
                  if expiry == "simulated" else
                  f"{wall_guard:.1f} s wall ({sim_elapsed:.1f} simulated s; "
                  "the stage is running far slower than real time)")
        final_offset = float(np.linalg.norm(here - target))
        # Where it started and how much of the gap it actually closed. "Never
        # in tolerance" reads identically for a vehicle that stood still and
        # for one that flew twenty metres and ran out of budget, and the two
        # need opposite responses.
        travel = ""
        if bool(np.isfinite(initial_offset)):
            travel = (f"; started {initial_offset:.2f} m out and closed "
                      f"{initial_offset - final_offset:.2f} m of that")
        if failsafe_since is not None:
            failsafe_seconds += time.monotonic() - failsafe_since
        if failsafe_seconds > 0.0:
            travel += (f"; PX4 was in a recoverable link failsafe "
                       f"({', '.join(failsafe_reasons) or 'unknown'}) for "
                       f"{failsafe_seconds:.1f} s of the climb, during which it "
                       "ignores the entry setpoint")
        # The gate scores position, speed and pad visibility, but visibility
        # also depends on where the aircraft is *pointing*, and that was the
        # one quantity it never reported. A run whose offset and speed are
        # both inside tolerance and whose view is not is either a geometry
        # fault or a heading the vehicle never turned to, and those need
        # opposite fixes.
        heading = ""
        if target_yaw is not None:
            actual_yaw = _yaw_from_quaternion_wxyz(state.get("quaternion_wxyz"))
            if actual_yaw is not None:
                error = math.degrees(_wrap_to_pi(actual_yaw - float(target_yaw)))
                heading = (f"; heading {math.degrees(actual_yaw):.0f} deg against "
                           f"a commanded {math.degrees(float(target_yaw)):.0f} deg "
                           f"({error:+.0f} deg out)")
        raise EntryResetError(
            f"PX4 did not hold the entry pose within {budget} "
            f"(last sample: offset {final_offset:.2f} m, "
            f"speed {speed:.2f} m/s, "
            "geometric pad-centre view offset "
            f"{'n/a' if view_margin is None else format(view_margin, '.2f')}"
            f"{heading}"
            # The distance to the target says nothing about *where* the pair
            # actually is. Offline the same entry geometry always leaves the
            # pad in view, so a live failure means these two vectors are not
            # the ones the reproduction assumed; print them rather than guess.
            f"; pad-relative [{here[0]:+.2f} {here[1]:+.2f} {here[2]:+.2f}] m"
            f" against a commanded [{target[0]:+.2f} {target[1]:+.2f} "
            f"{target[2]:+.2f}] m; "
            f"limits {limits}; longest hold {longest_streak:.2f} s of "
            f"{float(self.cfg.entry_settle):.2f} s; {cause}"
            + travel
            + (f"; re-aimed {view_retries}x" if view_retries else "") + ").")

    # ------------------------------------------------------------------- step
    def step(self, action: Iterable[float]) -> dict[str, Any]:
        a = np.asarray(list(action), dtype=float).reshape(-1)
        if a.size != 4 or not np.isfinite(a).all() or np.any(np.abs(a) > 1.0):
            raise BridgeError("Action must contain four finite values in [-1,1].")
        reply = self.transact("action", {"action": a.tolist()}, ("state",))
        state = self.pace_to_control_period(
            self.validate_state_with_estimator_grace(reply))
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
        state = self.pace_to_control_period(
            self.validate_state_with_estimator_grace(reply))
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
        observed = int(state["px4_time_us"])
        # An adopted older gateway may still expose XRCE-DDS's raw clock. Its
        # timesync filter can jump from Unix epoch back to boot time. Re-anchor
        # on that new domain instead of waiting for ~56,000 years and reporting
        # a huge negative "stall". New gateways normalize this upstream too.
        if observed < self.last_px4_time_us:
            self.last_px4_time_us = observed
        deadline = self.last_px4_time_us + self.control_period_us
        started = time.monotonic()
        while observed < deadline:
            if time.monotonic() - started > float(self.cfg.timeout):
                advanced = (observed - self.last_px4_time_us) / 1e3
                raise BridgeError(
                    f"PX4 simulated time advanced only {advanced:.1f} ms in "
                    f"{float(self.cfg.timeout):.2f} s of wall time; the simulator "
                    "has stalled.")
            state = self.validate_state_with_estimator_grace(
                self.transact("state", {}, ("state",)))
            observed = int(state["px4_time_us"])
            if observed < self.last_px4_time_us:
                self.last_px4_time_us = observed
                deadline = observed + self.control_period_us
        # Fixed cadence while on time, but re-anchor when an expensive monitor
        # or rendering callback let simulation advance past the deadline. If we
        # retain the missed deadline, subsequent actions run at odometry rate
        # until it catches up, under-counting a 20 s episode as only a few
        # seconds and starving PX4's setpoint stream again on the next callback.
        self.last_px4_time_us = pacing_anchor_us(deadline, observed)
        return state

    # ------------------------------------------------------------- commands
    def get_state(self) -> dict[str, Any]:
        state = self.validate_state(self.transact("state", {}, ("state",)))
        self.last_state = state
        return state

    def validate_state_with_estimator_grace(
            self, state: dict[str, Any], timeout: float | None = None
            ) -> dict[str, Any]:
        """Wait out a short EKF-validity flap without recording stale state.

        A two-camera rendered frame advances simulated time much more slowly
        than wall time. PX4 may publish one invalid local-position sample
        during estimator handover even though the next simulated sample is
        valid. The gateway continues streaming the current setpoint while this
        method waits; persistent invalidity still raises after the configured
        estimator warm-up window.
        """
        deadline = time.monotonic() + float(
            (getattr(self.cfg, "estimator_warmup", self.cfg.timeout)
             if timeout is None else timeout))
        current = state
        while True:
            try:
                return self.validate_state(current)
            except PX4EstimatorInvalid:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
                current = self.transact("state", {}, ("state",))

    def wait_for_failsafe_clear(self, timeout: float | None = None) -> bool:
        """Wait out a gateway-classified recoverable SITL link failsafe.

        Returns whether PX4 left the failsafe within the budget. The caller
        still discards whatever episode was interrupted -- a step PX4 did not
        fly is not a transition PPO may learn from -- but a self-clearing
        transport blip does not have to cost a rebuild of the shared simulator
        and, with it, the other pair's episode too.

        A hard failsafe or a dead transport returns ``False`` immediately, so
        the caller falls through to the stack restart it would have done.
        """
        deadline = time.monotonic() + float(
            getattr(self.cfg, "failsafe_grace", 10.0)
            if timeout is None else timeout)
        while True:
            try:
                state = self.transact("state", {}, ("state",))
            except BridgeError:
                return False
            extra = state.get("extra") if isinstance(
                state.get("extra"), dict) else {}
            if not bool(extra.get("px4_failsafe", False)):
                return True
            detail = (extra.get("px4_failsafe_detail")
                      if isinstance(extra.get("px4_failsafe_detail"), dict)
                      else {})
            if not bool(detail.get("recoverable_infrastructure", False)):
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def disarm(self) -> None:
        self.transact("disarm", {}, ("ack",))

    def land_and_wait(self, timeout: float | None = None) -> bool:
        """End an unfinished episode and wait until PX4 reports landed."""
        self.disable_offboard()
        # The gateway translates an in-air disarm request into NAV_LAND.
        self.disarm()
        deadline = time.monotonic() + float(
            timeout if timeout is not None else self.cfg.outcome_settle_timeout)
        while time.monotonic() < deadline:
            state = self.get_state()
            if bool(state.get("landed", False)):
                self.disarm()
                return True
            time.sleep(0.05)
        return False

    def hold_for_next_airborne_reset(self) -> None:
        """Keep an unfinished flight controlled between measured episodes.

        PPO optimization can take far longer than the action deadman. A
        bounded position hold keeps PX4 in OFFBOARD instead of entering
        AUTO.LAND, which cannot always be cancelled before the next entry
        timeout. The next episode is still gated on its independently seeded
        entry hover; this target is only unmeasured staging.

        The hold must outlast the whole between-episode gap: when it expired
        the gateway released the setpoint stream and commanded a landing, PX4
        raised ``offboard_control_signal_lost``, and the next reset rebuilt
        the shared simulator under both pairs. On a rendered two-pair stage
        that gap exceeded the old 120 s during estimator warm-up as well as
        during PPO updates.
        """
        state = self.last_state or self.get_state()
        truth = state.get("truth") if isinstance(state.get("truth"), dict) else {}
        position = np.asarray(
            truth.get("position") if truth.get("valid", False) else state["position"],
            dtype=float).reshape(3)
        radius = float(np.linalg.norm(position[:2]))
        if radius > 9.0:
            position[:2] *= 9.0 / radius
        position[2] = float(np.clip(position[2], 2.0, 8.0))
        yaw = float(quat_to_euler_zyx(state["quaternion_wxyz"])[2])
        self.transact(
            "goto", {"position": position.tolist(), "yaw": yaw,
                     "frame": "pad",
                     "hold_s": float(getattr(
                         getattr(self, "cfg", None),
                         "between_episode_hold_s", 900.0))},
            ("ack",))

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
        if contact_confirmed:
            # Keep OFFBOARD position streaming until PX4 acknowledges the
            # forced SITL disarm. Cutting it first creates an OFFBOARD-loss
            # failsafe whenever PX4 temporarily rejects the first disarm
            # request on a moving deck.
            deadline = time.monotonic() + float(
                timeout if timeout is not None else self.cfg.outcome_settle_timeout)
            while time.monotonic() < deadline:
                self.disarm()
                try:
                    state = self.get_state()
                except PX4EstimatorInvalid:
                    time.sleep(0.10)
                    continue
                if not bool(state.get("armed", False)):
                    try:
                        self.disable_offboard()
                    except BridgeError:
                        pass
                    return True
                time.sleep(0.20)
            return False
        try:
            self.disable_offboard()
        except BridgeError:
            pass
        self.disarm()
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
        extra = state.get("extra") if isinstance(state.get("extra"), dict) else {}
        ignored_disarmed_link_failsafe = False
        attitude_failure = False
        if bool(extra.get("px4_failsafe", False)):
            detail = (extra.get("px4_failsafe_detail")
                      if isinstance(extra.get("px4_failsafe_detail"), dict)
                      else {})
            reasons = detail.get("reasons", ())
            if not isinstance(reasons, (list, tuple)):
                reasons = ()
            recoverable = bool(detail.get(
                "recoverable_infrastructure", False))
            # PX4's attitude failure detector, alone: the vehicle tipped past
            # FD_FAIL_R/FD_FAIL_P. Raising here made a flipped drone end the
            # whole run -- ``collect_episode_resilient`` re-raises a
            # non-recoverable failsafe without spending a single retry -- even
            # though flight termination is circuit-broken in this build
            # (CBRK_FLIGHTTERM), so PX4 only warns and the state is still a
            # true reading of a real, already-failed flight. Hand it to the
            # environment instead; ``LiveShinEnvironment`` ends the episode as
            # a crash, PPO learns from the terminal penalty, and the run lives.
            attitude_failure = bool(detail.get("attitude_failure", False))
            # After a completed touchdown we intentionally disarm and stop
            # OFFBOARD. PX4 can retain VehicleStatus.failsafe for one callback
            # while its benign SITL link-loss flags clear. Rejecting that
            # disarmed transition restarts the entire shared two-pair world
            # and discards the unrelated trajectory. It is safe to accept
            # only this gateway-classified case while explicitly disarmed;
            # the same flag while armed still aborts the episode immediately.
            ignored_disarmed_link_failsafe = bool(
                recoverable and state.get("armed") is False)
            if not (ignored_disarmed_link_failsafe or attitude_failure):
                raise PX4Failsafe(reasons, recoverable=recoverable)
        if not state["estimator_valid"]:
            raise PX4EstimatorInvalid("PX4 estimator state is not valid yet.")
        if int(extra.get("offboard_mode_rejections", 0)) >= 6:
            raise BridgeError(
                "PX4 repeatedly rejected OFFBOARD mode; refusing a corrupted episode.")
        out = dict(state)
        if ignored_disarmed_link_failsafe or attitude_failure:
            out["extra"] = dict(extra)
            if ignored_disarmed_link_failsafe:
                out["extra"]["ignored_disarmed_link_failsafe"] = True
            if attitude_failure:
                out["extra"]["px4_attitude_failure"] = True
        for field in ("position", "velocity", "quaternion_wxyz",
                      "angular_velocity", "acceleration", "wind", "aero_force"):
            out[field] = np.asarray(state[field], dtype=float).reshape(-1)
        out["marker_quality"] = float(state["marker_quality"])
        return out
