"""Common causal task over local spatial dynamics or actual Isaac telemetry.

Truth has one consumer: ``Evaluator`` for reward/outcomes. Neither backend
passes it through the Measurement/Estimator/Observation policy boundary.
"""
from dataclasses import dataclass, replace
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .core import (
    SpatialConfig,
    Measurement,
    Estimator,
    safety_status,
    supervised_action,
    observation,
    camera_bearings,
)
from ..controllers.spatial_controller import SpatialAccelerationController
from ..landing.terminal import REFERENCE_TERMINAL_REWARDS, UNSAFE_REASONS, REFERENCE_SPIN_WEIGHT


@dataclass(frozen=True)
class Truth:
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    roll_pitch: np.ndarray
    angular_rate: np.ndarray
    contact: bool


class Evaluator:
    # One table, shared with the 2D route and with upstream MATLAB. A 3D
    # extension of a 2D task is only comparable to it if the objective is the
    # same, so these are NOT redefined here -- see landing/terminal.py for the
    # measurement that retired the separate spatial table.
    UNSAFE_STATUSES = frozenset(UNSAFE_REASONS)
    BONUSES = dict(REFERENCE_TERMINAL_REWARDS)
    NOMINAL_UNSAFE_PENALTY = max(
        REFERENCE_TERMINAL_REWARDS[name] for name in UNSAFE_REASONS)

    def __init__(self, cfg, *, unsafe_penalty=NOMINAL_UNSAFE_PENALTY):
        self.cfg = cfg
        self.unsafe_penalty = float(unsafe_penalty)
        if not self.unsafe_penalty < self.BONUSES["SAFE_ABORT"]:
            raise ValueError("unsafe penalty must remain worse than safe abort")
        self.previous_goal = None
        self.previous_readiness = None

    def terms(self, truth):
        cfg = self.cfg
        r, v = truth.relative_position, truth.relative_velocity
        xy = float(np.linalg.norm(r[:2]))
        sat = lambda value: value * value / (1 + value * value)
        goal = 0.65 * sat(xy / 3.0) + 0.35 * sat(max(r[2], 0) / 4.0)
        desired_xy = -r[:2] * min(0.6, cfg.touchdown_xy_speed / max(xy, 1e-9))
        desired_z = -min(0.8 * cfg.touchdown_z_speed, 0.5 * max(r[2], 0))
        if (cfg.schema != "spatial-causal-rgat/3"
                and 0 < r[2] - cfg.landing_gear_extension_m <= 0.45):
            # The offset nadir camera loses the pad before physical contact.
            # Preserve the existing sub-limit sink target in this near-field
            # interval instead of rewarding an asymptotic blind hover.
            # Derived from task limits, never from diagnostic actions.
            desired_z = -0.8 * cfg.touchdown_z_speed
        risk = (xy / cfg.pad_half_width) ** 2 + max(r[2], 0) ** 2
        risk += float(np.sum(((v[:2] - desired_xy) / cfg.touchdown_xy_speed) ** 2))
        risk += ((v[2] - desired_z) / cfg.touchdown_z_speed) ** 2
        risk += float(np.sum((truth.roll_pitch / cfg.touchdown_tilt) ** 2))
        risk += float(np.sum((truth.angular_rate[:2] / cfg.touchdown_rate) ** 2))
        return goal, math.exp(-0.5 * min(risk, 100))

    def reset(self, truth):
        self.previous_goal, self.previous_readiness = self.terms(truth)

    def evaluate(
        self, truth, *, elapsed, dt, action, safety, abort_elapsed, bearings, visible
    ):
        cfg = self.cfg
        r, v = truth.relative_position, truth.relative_velocity
        xy = float(np.linalg.norm(r[:2]))
        status = "RUNNING"
        # Contact/unsafe ordering always precedes benign timeouts/abort.
        if truth.contact:
            fast = (np.linalg.norm(v[:2]) > cfg.touchdown_xy_speed
                    or abs(v[2]) > cfg.touchdown_z_speed)
            if xy > cfg.pad_half_width:
                status = "MISSED_PAD_CONTACT"
            elif cfg.contact_verdict_by_speed:
                # /7: the impact speed is the contact criterion. Tilt and body
                # rate are the corridor's commit condition (safety.py
                # ``settled``) and the readiness term's price, not a reason to
                # score a 0.03 m/s settle onto the legs as a crash. The 21 deg
                # envelope still applies at the contact instant, and a contact
                # while the abort hold owns the vehicle is still unauthorized.
                if fast:
                    status = "UNSAFE_CONTACT"
                elif math.acos(float(np.clip(
                        math.cos(truth.roll_pitch[0]) * math.cos(truth.roll_pitch[1]),
                        -1., 1.))) > math.radians(21):
                    status = "SAFETY_ENVELOPE_VIOLATION"
                elif safety.abort:
                    status = "UNAUTHORIZED_CONTACT"
                else:
                    status = "SUCCESS"
            elif (
                fast
                or np.linalg.norm(truth.roll_pitch) > cfg.touchdown_tilt
                or np.linalg.norm(truth.angular_rate[:2]) > cfg.touchdown_rate
            ):
                status = "UNSAFE_CONTACT"
            elif safety.inhibited or safety.abort:
                status = "UNAUTHORIZED_CONTACT"
            else:
                status = "SUCCESS"
        elif xy > 60 or r[2] > 20 or r[2] < -0.2:
            status = "SAFETY_ENVELOPE_VIOLATION"
        elif cfg.reference_supervisor and math.acos(float(np.clip(
                math.cos(truth.roll_pitch[0])*math.cos(truth.roll_pitch[1]),-1.,1.))) > math.radians(21):
            status = "SAFETY_ENVELOPE_VIOLATION"
        elif abort_elapsed >= cfg.recovery_timeout:
            status = "SAFE_ABORT"
        elif elapsed >= cfg.horizon - 1e-8:
            status = "TASK_TIMEOUT"
        goal, readiness = self.terms(truth)
        view = (
            min(1.0, float(np.mean((bearings / (np.asarray(cfg.fov) / 2)) ** 2)))
            if visible
            else 1.0
        )
        control = float(np.mean(np.square(action)))
        # Squared NORM, matching the supervisor's own settled test
        # (|w_xy| <= touchdown_rate) and the 2D route's single axis: both
        # dimensions pay exactly 1.0 sitting on their own rate limit.
        spin = float(np.sum((truth.angular_rate[:2] / cfg.touchdown_rate) ** 2))
        running = -dt / 70.0 * (
            2.0 * goal + view + 0.25 * control + REFERENCE_SPIN_WEIGHT * spin
        ) + 8.0 * (readiness - self.previous_readiness)
        phi_next = -2.0 * goal if status == "RUNNING" else 0.0
        potential = (
            math.exp(-dt / cfg.discount_tau) * phi_next + 2.0 * self.previous_goal
        )
        terminal = (
            self.unsafe_penalty
            if status in self.UNSAFE_STATUSES
            else self.BONUSES.get(status, 0.0)
        )
        self.previous_goal, self.previous_readiness = goal, readiness
        return (
            float(running + potential + terminal),
            status,
            {
                "running": running,
                "potential": potential,
                "terminal": terminal,
                "goal": goal,
                "readiness": readiness,
                "view": view,
                "control": control,
                "spin": spin,
            },
        )


_BOARD_CACHE: dict = {}
_BOARD_ID_CACHE: dict = {}


def deployed_board_ids(cfg):
    """Registered marker ids of the deployed board, in the profile's order.

    This order is the common observation's fixed slot order; it never changes
    with what a frame happens to detect.
    """
    if cfg.schema not in _BOARD_ID_CACHE:
        from .runtime_contract import deployment_profile
        board = deployment_profile(cfg.schema)["resolved_scientific_configuration"]["vision"]["board"]
        _BOARD_ID_CACHE[cfg.schema] = tuple(int(m["id"]) for m in board)
    return _BOARD_ID_CACHE[cfg.schema]


#: Corner localisation noise of the local detector model, pixels (1 sigma).
#: An assumption, not a measurement: Isaac's corner error is whatever its
#: renderer and cv2 sub-pixel refinement produce, and no trace records it.
LOCAL_CORNER_NOISE_PX = 0.5


def deployed_board(cfg):
    """The tag layout the deployment profile paints, as (x, y, side) rows."""
    if cfg.schema not in _BOARD_CACHE:
        from .runtime_contract import deployment_profile
        board = deployment_profile(cfg.schema)["resolved_scientific_configuration"]["vision"]["board"]
        _BOARD_CACHE[cfg.schema] = np.array(
            [[float(m["center_xy_m"][0]), float(m["center_xy_m"][1]), float(m["side_m"])]
             for m in board], dtype=float)
    return _BOARD_CACHE[cfg.schema]


class LocalBackend:
    """Fast spatial sensor/plant fixture, never reported as Isaac flight."""

    name = "local-spatial"

    def __init__(self, cfg, *, difficulty=1.0):
        if not 0 <= difficulty <= 1:
            raise ValueError("difficulty must be in [0,1]")
        self.cfg = cfg
        self.difficulty = float(difficulty)
        self.board = deployed_board(cfg) if cfg.optical_realism else None

    def _optical_quality(self, r, q):
        """Detector-faithful visibility and confidence for the deployed board.

        A tag counts as seen when its padded quad (4/3 of the tag side, the
        white quiet zone ArUco needs) projects fully inside the frame at a
        decodable size. Confidence is the detector's own metric: the largest
        seen tag's pixel side over the full-scale 120 px, times a per-frame
        sharpness, times 0.85 when one tag braces the solve. Returns
        (visible, confidence).
        """
        from .core import (REFERENCE_OPTICAL_FULL_SCALE_PX, REFERENCE_OPTICAL_IMAGE_PX,
                           REFERENCE_OPTICAL_MIN_TAG_PX, REFERENCE_OPTICAL_MISS_PROBABILITY,
                           REFERENCE_OPTICAL_SHARPNESS)
        half_fov = np.asarray(self.cfg.fov) / 2
        fx = (REFERENCE_OPTICAL_IMAGE_PX[0] / 2) / math.tan(half_fov[0])
        sides_px = []
        for tx, ty, side in self.board:
            bearing, depth = camera_bearings(r + np.array([tx, ty, 0.0]), q, self.cfg)
            if depth <= 0.0:
                continue
            half_angle = math.atan((side * 4.0 / 3.0) / 2.0 / depth)
            if np.all(np.abs(bearing) + half_angle < half_fov):
                px = side * fx / depth
                if px >= REFERENCE_OPTICAL_MIN_TAG_PX:
                    sides_px.append(px)
        # Paired draws so a seed's random stream does not depend on geometry.
        miss = self.rng.uniform() < REFERENCE_OPTICAL_MISS_PROBABILITY
        sharpness = float(np.clip(self.rng.normal(*REFERENCE_OPTICAL_SHARPNESS), 0.0, 1.0))
        if not sides_px or miss:
            return False, 0.0
        scale = min(1.0, max(sides_px) / REFERENCE_OPTICAL_FULL_SCALE_PX)
        corroboration = 1.0 if len(sides_px) > 1 else 0.85
        return True, float(np.clip(sharpness * scale * corroboration, 0.0, 1.0))

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.pad = np.zeros(3)
        self.position = np.r_[
            self.rng.uniform(-self.cfg.initial_xy, self.cfg.initial_xy, 2),
            self.rng.uniform(*self.cfg.initial_height),
        ]
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.angles = np.zeros(3)
        self.rates = np.zeros(3)
        self.euler_rates = np.zeros(3)
        self.thrust_acceleration = 9.80665
        self.domain_sample = None
        self.domain_initial_pending = False
        if self.cfg.matched_disturbances:
            from ..benchmarks.randomization import sample_domain_randomization
            self.domain_sample = sample_domain_randomization(seed)
            self.domain_initial_pending = True
        self.t1 = float(self.rng.uniform(0.5, 2.0))
        self.t2 = float(self.rng.uniform(1.0, 2.0))
        self.v0 = float(self.rng.uniform(0.3, 0.8))
        self.a2 = float(self.rng.uniform(0.1, 0.4))
        if self.difficulty < 1.0:
            d = self.difficulty
            c = self.cfg.curriculum

            def interpolate_sample(value, nominal, start):
                fraction = (value - nominal[0]) / (nominal[1] - nominal[0])
                easy = start[0] + fraction * (start[1] - start[0])
                return (1 - d) * easy + d * value

            self.position[:2] *= 0.15 + 0.85 * d
            # The easy rung starts low. With landing gear the body touches
            # down higher, and from /5 on the camera must see a tag at the
            # band's low end, so the band is lifted (SpatialConfig
            # .easy_start_lift_m: the gear on /4-/7, kept on /8 without it).
            start_heights = tuple(v + self.cfg.easy_start_lift_m
                                  for v in c.start_height_range_m)
            self.position[2] = interpolate_sample(
                self.position[2], self.cfg.initial_height, start_heights
            )
            if self.cfg.schema != "spatial-causal-rgat/3":
                # The mounted camera's depth, not body altitude, determines
                # the easy spawn footprint. No extra random draw is consumed.
                depth = max(self.position[2] - 0.16, 0.01)
                limit = 0.4 * depth * min(np.tan(np.asarray(self.cfg.fov) / 2))
                radius = np.linalg.norm(self.position[:2])
                self.position[:2] *= min(1.0, limit / max(radius, 1e-12))
            self.v0 = interpolate_sample(self.v0, (0.3, 0.8), c.start_v1_range_m_s)
            self.a2 = interpolate_sample(self.a2, (0.1, 0.4), c.start_a2_range_m_s2)
            self.t1 = interpolate_sample(self.t1, (0.5, 2.0), c.start_T1_range_s)
        heading = float(self.rng.uniform(-0.2, 0.2))
        self.direction = np.array([math.cos(heading), math.sin(heading), 0.0])
        self.pad_velocity = self.direction * self.v0
        self.sensor_id = 0
        self.camera_sequence = 0
        self.camera_pending = []
        self.camera_time = -math.inf
        self.optical_position = None
        self.optical_confidence = 0.0
        self.optical_corners = None
        self.optical_corner_time = None
        # A SEPARATE stream for corner noise, so adding the common observation
        # moves no draw of the established sensor/plant stream.
        self.corner_rng = np.random.default_rng([int(seed), 0xC0])
        # Actuation latency (/3): commands arrive at the plant after
        # cfg.actuation_delay_s. Until the first one lands the vehicle keeps
        # the hover it was handed over in, as the entry hover does in Isaac.
        self.command_queue = []
        self.effective_command = None
        return self.measure(), self.truth()

    def _hover_like(self, command):
        from dataclasses import replace as _replace
        zeros = np.zeros(3)
        return _replace(command, velocity_enu_m_s=zeros, acceleration_enu_m_s2=zeros,
                        normalized_action=zeros, derived_roll_pitch_rad=(0.0, 0.0),
                        thrust_weight_ratio=1.0, constrained=False)

    def _active_command(self, command):
        """The command the plant runs during this substep."""
        delay = self.cfg.actuation_delay_s
        if delay <= 0:
            return command
        while self.command_queue and self.command_queue[0][0] <= self.t + 1e-9:
            self.effective_command = self.command_queue.pop(0)[1]
        if self.effective_command is None:
            return self._hover_like(command)
        return self.effective_command

    def measure(self):
        q = Rotation.from_euler("xyz", self.angles).as_quat()[[3, 0, 1, 2]]
        if self.t - self.camera_time >= self.cfg.camera_dt - 1e-9:
            r = self.position - self.pad
            bearing, depth = camera_bearings(r, q, self.cfg)
            noise = self.rng.normal(0, 0.02, 3)  # paired draw even on missed frames
            if self.board is not None:
                visible, confidence = self._optical_quality(r, q)
            else:
                visible = bool(
                    np.all(np.abs(bearing) < np.asarray(self.cfg.fov) / 2) and depth > 0
                )
                confidence = .98 if visible else 0.
            self.camera_time = self.t
            self.camera_sequence += 1
            corners = (self._marker_corners(r, q, visible)
                       if self.board is not None else None)
            frame = (self.camera_sequence,self.t,r+noise if visible else None,
                     confidence,self.position.copy(),q.copy(),corners)
            if self.cfg.reference_context and self.t > 0:
                self.camera_pending.append(frame)
            else:
                # Reset primes an already available pre-policy optical track,
                # just as actual handover waits for the first valid image.
                self._receive_camera(frame)
        if self.cfg.reference_context:
            while self.camera_pending and self.camera_pending[0][1] <= self.t-self.cfg.camera_transport_delay+1e-9:
                self._receive_camera(self.camera_pending.pop(0))
        return Measurement(
            self.t,
            self.position.copy(),
            self.velocity.copy(),
            q,
            self.rates.copy(),
            self.optical_position,
            self.optical_confidence,
            self.sensor_id,
            self.optical_capture_time if self.cfg.reference_context else None,
            self.optical_own_position.copy() if self.cfg.reference_context else None,
            self.optical_quaternion.copy() if self.cfg.reference_context else None,
            self.optical_corners,
            self.optical_corner_time,
        )

    def _receive_camera(self, frame):
        (self.sensor_id,self.optical_capture_time,self.optical_position,
         self.optical_confidence,self.optical_own_position,self.optical_quaternion,
         self.optical_corners)=frame
        self.optical_corner_time = (None if self.optical_corners is None
                                    else self.optical_capture_time)

    def _marker_corners(self, r, q, visible):
        """{marker id: (4, 2) pixel corners} of every decodable board tag.

        Corners in the detector's own order (marker_vision.BoardMarker): from
        the marker image's top-left clockwise, image "up" along pad +Y. A tag
        counts when its padded quad (the white quiet zone) projects fully into
        the 640x480 frame at >= 12 px, the rule the local confidence already
        uses, and only in a frame the local detector did not miss. Noise is
        drawn for every registered tag every frame so the stream does not
        depend on geometry.
        """
        from .core import REFERENCE_OPTICAL_IMAGE_PX, REFERENCE_OPTICAL_MIN_TAG_PX
        from scipy.spatial.transform import Rotation
        width, height = REFERENCE_OPTICAL_IMAGE_PX
        fx = (width / 2) / math.tan(self.cfg.fov[0] / 2)
        body_from_world = Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).inv()
        noise = self.corner_rng.normal(0.0, LOCAL_CORNER_NOISE_PX, (len(self.board), 4, 2))
        ids = deployed_board_ids(self.cfg)

        def project(points):
            body = body_from_world.apply(points - r)          # pad points from the body
            optical = (body - np.array([0, 0, -0.16])) * np.array([1, -1, -1])
            if np.any(optical[:, 2] <= 1e-6):
                return None
            return np.column_stack([width / 2 + fx * optical[:, 0] / optical[:, 2],
                                    height / 2 + fx * optical[:, 1] / optical[:, 2]])
        out = {}
        if not visible:
            return out
        for k, (tx, ty, side) in enumerate(self.board):
            def square(s):
                h = s / 2
                return np.array([[tx - h, ty + h, 0.0], [tx + h, ty + h, 0.0],
                                 [tx + h, ty - h, 0.0], [tx - h, ty - h, 0.0]])
            padded = project(square(side * 4.0 / 3.0))
            quad = project(square(side))
            if padded is None or quad is None:
                continue
            inside = (np.all(padded[:, 0] >= 0) and np.all(padded[:, 0] <= width - 1)
                      and np.all(padded[:, 1] >= 0) and np.all(padded[:, 1] <= height - 1))
            side_px = math.sqrt(abs(0.5 * np.cross(quad[2] - quad[0], quad[3] - quad[1])))
            if inside and side_px >= REFERENCE_OPTICAL_MIN_TAG_PX:
                out[int(ids[k])] = quad + noise[k]
        return out

    def pad_state(self, t):
        """Exact CV–CA–CV trajectory, continuous in both position and velocity."""
        middle = float(np.clip(t - self.t1, 0, self.t2))
        coast = max(0.0, t - self.t1 - self.t2)
        distance = self.v0 * t + 0.5 * self.a2 * middle ** 2 + self.a2 * self.t2 * coast
        return self.direction * distance, self.direction * (self.v0 + self.a2 * middle)

    def truth(self):
        r = self.position - self.pad
        return Truth(
            r.copy(),
            self.velocity - self.pad_velocity,
            self.angles[:2].copy(),
            self.rates.copy(),
            bool(r[2] <= self.cfg.touchdown_height),
        )

    def advance(self, command, callback):
        if self.domain_initial_pending:
            sample = self.domain_sample
            scale = self.difficulty
            self.velocity += scale*Rotation.from_euler("xyz", self.angles).apply(
                sample.initial_velocity_m_s)
            self.rates += scale*sample.initial_angular_rate_rad_s
            self.euler_rates += scale*sample.initial_angular_rate_rad_s
            self.domain_initial_pending = False
        if self.cfg.actuation_delay_s > 0:
            self.command_queue.append((self.t + self.cfg.actuation_delay_s, command))
        end = self.t + self.cfg.dt
        while self.t < end - 1e-9:
            h = min(self.cfg.sensor_dt, end - self.t)
            active = self._active_command(command)
            previous_r = self.position - self.pad
            previous_v = self.velocity.copy()
            previous_position = self.position.copy()
            previous_angles = self.angles.copy()
            previous_rates = self.rates.copy()
            self.pad, self.pad_velocity = self.pad_state(self.t + h)
            if self.cfg.direct_acceleration:
                from .dynamics import advance_attitude_thrust
                disturbance = {}
                if self.domain_sample is not None:
                    sample = self.domain_sample
                    d = self.difficulty
                    gains = np.array([sample.attitude_gain_roll_pitch/1.725]*2 +
                                     [sample.attitude_gain_yaw/.325])
                    disturbance = dict(force_body=d*sample.external_force_n,
                        torque_body=d*sample.external_torque_nm,
                        attitude_gain_scale=1+d*(gains-1))
                (self.angles, self.euler_rates, self.thrust_acceleration,
                 self.acceleration, self.rates) = advance_attitude_thrust(
                    self.angles, self.euler_rates, self.thrust_acceleration, active, h,
                    attitude_omega=self.cfg.attitude_omega,
                    attitude_damping=self.cfg.attitude_damping, **disturbance)
                self.position += self.velocity*h + 0.5*self.acceleration*h*h
                self.velocity += self.acceleration*h
            else:
                desired = active.acceleration_enu_m_s2 + 1.8 * (
                    active.velocity_enu_m_s - self.velocity
                )
                self.acceleration += (desired - self.acceleration) * min(1, h / 0.08)
                self.velocity += self.acceleration * h
                self.position += self.velocity * h
                target_angles = np.r_[active.derived_roll_pitch_rad, active.yaw_enu_rad]
                self.rates = (target_angles - self.angles) / 0.15
                self.angles += self.rates * h
            self.t += h
            # Interpolate first crossing to keep contact time and state coherent.
            r = self.position - self.pad
            if (
                r[2] <= self.cfg.touchdown_height
                and previous_r[2] > self.cfg.touchdown_height
            ):
                alpha = (previous_r[2] - self.cfg.touchdown_height) / (
                    previous_r[2] - r[2]
                )
                self.t -= h * (1 - alpha)
                self.pad, self.pad_velocity = self.pad_state(self.t)
                self.position = previous_position + alpha * (
                    self.position - previous_position
                )
                self.position[2] = self.pad[2] + self.cfg.touchdown_height
                self.velocity = previous_v + alpha * (self.velocity - previous_v)
                self.angles = previous_angles + alpha * (self.angles - previous_angles)
                self.rates = previous_rates + alpha * (self.rates - previous_rates)
                callback(self.measure())
                break
            callback(self.measure())
        return self.truth()

    def close(self):
        pass

    def finish(self):
        return True


class IsaacBackend:
    name = "isaac-px4-spatial"

    def __init__(self, cfg, *, pair=0, scenario=None):
        from ..config import default_config
        from ..bridge import PX4Bridge
        from .runtime_contract import deployment_profile, runtime_source_hash, wait_runtime_identity

        self.profile_sha256 = deployment_profile(cfg.schema)["sha256"]
        if cfg.isaac_profile_sha256 and cfg.isaac_profile_sha256 != self.profile_sha256:
            raise ValueError(
                "Isaac scientific profile differs from checkpoint deployment contract"
            )
        self.cfg = cfg
        self.pair = pair
        self.scenario = scenario or ('spatial_reference_cv_ca_cv' if cfg.reference_context
                                     else 'segmented_cruise_slow')
        bridge_cfg = default_config("quick")
        bridge_cfg.external.gateway_port = 14650 + 2 * pair
        bridge_cfg.external.local_port = 14651 + 2 * pair
        bridge_cfg.external.timeout = 8.0
        bridge_cfg.external.setup_timeout = 30.0
        bridge_cfg.external.entry_timeout = 180.0
        bridge_cfg.external.entry_sim_budget = 90.0
        bridge_cfg.external.entry_settle = 0.5
        bridge_cfg.external.entry_speed_tolerance = 0.6
        bridge_cfg.external.entry_require_offboard = True
        if cfg.direct_acceleration:
            # Legacy moving-entry tolerance (.9 m/.6 m/s) admitted a spatial
            # handover at 1.68 m for a seeded 2.04 m start, before the SAME
            # deliberate DR impulse. The pad is parked throughout entry here.
            # Require convergence before applying that impulse; never redraw
            # the seed or relax any flight/landing/preflight criterion.
            bridge_cfg.external.entry_tolerance = 0.2
            bridge_cfg.external.entry_speed_tolerance = 0.15
            bridge_cfg.external.entry_settle = 1.0
        self.entry_contract = {
            "position_tolerance_m": bridge_cfg.external.entry_tolerance,
            "speed_tolerance_m_s": bridge_cfg.external.entry_speed_tolerance,
            "settle_s": bridge_cfg.external.entry_settle,
            "seeded_disturbance_after_entry": True,
            "offboard_throughout_settle": True,
        }
        bridge_cfg.external.landing_camera = {
            "resolution": [640, 480],
            "horizontal_fov_deg": 90.0,
            "pitch_down_deg": 90.0,
            "mount_translation_flu_m": [0, 0, -0.16],
        }
        self.bridge = PX4Bridge(bridge_cfg)
        try:
            # Verify the configuration actually loaded by the running world,
            # not merely the YAML currently on disk, BEFORE reset/arming.
            # Runtime identity is a read-only preflight query, not an accepted
            # control sample. A degraded *previous* episode must not prevent
            # an explicitly scheduled cold restart in reset(). The reset and
            # every policy measurement still enforce estimator validity.
            self.identity_readiness = wait_runtime_identity(self.bridge,
                profile_sha256=self.profile_sha256, source_sha256=runtime_source_hash())
            from .lifecycle import record_flight_event
            record_flight_event(dict(phase='runtime_identity_ready',
                                     **self.identity_readiness))
        except BaseException:
            self.bridge.close()
            raise
        # Timing uses the explicit Isaac clock below, not PX4 DDS timestamps.
        self.bridge.control_period_us = 0
        self.last_state = None
        self.finished = True

    def reset(self, seed):
        from ..bridge import EntryResetError, PX4Failsafe
        from .lifecycle import recover_refused_reset, prepare_isolated_episode
        if prepare_isolated_episode(seed=seed,release=self.bridge.close):
            self.__init__(self.cfg,pair=self.pair,scenario=self.scenario)
        while True:
            try:
                try:
                    return self._reset_once(seed)
                except PX4Failsafe as failsafe:
                    # Raised during the entry hover, before any policy step:
                    # a gateway-classified link failsafe that outlived the
                    # bridge's grace is an entry failure, the same
                    # infrastructure outcome EntryResetError names, so it uses
                    # the same owned-restart budget and same-seed retry. A hard
                    # failsafe (battery, estimator, geofence) still propagates.
                    # Measured 2026-10-07: one such failsafe ended a whole
                    # minimal-contract Isaac run at its fifth entry.
                    if not failsafe.recoverable:
                        raise
                    raise EntryResetError(
                        f"pre-policy link failsafe did not clear: {failsafe}") from failsafe
            except EntryResetError as exc:
                def release():
                    # The owned stack takes responsibility for stopping SITL.
                    # Do not issue cleanup commands through a closed socket.
                    self.finished = True
                    self.bridge.close()
                if not recover_refused_reset(exc, seed=seed, release=release):
                    raise
                self.__init__(self.cfg, pair=self.pair, scenario=self.scenario)

    def _reset_once(self, seed):
        self.finished = False
        self.seed = int(seed)
        self.start_s = math.inf
        self.last_state = self.bridge.reset(
            seed, scenario=self.scenario, initial_condition_scale=1.0
        )
        start = time.monotonic()
        while True:
            self.assert_policy_flight(self.last_state, pre_policy=True)
            m = Measurement.from_wire(self.last_state,capture_aligned=self.cfg.reference_context)
            if m.optical_position is not None:
                break
            if time.monotonic() - start > 15:
                from ..bridge import EntryResetError
                raise EntryResetError("no real optical pad measurement at handover; no policy transition collected")
            time.sleep(0.03)
            self.last_state = self.bridge.get_state()
        self.start_s = m.time_s
        self.t = m.time_s
        return m, self.truth()

    def assert_policy_flight(self, state, *, allow_contact=False, pre_policy=False):
        """Reject setup/failed-control states before they become RL samples.

        A fresh evaluator-only contact may force-disarm between physics and
        telemetry callbacks. That terminal snapshot remains evaluable; an old
        contact or an unrelated disarm must not authorize continued learning.
        """
        event = state.get("extra", {}).get("truth_contact_event")
        if (allow_contact and event is not None
                and event.get("seed") == self.seed
                and event.get("sample_time_s", -math.inf) >= self.start_s):
            return
        if state.get("armed") is not True or state.get("nav_state") != 14:
            if pre_policy:
                from ..bridge import EntryResetError
                from .lifecycle import record_flight_event
                record_flight_event(dict(phase='entry_refused_before_policy',
                    seed=getattr(self,'seed',None),transitions_collected=0,
                    armed=state.get('armed'),nav_state=state.get('nav_state')))
                error_type = EntryResetError
            else:
                error_type = RuntimeError
            raise error_type(
                "spatial policy requires armed PX4 in OFFBOARD; "
                f"armed={state.get('armed')}, nav_state={state.get('nav_state')}; "
                "setup/arming failure is not an RL episode")

    def truth(self):
        s = self.last_state
        event = s.get("extra", {}).get("truth_contact_event")
        if (
            event is not None
            and event.get("seed") == self.seed
            and event.get("sample_time_s", -math.inf) >= self.start_s
        ):
            return Truth(
                np.asarray(event["relative_position"]),
                np.asarray(event["relative_velocity"]),
                np.asarray(event["roll_pitch"]),
                np.asarray(event["angular_rate"]),
                True,
            )
        truth = s.get("truth", {})
        if not truth.get("valid") or not truth.get("attitude_valid"):
            raise RuntimeError(
                "isolated evaluator needs valid truth; never substitute measurement"
            )
        q = np.asarray(truth["quaternion_wxyz"])
        angles = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")
        # Contact is evaluator-only. Policy and supervisor never receive it.
        contact = bool(s.get("extra", {}).get("pad_contact"))
        if contact:
            raise RuntimeError(
                "physical contact has no fresh pre-impact evaluator snapshot"
            )
        return Truth(
            np.asarray(truth["position"]),
            np.asarray(truth["velocity"]),
            angles[:2],
            np.asarray(s["angular_velocity"]),
            contact,
        )

    def advance(self, command, callback):
        self.assert_policy_flight(self.last_state)
        end = self.t + self.cfg.dt
        self.last_state = (self.bridge.step_spatial_acceleration(command)
                           if self.cfg.direct_acceleration
                           else self.bridge.step_spatial_velocity(command))
        started = time.monotonic()
        while True:
            self.assert_policy_flight(self.last_state, allow_contact=True)
            measurement = Measurement.from_wire(self.last_state,capture_aligned=self.cfg.reference_context)
            if measurement.time_s >= self.t:
                callback(measurement)
                self.t = measurement.time_s
            extra = self.last_state.get("extra", {})
            event = extra.get("truth_contact_event")
            first_contact = (
                event is not None
                and event.get("seed") == self.seed
                and event.get("sample_time_s", -math.inf) >= self.start_s
            )
            if first_contact or (self.t >= end - 1e-8 and not extra.get("pad_contact")):
                break
            if time.monotonic() - started > 8.0:
                raise RuntimeError("Isaac clock stalled during a spatial step")
            time.sleep(0.01)
            self.last_state = self.bridge.validate_state_with_estimator_grace(
                self.bridge.transact("state", {}, ("state",)), timeout=2.0)
            self.bridge.last_state = self.last_state
        return self.truth()

    def finish(self):
        from .lifecycle import record_confirmed_stop, record_flight_event
        self.finished = False
        started = time.monotonic()
        try:
            self.finished = self.bridge.stop_after_outcome(timeout=180.0)
            return self.finished
        finally:
            record_confirmed_stop(self.finished)
            record_flight_event(dict(phase="cleanup_finished", seed=getattr(self, "seed", None),
                stop_confirmed=self.finished, wall_seconds=time.monotonic()-started))

    def close(self):
        try:
            if not self.finished and not self.finish():
                raise RuntimeError("spatial flight stop was not confirmed")
        finally:
            self.bridge.close()


class SpatialLandingEnv:
    def __init__(self, config=None, backend=None, *, difficulty=1.0):
        self.cfg = config or SpatialConfig()
        if backend is not None and difficulty != 1.0:
            raise ValueError("Isaac validation cannot silently use a curriculum")
        self.backend = backend or LocalBackend(self.cfg, difficulty=difficulty)
        self.difficulty = float(difficulty)
        self.controller = SpatialAccelerationController(
            max_velocity=self.cfg.max_velocity,
            max_acceleration=self.cfg.max_acceleration,
            dt=self.cfg.dt,
            acceleration_only=self.cfg.direct_acceleration,
            max_tilt_deg=20. if self.cfg.reference_supervisor else 25.,
        )
        self.done = True
        self.episode_history = []

    def reset(self, *, seed, difficulty=None, angular_curriculum_scales=(1.0, 1.0),
              loss_curriculum_start=None):
        if loss_curriculum_start is not None:
            if (not isinstance(self.backend, LocalBackend)
                    or not np.isfinite(loss_curriculum_start)
                    or not self.cfg.loss_timeout <= loss_curriculum_start <= 30.):
                raise ValueError('loss-timeout curriculum is local training only, in [nominal,30]')
        scales = np.asarray(angular_curriculum_scales, dtype=float)
        if (
            scales.shape != (2,)
            or not np.isfinite(scales).all()
            or np.any(scales < 1)
            or np.any(scales > 4)
        ):
            raise ValueError("angular curriculum scales must be two values in [1,4]")
        if not isinstance(self.backend, LocalBackend) and np.any(scales != 1):
            raise ValueError("angular curriculum is restricted to local training")
        if difficulty is not None:
            if not isinstance(self.backend, LocalBackend):
                raise ValueError("curriculum is restricted to local training")
            if not 0 <= difficulty <= 1:
                raise ValueError("difficulty must be in [0,1]")
            self.difficulty = self.backend.difficulty = float(difficulty)
        d, c = self.difficulty, self.cfg.curriculum
        # At nominal difficulty preserve the exact object and all semantics.
        self.task_cfg = (
            self.cfg
            if d == 1
            else replace(
                self.cfg,
                touchdown_xy_speed=(1 - d) * c.start_touchdown_relative_speed_m_s
                + d * self.cfg.touchdown_xy_speed,
                touchdown_z_speed=(1 - d) * c.start_touchdown_vertical_speed_m_s
                + d * self.cfg.touchdown_z_speed,
                touchdown_tilt=((1 - d) * scales[0] + d) * self.cfg.touchdown_tilt,
                touchdown_rate=((1 - d) * scales[1] + d) * self.cfg.touchdown_rate,
                loss_timeout=(self.cfg.loss_timeout if loss_curriculum_start is None else
                              (1-d)*loss_curriculum_start+d*self.cfg.loss_timeout),
            )
        )
        self.episode_seed = int(seed)
        m, truth = self.backend.reset(seed)
        self.estimator = Estimator(self.cfg)
        self.estimator.update(m)
        self.start = m.time_s
        self.time = m.time_s
        self.abort_since = None
        self.evaluator = Evaluator(
            self.task_cfg,
            unsafe_penalty=(
                Evaluator.NOMINAL_UNSAFE_PENALTY if d == 1
                else (1 - d) * c.start_unsafe_contact_penalty
                + Evaluator.NOMINAL_UNSAFE_PENALTY * d
            ),
        )
        self.evaluator.reset(truth)
        yaw = Rotation.from_quat(m.quaternion[[1, 2, 3, 0]]).as_euler("xyz")[2]
        self.controller.reset(
            own_velocity_enu_m_s=m.own_velocity, yaw_enu_rad=float(yaw)
        )
        self.supervisor = None
        if self.cfg.reference_supervisor:
            from .safety import ReferenceSpatialSupervisor
            self.supervisor = ReferenceSpatialSupervisor()
        self.safety = (safety_status(self.estimator, self.task_cfg) if self.supervisor is None
                       else self.supervisor.status(self.estimator, self.task_cfg))
        self.done = False
        self.common_builder = None
        if self.cfg.optical_realism:
            from ..landing.common_observation import CommonObservationBuilder
            from .core import REFERENCE_OPTICAL_IMAGE_PX
            self.common_builder = CommonObservationBuilder(
                deployed_board_ids(self.cfg), REFERENCE_OPTICAL_IMAGE_PX)
        return self._with_common(observation(self.estimator, self.safety, 0.0, self.cfg)), {
            "backend": self.backend.name
        }

    def _with_common(self, obs):
        """Attach O_t built from the latest allowlisted Measurement only."""
        if self.common_builder is None:
            return obs
        from ..landing.common_observation import drone_state_3d
        m = self.estimator.own
        pad = self.common_builder.pad_observation(
            m.marker_corners_px, m.marker_capture_time_s, m.time_s)
        drone = drone_state_3d(m.own_position, m.own_velocity, m.quaternion,
                               m.angular_rate, m.time_s)
        self.common_observation = self.common_builder.observe(pad, drone)
        return replace(obs, common=self.common_observation)

    def step(self, action):
        if self.done:
            raise RuntimeError("reset required after terminal")
        est = self.estimator
        applied = (supervised_action(action, est, self.safety, self.task_cfg)
                   if self.supervisor is None else
                   self.supervisor.action(action, est, self.safety, self.task_cfg))
        command = self.controller.command(applied, own_velocity_enu_m_s=est.own.own_velocity)
        preceding = self.safety
        truth = self.backend.advance(command, est.update)
        dt = est.last_t - self.time
        if dt <= 0 or dt > 2:
            raise RuntimeError("invalid elapsed spatial control time")
        self.time = est.last_t
        elapsed = self.time - self.start
        self.safety = (safety_status(est, self.task_cfg) if self.supervisor is None
                       else self.supervisor.status(est, self.task_cfg))
        if self.safety.abort:
            if self.abort_since is None:
                self.abort_since = elapsed
        else:
            self.abort_since = None
        bearings, _ = camera_bearings(est.r, est.own.quaternion, self.cfg)
        view_available = est.detected
        if self.cfg.reference_supervisor:
            measurement = est.own
            view_available = measurement.optical_position is not None
            if view_available:
                bearings, _ = camera_bearings(measurement.optical_position,
                    measurement.optical_quaternion, self.cfg)
        reward, status, components = self.evaluator.evaluate(
            truth,
            elapsed=elapsed,
            dt=dt,
            action=action,
            safety=preceding,
            abort_elapsed=0 if self.abort_since is None else elapsed - self.abort_since,
            bearings=bearings,
            visible=view_available,
        )
        self.done = status != "RUNNING"
        if self.done and isinstance(self.backend, IsaacBackend):
            from .lifecycle import record_flight_event
            record_flight_event(dict(phase="terminal_before_cleanup", seed=self.episode_seed,
                status=status, elapsed_s=elapsed, stop_confirmed=False,
                truth_relative_position=truth.relative_position.tolist(),
                truth_relative_velocity=truth.relative_velocity.tolist(),
                requested_action=np.asarray(action).tolist(),
                applied_acceleration=command.acceleration_enu_m_s2.tolist()))
        if self.done and not self.backend.finish():
            raise RuntimeError(
                "spatial terminal was scored but PX4 stop is unconfirmed"
            )
        if self.done:
            self.episode_history.append(
                {
                    "seed": self.episode_seed,
                    "status": status,
                    "elapsed_s": elapsed,
                    "backend": self.backend.name,
                    "stop_confirmed": True,
                    "difficulty": self.difficulty,
                }
            )
        info = {
            "status": status,
            "dt_s": dt,
            "elapsed_s": elapsed,
            "requested_acceleration_m_s2": (
                np.asarray(action) * self.cfg.max_acceleration
            ).tolist(),
            "applied_acceleration_m_s2": command.acceleration_enu_m_s2.tolist(),
            "reward_components": components,
            "safety_intervened": bool(not np.allclose(action, applied)),
            "safety_reasons": preceding.reasons,
            "abort_requested": preceding.abort,
            "truth_relative_position": truth.relative_position.tolist(),
            "truth_relative_velocity": truth.relative_velocity.tolist(),
            "estimated_relative_position": est.r.tolist(),
            "estimated_relative_velocity": est.rv.tolist(),
            "estimated_pad_velocity": est.pad_v.tolist(),
            "optical_age_s": est.age,
            "optical_capture_time_s": est.own.optical_time_s,
            "optical_sample_id": est.own.sample_id,
            "optical_transport_age_s": (None if est.own.optical_time_s is None else
                                        est.own.time_s-est.own.optical_time_s),
            "optical_detected": est.detected,
            "truth_contact": truth.contact,
            "truth_roll_pitch": truth.roll_pitch.tolist(),
            "truth_angular_rate": truth.angular_rate.tolist(),
            "estimator_updates": est.updates,
            "backend": self.backend.name,
            "backend_profile_sha256": getattr(self.backend, "profile_sha256", None),
            "own_velocity_enu_m_s": est.own.own_velocity.tolist(),
            "own_quaternion_wxyz": est.own.quaternion.tolist(),
            "commanded_velocity_enu_m_s": command.velocity_enu_m_s.tolist(),
            "acceleration_only": command.acceleration_only,
            "optical_relative_position": (None if est.own.optical_position is None
                                           else est.own.optical_position.tolist()),
        }
        if isinstance(self.backend, IsaacBackend):
            # Audit-only, never part of actor/critic observations.
            info["flight_armed"] = self.backend.last_state["armed"]
            info["flight_nav_state"] = self.backend.last_state["nav_state"]
            info["backend_entry_contract"] = self.backend.entry_contract
        if self.cfg.reference_tracking:
            est.previous_action = np.clip(np.asarray(action, dtype=float), -1., 1.).copy()
            # Own commanded acceleration, for the causal disturbance observer.
            # It is what the vehicle asked for, not what the world did.
            est.observe_actuation(command.acceleration_enu_m_s2)
        obs = self._with_common(observation(est, self.safety, elapsed, self.cfg))
        if self.common_builder is not None:
            info["common_markers_detected"] = int(np.sum(obs.common.pad.detected_mask))
        return (
            obs,
            reward,
            self.done,
            False,
            info,
        )

    def close(self):
        self.backend.close()
