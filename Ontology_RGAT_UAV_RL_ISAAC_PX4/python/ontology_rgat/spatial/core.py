"""Causal measurement/estimation/packet/graph/supervision boundaries.

Only ``Measurement`` can enter the estimator. It has no truth, contact,
reward, outcome, or platform ground-truth fields. Simulator truth is isolated
in the environment evaluator. Spatial relative position means UAV minus pad.
"""
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import numpy as np

from ..two_axis.config import GraphConfig, CurriculumConfig
from ..landing.packet import (SPATIAL_AXES, field_names as reference_field_names,
                              registry as reference_registry)
from ..landing.terminal import (
    REFERENCE_CURRICULUM_START_UNSAFE,
    REFERENCE_DISCOUNT_TAU_S,
    REFERENCE_MISSION_HORIZON_S,
    REFERENCE_TERMINAL_REWARDS,
    validate_curriculum_ramp,
    validate_discount_horizon,
    validate_terminal_ordering,
)
from ..two_axis.ontology_v28 import GRAPH_SCHEMA_HASH as TOPOLOGY_HASH


#: The dimension-generic contract: the 2D reference packet with one more
#: horizontal axis, the reference 9x12 plane per axis, and the v10 capability
#: set (direct acceleration, ABG tracking, matched disturbances, capture-time
#: optical alignment, reference supervisor). It is derived, not a ladder rung.
#:
#: /2 differs from /1 in exactly one law: while descent is inhibited the
#: supervisor holds the vertical rate instead of braking a descent one-sidedly.
#: Observation, packet, graph, reward and every capability are identical, so /1
#: stays loadable and comparable -- see ``vertical_inhibit_holds``.
#: /3 differs from /2 in exactly one law: the local plant applies a command
#: only after ``REFERENCE_ACTUATION_DELAY_S`` of actuation latency, and the
#: supervisor's stopping margin counts that latency. Measured 2026-10-06 from
#: the first learned-policy Isaac flight (12 episodes, 0 landings, pad lost
#: within ~1 s): a joint impulse-response fit of ground-truth lateral
#: acceleration against the commanded one gives Isaac/PX4 h = [-0.02, 0.74,
#: 0.24, -0.01] over 0.1 s lags -- nothing in the step the command is issued
#: -- where the /2 plant gives [0.44, 0.15, 0.18, -0.04], i.e. responds inside
#: the same step. Policies tuned on /2 therefore over-command in Isaac, tilt
#: to 18-21 deg and lose the pad from the camera. With one step of latency the
#: local fit becomes [0.04, 0.36, 0.24, 0.14]. Observation, packet, graph,
#: reward and every capability are identical to /2, so /1 and /2 checkpoints
#: stay loadable under their own schema; only ``config_sha256`` differs.
#: /4 is the Isaac-calibrated vehicle, in three measured parts, each with its
#: own tool and number (2026-10-06, after the first two learned-policy Isaac
#: flights landed 0 of 22 episodes while the same checkpoints landed locally):
#:
#: * attitude bandwidth 14 rad/s: replaying the recorded Isaac commands through
#:   the delayed local plant and comparing the velocity response after large
#:   command changes (tools/audit_actuation_response.py, 146 events over 21
#:   episodes) gives Isaac [0.34, 1.09, 1.65, 1.80] against 10 rad/s
#:   [0.25, 0.75, 1.24, 1.49] (error 0.286) and 14 rad/s [0.41, 1.05, 1.55,
#:   1.68] (error 0.084); 20 rad/s and above overshoot. The /3 flight still
#:   limit-cycled laterally with the 10 rad/s policy.
#: * landing gear 0.18 m: the camera sits 0.16 m below the body and the body
#:   touched down at 0.12 m, so the camera was 0.04 m BELOW the pad surface at
#:   contact and lost the markers before touchdown; with the gear it stays
#:   0.14 m above the pad, where the centre marker of the v11 board fills
#:   ~180 px of a 640 px frame. Touchdown is at ``touchdown_height``.
#: * the v11 marker board (config/spatial-isaac-system-v11.yaml): 17 ArUco
#:   tags at three scales, 0.26 m corners for approach, 0.14 m mid-ring, and
#:   0.08/0.05 m centre cluster for the last 0.3 m, in place of five.
#:
#: Packet, graph, ontology, reward and every capability are identical to /3,
#: so /1-/3 checkpoints load under their own schema; only ``config_sha256``
#: and the deployment profile differ.
#: /5 is /4 plus one law, the OPTICAL REALISM of the local sensor: the pad is
#: detected when at least one tag of the deployed board (read from the
#: deployment profile) projects fully inside the frame at a decodable size,
#: and ``detectionConfidence`` is the detector's own quality metric -- the
#: largest seen tag's pixel side over 120 px, times a sharpness term, times
#: 0.85 when a single tag braces the solve -- with a 5 % frame-miss rate.
#: Measured 2026-10-06 on the third Isaac flight (12 episodes, 0 landings, 0
#: unsafe, detection 100 % for 70 s, estimate error 5-13 cm): the one packet
#: channel that differed between Isaac and the local replay of the SAME
#: checkpoint on the SAME seeds by more than its own spread was
#: ``detectionConfidence``, a constant 0.98 locally and 0.22-0.58 in Isaac,
#: falling with camera depth exactly as the tag-size formula predicts (0.22 at
#: 1.8-2.6 m, 0.45 at 0.8-1.1 m, 0.58 at 0.2-0.5 m). A channel that never
#: varied in training is an untrained direction of the policy, and in Isaac
#: it moved every step; the policy's lateral commands were 2.5x larger and 5x
#: jerkier than locally, the tilt stayed above the 5 deg corridor gate and
#: the supervisor held the descent for 70 s. The estimator never reads the
#: confidence, so this law changes what the policy sees and nothing else.
#: /6 is /5 plus one law, the attitude servo's DAMPING. Identified 2026-10-06
#: from the tilt traces of all four Isaac flights (46 episodes, 1360 steps):
#: simulating the local second-order servo on the recorded commanded attitude
#: and minimising the RMS tilt error gives a plateau at omega 9-10 rad/s,
#: zeta 0.65-0.7, 0.10-0.12 s input delay (2.43 deg) against 2.91 deg for the
#: /4-/5 servo (14 rad/s, zeta 1) and 2.73 deg for /3's (10, 1). The velocity
#: reversal estimator (225 events) ranks the same (10, 0.7) a close second to
#: (14, 1) -- error 0.080 against 0.035 -- so both estimators admit it, and
#: only zeta < 1 reproduces what ended the fourth flight's one envelope
#: violation: a command reversal from +8.5 to -15 deg of roll that the PX4
#: vehicle overshot to -21.2 deg at 107 deg/s. A critically damped local
#: servo never overshoots, so no policy trained on it could learn that
#: margin. Steady-state tilt gain is 0.90-1.02, i.e. there is no gain error
#: to model. Packet, graph, reward and capabilities are identical to /5.
#: /7 is /6 plus one law, the CONTACT VERDICT. Decided by the user on
#: 2026-10-06 over the six non-landing contacts of the five-seed /6 Isaac
#: flight (results/full_pipeline_20261006_v6_5seeds): four were vehicles that
#: sank onto the pad from a low hover while the supervisor held the vertical
#: rate (``vertical_stopping_margin``), touching at |vz| 0.01-0.07 m/s, and
#: were scored UNAUTHORIZED_CONTACT (two, every touchdown limit met) or
#: UNSAFE_CONTACT (two: body rate 10.47 deg/s; tilt 5.27 deg at 12.51 deg/s,
#: against the 5 deg / 10 deg/s touchdown band, both at |vz| <= 0.07). Two
#: touched inside the terminal-descent corridor at vz -0.62 and -0.38 m/s
#: against the 0.30 m/s limit. The user ruled the four acceptable landings
#: and the two not. The one rule that draws exactly that line is the IMPACT
#: SPEED: on /7 a pad contact is UNSAFE_CONTACT when its horizontal or
#: vertical speed exceeds the touchdown limits, SAFETY_ENVELOPE_VIOLATION
#: when the thrust axis is past the 21 deg hard limit at contact,
#: UNAUTHORIZED_CONTACT only while the abort hold owns the vehicle, and
#: SUCCESS otherwise -- whether or not descent was inhibited, whatever the
#: tilt and body rate inside the envelope. Attitude and rate remain the
#: corridor's ``settled`` commit condition and the readiness term's price;
#: they stop being contact criteria. The limit VALUES are unchanged. Plant,
#: packet, graph, ontology, supervisor and terminal table are identical to
#: /6 (guarded in tests/test_contact_verdict_rung.py). Re-scoring the
#: recorded /6 flight under this rule gives 27 SUCCESS / 2 UNSAFE_CONTACT /
#: 1 SAFE_ABORT; acceptance still forbids the two.
#: /8 is /7 MINUS one law: the /4 landing gear is removed, on the user's
#: instruction of 2026-10-06 late evening ("remove drone legs again"). The
#: body touches down at the stock Iris height again (``touchdown_height`` =
#: ``contact_height`` = 0.12 m), the supervisor's gate, corridor and stopping
#: margin shift down with it (they are heights above the stock-gear
#: touchdown), the curriculum's low start band follows, and the deployment
#: profile is v12 = v11 with ``vehicle.landing_gear.extension_m: 0`` so the
#: Isaac world attaches no leg colliders. The 17-tag board, optical realism,
#: the delayed /6 servo and the /7 contact verdict are all inherited. What
#: this gives back is the geometry /4 was built to avoid: the camera sits
#: 0.16 m below the body, so at contact it is 0.04 m BELOW the pad plane and
#: the last centimetres are flown on the estimator's memory (``terminal_coast``
#: and the capture-time-aligned track), not on a detection. Whether the
#: centre cluster of the v11 board and the /5 optical model carry that
#: interval is what the /8 run measures; it was not measured before the legs
#: were added.
REFERENCE_SCHEMAS = ("spatial-reference/1", "spatial-reference/2",
                     "spatial-reference/3", "spatial-reference/4",
                     "spatial-reference/5", "spatial-reference/6",
                     "spatial-reference/7", "spatial-reference/8")
REFERENCE_SCHEMA = REFERENCE_SCHEMAS[-1]
#: /6 attitude servo: natural frequency (rad/s) and damping ratio.
REFERENCE_ATTITUDE_SERVO_6 = (10.0, 0.7)
#: Isaac detector quality constants (isaac_sim/marker_vision.py) and the
#: sharpness the third flight measured (reprojection ~1.0-1.2 px of 3 px).
REFERENCE_OPTICAL_FULL_SCALE_PX = 120.0
REFERENCE_OPTICAL_SHARPNESS = (0.65, 0.08)
REFERENCE_OPTICAL_MIN_TAG_PX = 12.0
REFERENCE_OPTICAL_MISS_PROBABILITY = 0.05
REFERENCE_OPTICAL_IMAGE_PX = (640, 480)
#: Isaac/PX4 command-to-response latency the /3 and /4 plants reproduce, s.
REFERENCE_ACTUATION_DELAY_S = 0.10
#: /4 attitude servo bandwidth, rad/s (the /1-/3 plant uses dynamics.ATTITUDE_OMEGA).
REFERENCE_ATTITUDE_OMEGA_RAD_S = 14.0
#: /4 landing-gear extension below the stock Iris gear, metres.
REFERENCE_LANDING_GEAR_EXTENSION_M = 0.18


def schema_for(contract_version):
    """CLI contract selector -> schema string.

    ``reference`` is the active, derived contract; the bare numbers are the
    frozen historical rungs, kept so their checkpoints stay loadable. A
    superseded reference rung is selectable by its full schema string, which is
    how ``spatial-reference/1`` -- the one-sided vertical brake -- stays
    reachable for reproducing anything trained before 2026-10-05.
    """
    version = str(contract_version)
    if version == "reference":
        return REFERENCE_SCHEMA
    if version in REFERENCE_SCHEMAS:
        return version
    return f"spatial-causal-rgat/{version}"


FIELDS = tuple(
    "rx ry rz rvx rvy rvz pvx pvy pvz pax pay paz vx vy vz qw qx qy qz wx wy wz bx by mx my detected confidence age initialized posstd velstd remaining inhibited abort".split()
)
SCALES = np.array(
    [
        3,
        3,
        4,
        3,
        3,
        2,
        4,
        4,
        2,
        2,
        2,
        2,
        4,
        4,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0.785,
        0.644,
        0.785,
        0.644,
        1,
        1,
        3,
        1,
        1,
        1,
        1,
        1,
        1.0,
    ]
)
DIRECT = np.array(
    [
        FIELDS.index(k)
        for k in "qw qx qy qz detected confidence initialized remaining inhibited abort".split()
    ]
)
LEGACY_REGISTRY_HASH = hashlib.sha256(
    json.dumps(
        {
            "schema": "spatial-causal-packet/1",
            "fields": FIELDS,
            "scales": SCALES.tolist(),
            "normalization": "smooth-signed",
            "relative_convention": "uav-minus-pad-ENU",
            "graph": "spatial-9x12/1",
            "topology": TOPOLOGY_HASH,
        },
        sort_keys=True,
    ).encode()
).hexdigest()
REGISTRY_HASH = hashlib.sha256(
    json.dumps(
        {
            "previous_registry": LEGACY_REGISTRY_HASH,
            "schema": "spatial-causal-packet/2",
            "bearings": "actual-attitude optical frame, body mount [0,0,-0.16]",
        },
        sort_keys=True,
    ).encode()
).hexdigest()
FIELDS_V6 = FIELDS + ("pred_bx", "pred_by", "pred_mx", "pred_my")
REGISTRY_HASH_V6 = hashlib.sha256(json.dumps({
    "schema": "spatial-causal-packet/3", "previous_registry": REGISTRY_HASH,
    "fields": FIELDS_V6, "prediction": "0.5s relative translation and own body-rate attitude",
    "graph": "spatial-9x12/2-predicted-view", "topology": TOPOLOGY_HASH,
}, sort_keys=True).encode()).hexdigest()
FIELDS_V7 = FIELDS_V6 + ("accstd", "previous_ax", "previous_ay", "previous_az")
REGISTRY_HASH_V7 = hashlib.sha256(json.dumps({
    "schema": "spatial-causal-packet/4", "previous_registry": REGISTRY_HASH_V6,
    "fields": FIELDS_V7, "tracker": "absolute-pad-abg-reference-v28-20Hz-camera",
    "graph": "spatial-9x12/3-acceleration-uncertainty", "topology": TOPOLOGY_HASH,
}, sort_keys=True).encode()).hexdigest()
FIELDS_V9 = FIELDS_V7 + ('meas_bx','meas_by','optical_valid','optical_age')
REGISTRY_HASH_V9 = hashlib.sha256(json.dumps({
    'schema': 'spatial-causal-packet/5', 'previous_registry': REGISTRY_HASH_V7,
    'fields': FIELDS_V9, 'graph': 'two-ENU-planes-each-reference-9x12',
    'topology': TOPOLOGY_HASH, 'descent': 'joint-3D-reference-risk-product',
    'optical': 'unique-capture-time-with-own-state-alignment',
    'local_camera_transport_delay_s': .075,
    'trajectory': 'shared-seeded-spatial-reference-CV-CA-CV',
    'attitude_context': 'ENU-thrust-axis-tilt-and-derivative',
}, sort_keys=True).encode()).hexdigest()
REGISTRY_HASH_V10 = hashlib.sha256(json.dumps({
    'previous_registry': REGISTRY_HASH_V9,
    'supervisor': 'reference-trust-latch-delay-stopping-corridor/1',
    'persistent_disturbance_abort': 'own-EKF-position-anchor-full-xyz-override',
    'reward_view': 'captured-optical-bearing-not-estimated-bearing',
    'attitude_limit': '20deg-command-21deg-hard-envelope',
    'stopping_brake': 'net-az-limit-minus-sampled-force-norm-bound',
}, sort_keys=True).encode()).hexdigest()


def camera_bearings(relative_enu, quaternion_wxyz, cfg):
    """Causal optical bearings/depth; no simulator or target-truth access.

    Caller supplies either an estimate (policy/reward view term) or physical
    pose (sensor generation only). These uses never share an input object.
    """
    r = np.asarray(relative_enu)
    if cfg.schema == "spatial-causal-rgat/3":
        depth = float(r[2] - 0.16)
        return np.arctan2(r[:2], max(depth, 0.01)), depth
    from scipy.spatial.transform import Rotation

    q = np.asarray(quaternion_wxyz)
    body_to_pad = Rotation.from_quat(q[[1, 2, 3, 0]]).inv().apply(-r)
    optical = (body_to_pad - np.array([0, 0, -0.16])) * np.array([1, -1, -1])
    return np.arctan2(optical[:2], optical[2]), float(optical[2])


@dataclass(frozen=True)
class SpatialConfig:
    schema: str = REFERENCE_SCHEMA
    isaac_profile_sha256: str = ""
    dt: float = 0.1
    sensor_dt: float = 0.01
    camera_dt: float = 0.05
    horizon: float = REFERENCE_MISSION_HORIZON_S
    # Upstream's value, shared with the 2D route. It was briefly 350 on the
    # argument that tau == horizon makes waiting dominant; measurement showed
    # attempting wins under both settings (landing/terminal.py), and the
    # divergence only cost 2D/3D comparability.
    discount_tau: float = REFERENCE_DISCOUNT_TAU_S
    max_velocity: tuple = (10.0, 10.0, 5.0)
    max_acceleration: tuple = (2.5, 2.5, 2.0)
    pad_half_width: float = 0.5
    contact_height: float = 0.12
    touchdown_xy_speed: float = 0.35
    touchdown_z_speed: float = 0.3
    touchdown_tilt: float = math.radians(5)
    touchdown_rate: float = math.radians(10)
    fov: tuple = (math.radians(90), 2 * math.atan(0.75))
    loss_timeout: float = 3.0
    recovery_timeout: float = 8.0
    initial_height: tuple = (2.0, 2.5)
    initial_xy: float = 0.4
    level_ground_pad: bool = True
    curriculum: CurriculumConfig = field(
        default_factory=lambda: CurriculumConfig(
            start_v1_range_m_s=(0.045, 0.12),
            start_a2_range_m_s2=(0.03, 0.12),
            start_height_range_m=(0.2, 0.5),
            start_touchdown_vertical_speed_m_s=0.6,
            # Upstream primaryConfig.m. validate_curriculum_ramp pins it below
            # SAFE_ABORT at every rung so the easy end cannot make crashing
            # cheaper than aborting -- two_axis measured 92-98 % unsafe contact
            # when that inverted.
            start_unsafe_contact_penalty=REFERENCE_CURRICULUM_START_UNSAFE,
            promotion_landing_rate=0.1,
            promotion_window_episodes=18,
            scheduled_floor=True,
        )
    )
    ontology: GraphConfig = field(
        default_factory=lambda: GraphConfig(
            schema="compact_context_graph_v3_grouped",
            hidden_dimension=8,
            policy_hidden_dimension=48,
            relation_dimension=4,
            freeze_static_backbone=True,
            preserve_raw_during_adaptation=True,
            adaptation_warmup_fraction=0.9,
            pretrain_episodes=4,
            pretrain_decisions=32,
            pretrain_epochs=4,
            selection_score_model="outcome_weighted_v28",
            selection_score_margin=5.0,
            relation_activation_enabled=True,
        )
    )

    def __post_init__(self):
        if (
            self.schema not in ("spatial-causal-rgat/3", "spatial-causal-rgat/4",
                                "spatial-causal-rgat/5", "spatial-causal-rgat/6", "spatial-causal-rgat/7", "spatial-causal-rgat/8", "spatial-causal-rgat/9", "spatial-causal-rgat/10",
                                *REFERENCE_SCHEMAS)
            or not 0 < self.sensor_dt <= self.dt <= 0.2
        ):
            raise ValueError("invalid spatial schema/timing")
        if self.horizon <= 0 or self.discount_tau <= 0:
            raise ValueError("positive spatial horizons required")
        # The 3D route carried its own terminal table and ramp for a week and
        # drifted away from the 2D reference unnoticed. These are the same
        # load-time assertions the 2D config runs, on the same definitions.
        validate_terminal_ordering(REFERENCE_TERMINAL_REWARDS)
        validate_curriculum_ramp(
            self.curriculum.start_unsafe_contact_penalty, REFERENCE_TERMINAL_REWARDS)
        validate_discount_horizon(self.discount_tau, self.horizon)
        if not self.sensor_dt <= self.camera_dt <= 0.2:
            raise ValueError("camera sampling must match the spatial sensor profile")

    @property
    def signature(self):
        return {
            "schema": self.schema,
            "packet_registry": self.registry_hash,
            "action": ("spatial-enu-direct-net-acceleration-v2" if self.direct_acceleration
                       else "spatial-enu-net-acceleration-v1"),
            "config_sha256": hashlib.sha256(
                json.dumps(asdict(self), sort_keys=True).encode()
            ).hexdigest(),
        }

    @property
    def axes(self):
        """Horizontal axes of the task. The dimension IS this tuple."""
        return SPATIAL_AXES

    @property
    def registry_hash(self):
        if self.schema in REFERENCE_SCHEMAS:
            return reference_registry(self.axes, extras=True)["sha256"]
        if self.reference_supervisor:
            return REGISTRY_HASH_V10
        if self.reference_context:
            return REGISTRY_HASH_V9
        if self.reference_tracking:
            return REGISTRY_HASH_V7
        if self.direct_acceleration:
            return REGISTRY_HASH_V6
        return (
            LEGACY_REGISTRY_HASH
            if self.schema == "spatial-causal-rgat/3"
            else REGISTRY_HASH
        )

    @property
    def direct_acceleration(self):
        return self.schema in ("spatial-causal-rgat/6", "spatial-causal-rgat/7", "spatial-causal-rgat/8", "spatial-causal-rgat/9", "spatial-causal-rgat/10", *REFERENCE_SCHEMAS)

    @property
    def reference_tracking(self):
        return self.schema in ("spatial-causal-rgat/7", "spatial-causal-rgat/8", "spatial-causal-rgat/9", "spatial-causal-rgat/10", *REFERENCE_SCHEMAS)

    @property
    def matched_disturbances(self):
        return self.schema in ("spatial-causal-rgat/8", "spatial-causal-rgat/9", "spatial-causal-rgat/10", *REFERENCE_SCHEMAS)

    @property
    def reference_context(self):
        return self.schema in ("spatial-causal-rgat/9", "spatial-causal-rgat/10", *REFERENCE_SCHEMAS)

    @property
    def reference_supervisor(self):
        return self.schema in ("spatial-causal-rgat/10", *REFERENCE_SCHEMAS)

    @property
    def vertical_inhibit_holds(self):
        """Does the inhibited supervisor HOLD the vertical rate, or only brake a descent?

        /1 and every ladder rung rewrite the vertical command only when it is
        negative: a descending vehicle is braked with up to full upward
        authority, an ascending one passes through untouched. The
        direct-acceleration plant has no restoring force on altitude, so that
        asymmetry is a ratchet. Measured on a fresh policy at the easiest
        curriculum rung, a requested -0.065 m/s^2 came out as an applied
        +0.082, the median climb was +19.71 m and 27 of 40 episodes ended on
        the 20 m ceiling; holding the rate instead gives +0.70 m and 0 of 40.

        The swept oracle never meets this -- it is inside the terminal-descent
        corridor on 64 % of its steps -- so no ceiling measurement can see it,
        which is why it survived a 96-cell gain sweep.
        """
        return self.schema in ("spatial-reference/2", "spatial-reference/3",
                               "spatial-reference/4", "spatial-reference/5",
                               "spatial-reference/6", "spatial-reference/7",
                               "spatial-reference/8")

    @property
    def actuation_delay_s(self):
        """Command-to-response latency the local plant reproduces (/3 and /4).

        Zero on every earlier rung keeps them byte-identical. The supervisor
        reads the same value for its stopping margin, so the safety model and
        the plant it protects carry one latency, not two.
        """
        return (REFERENCE_ACTUATION_DELAY_S
                if self.schema in ("spatial-reference/3", "spatial-reference/4",
                                   "spatial-reference/5", "spatial-reference/6",
                                   "spatial-reference/7", "spatial-reference/8") else 0.0)

    @property
    def attitude_omega(self):
        """Attitude servo bandwidth of the local plant, rad/s (/4: Isaac-fitted)."""
        from .dynamics import ATTITUDE_OMEGA
        if self.schema in ("spatial-reference/6", "spatial-reference/7",
                           "spatial-reference/8"):
            return REFERENCE_ATTITUDE_SERVO_6[0]
        return (REFERENCE_ATTITUDE_OMEGA_RAD_S
                if self.schema in ("spatial-reference/4", "spatial-reference/5")
                else ATTITUDE_OMEGA)

    @property
    def attitude_damping(self):
        """Attitude servo damping ratio; critically damped before /6."""
        from .dynamics import ATTITUDE_DAMPING
        return (REFERENCE_ATTITUDE_SERVO_6[1]
                if self.schema in ("spatial-reference/6", "spatial-reference/7",
                                   "spatial-reference/8")
                else ATTITUDE_DAMPING)

    @property
    def landing_gear_extension_m(self):
        """How far below the stock gear the /4 legs reach; zero elsewhere.

        /4 through /7 carry the legs; /8 removes them again (user decision,
        see the /8 note above REFERENCE_SCHEMAS), so the stock Iris contact
        height is back and everything derived from this value follows.
        """
        return (REFERENCE_LANDING_GEAR_EXTENSION_M
                if self.schema in ("spatial-reference/4", "spatial-reference/5",
                                   "spatial-reference/6", "spatial-reference/7") else 0.0)

    @property
    def easy_start_lift_m(self):
        """How far above ``curriculum.start_height_range_m`` the easy spawn
        band sits, so the camera can see the pad at the band's low end.

        The camera is mounted 0.16 m under the body and, from /5 on, detection
        is tag-based: at the band's stock low end (body 0.20 m) the camera is
        0.04 m above the deck and no tag's padded quad fits the frame, so the
        estimator never initialises, the supervisor latches ``prolonged_visual
        _loss`` at the first step and climbs. Measured 2026-10-06 on /8 with
        the fixed-descent reachability probe: 5 of 12 easy-rung seeds ended
        SAFE_ABORT where /7 landed 12 of 12. On /4-/7 the legs lifted the band
        (body 0.38-0.68 m, camera 0.22-0.52 m) as a side effect; /8 removes
        the legs and keeps the lift, so the easy rung stays the same camera
        geometry it was trained on. Zero on /1-/3, whose optical model was not
        tag-based. Training-only: difficulty 1.0 never reads this.
        """
        if self.schema in ("spatial-reference/4", "spatial-reference/5",
                           "spatial-reference/6", "spatial-reference/7",
                           "spatial-reference/8"):
            return REFERENCE_LANDING_GEAR_EXTENSION_M
        return self.landing_gear_extension_m

    @property
    def optical_realism(self):
        """Tag-based detection and the detector's quality metric (/5 onward)."""
        return self.schema in ("spatial-reference/5", "spatial-reference/6",
                               "spatial-reference/7", "spatial-reference/8")

    @property
    def terminal_descent_speed_factor(self):
        """Descent the supervisor admits inside the terminal corridor, as a
        multiple of ``touchdown_z_speed``; it brakes only above this.

        1.5 is the reference approach margin (two_axis/safety.py) and is what
        every rung through /8 carries. The contact verdict needs |vz| <=
        1.0 x, so the band between 1.0 x and 1.5 x is descent the supervisor
        allows and the verdict then scores UNSAFE_CONTACT: on /7 and /8 every
        held-out unsafe terminal (74 of 720 and the /8 set) was a contact in
        that band, most of it within 0.05 m/s of the limit, and the five
        Isaac UNSAFE_CONTACTs of the /8 flight touched at -0.30 to -0.32.
        """
        return 1.5

    @property
    def contact_verdict_by_speed(self):
        """/7: a pad contact is judged by its impact speed.

        Not by whether descent was inhibited, and not by tilt or body rate
        inside the 21 deg envelope -- those stay the corridor's commit
        condition. See the /7 note above REFERENCE_SCHEMAS for the six Isaac
        contacts this rule was drawn from.
        """
        return self.schema in ("spatial-reference/7", "spatial-reference/8")

    @property
    def touchdown_height(self):
        """Body height above the pad at physical contact.

        ``contact_height`` stays the stock-gear figure (and stays in the
        config hash of every rung); the gear extension adds to it on /4.
        """
        return self.contact_height + self.landing_gear_extension_m

    @property
    def camera_transport_delay(self):
        # v9 capture diagnostic: detected-frame median age 96 ms at decisions.
        # 75 ms delivery delay + 20 Hz sampling approximates that cadence;
        # this is not a claim to reproduce PnP image/noise/dropout distributions.
        return .075 if self.reference_context else 0.

    @property
    def packet_fields(self):
        if self.schema in REFERENCE_SCHEMAS:
            return reference_field_names(self.axes, extras=True)
        if self.reference_context:
            return FIELDS_V9
        if self.reference_tracking:
            return FIELDS_V7
        return FIELDS_V6 if self.direct_acceleration else FIELDS


@dataclass(frozen=True)
class Measurement:
    time_s: float
    own_position: np.ndarray
    own_velocity: np.ndarray
    quaternion: np.ndarray
    angular_rate: np.ndarray
    optical_position: np.ndarray | None
    confidence: float
    sample_id: int
    optical_time_s: float | None = None
    optical_own_position: np.ndarray | None = None
    optical_quaternion: np.ndarray | None = None

    def __post_init__(self):
        for value, size in (
            (self.own_position, 3),
            (self.own_velocity, 3),
            (self.quaternion, 4),
            (self.angular_rate, 3),
        ):
            if np.shape(value) != (size,) or not np.isfinite(value).all():
                raise ValueError("nonfinite spatial own measurement")
        if not math.isfinite(self.time_s) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid measurement time/confidence")
        if self.optical_position is not None and (
            np.shape(self.optical_position) != (3,)
            or not np.isfinite(self.optical_position).all()
        ):
            raise ValueError("invalid optical solve")
        if self.optical_time_s is not None and (
            not math.isfinite(self.optical_time_s) or self.optical_time_s > self.time_s + .02
        ):
            raise ValueError('invalid/future optical capture time')
        if self.optical_own_position is not None and (
            np.shape(self.optical_own_position) != (3,) or not np.isfinite(self.optical_own_position).all()
        ):
            raise ValueError('invalid capture-aligned own position')
        if self.optical_quaternion is not None and (
            np.shape(self.optical_quaternion) != (4,) or not np.isfinite(self.optical_quaternion).all()
            or np.linalg.norm(self.optical_quaternion) < 1e-9
        ):
            raise ValueError('invalid capture-aligned own attitude')

    @classmethod
    def from_wire(cls, state, *, capture_aligned=False):
        # Deliberate allowlist: never pass a whole wire state to the estimator.
        extra = state.get("extra", {})
        clock, optical = extra.get("spatial_clock", {}), extra.get(
            "optical_measurement", {}
        )
        if not clock.get("valid") or not state.get("estimator_valid"):
            raise ValueError("valid Isaac clock and own EKF are required")
        valid = optical.get("valid") and optical.get("frame") == "pad_enu"
        if valid and capture_aligned and any(optical.get(name) is None for name in (
            'capture_time_s','own_position_at_capture_m','own_quaternion_at_capture_wxyz'
        )):
            raise ValueError('v9 requires capture-aligned optical telemetry')
        capture = optical.get('capture_time_s')
        if capture is not None and capture > float(clock['sim_time_s']):
            # Independent DDS topics may deliver the frame before its clock.
            # Defer it until that time, never treat it as a future observation.
            valid, capture = False, None
        return cls(
            float(clock["sim_time_s"]),
            np.asarray(state["world"]["position"]),
            np.asarray(state["world"]["velocity"]),
            np.asarray(state["quaternion_wxyz"]),
            np.asarray(state["angular_velocity"]),
            np.asarray(optical["position_m"]) if valid else None,
            float(optical.get("confidence", 0)) if valid else 0.0,
            int(optical.get("sample_id", -1)),
            capture,
            (np.asarray(optical['own_position_at_capture_m'])
             if optical.get('own_position_at_capture_m') is not None else None),
            (np.asarray(optical['own_quaternion_at_capture_wxyz'])
             if optical.get('own_quaternion_at_capture_wxyz') is not None else None),
        )


class Estimator:
    def __init__(self, config):
        self.cfg = config
        self.initialized = False
        self.r = np.zeros(3)
        self.pad_v = np.zeros(3)
        self.pad_a = np.zeros(3)
        self.rv = np.zeros(3)
        self.last_t = None
        self.last_detection = -math.inf
        self.last_sample = -1
        self.std = 10.0
        self.velocity_std = 5.0
        self.updates = 0
        self.own = None
        self.detected = False
        self.last_optical_r = None
        self.last_optical_rv = None
        self.previous_action = np.zeros(3)
        # Causal unmodelled-acceleration observer. 3D adds a per-episode
        # external force of up to 0.75 N on each axis (0.5 m/s^2 against a
        # 2.5 m/s^2 authority) that 2D has no counterpart for. It is observable
        # without any truth: difference own EKF velocity and subtract the
        # vehicle's OWN commanded acceleration. Both are own state.
        self.disturbance = np.zeros(3)
        self.previous_applied_acceleration = np.zeros(3)
        self._disturbance_velocity = None
        self._disturbance_time = None
        if config.reference_tracking:
            from .estimation import ReferenceSpatialTrack
            self.reference_track = ReferenceSpatialTrack(capture_aligned=config.reference_context)
            self.acceleration_std = 2.

    def update(self, measurement):
        if not isinstance(measurement, Measurement):
            raise TypeError("only allowlisted Measurement may enter estimator")
        if self.cfg.reference_tracking:
            return self._update_reference(measurement)
        m = measurement
        if self.last_t is not None and m.time_s < self.last_t - 1e-8:
            raise ValueError("simulation clock reversed; reset the episode")
        dt = 0.0 if self.last_t is None else m.time_s - self.last_t
        if dt > 2.0:
            raise ValueError("sensor gap exceeds 2 simulated seconds")
        # Propagate at 100 Hz using ONLY the preceding own measurement.
        # Never interpolate a future measurement backwards into history.
        remaining = dt
        previous_v = m.own_velocity if self.own is None else self.own.own_velocity
        while remaining > 1e-9:
            h = min(remaining, self.cfg.sensor_dt)
            self.r += (previous_v - self.pad_v) * h - 0.5 * self.pad_a * h * h
            self.pad_v += self.pad_a * h
            if self.cfg.schema in ("spatial-causal-rgat/5", "spatial-causal-rgat/6"):
                # Reference updatePadTrack.m's 1.5 s acceleration decay. An
                # old noisy acceleration cannot drive unbounded target-speed
                # growth during an optical outage. No target truth is used.
                self.pad_a *= math.exp(-h / 1.5)
            self.std = min(10.0, self.std + 0.15 * h)
            self.velocity_std = min(5.0, self.velocity_std + 0.3 * h)
            remaining -= h
            self.updates += 1
        fresh = m.optical_position is not None and m.sample_id != self.last_sample
        self.detected = bool(m.optical_position is not None)
        if fresh:
            if not self.initialized:
                self.r = m.optical_position.copy()
                self.pad_v = m.own_velocity.copy()
                self.pad_a[:] = 0.0
                self.initialized = True
            else:
                elapsed = max(m.time_s - self.last_detection, self.cfg.sensor_dt)
                innovation = m.optical_position - self.r
                # Reject impossible PnP branch jumps; do not reset uncertainty.
                if np.linalg.norm(innovation) > max(1.0, 4.0 * elapsed):
                    fresh = False
                else:
                    self.r += 0.65 * innovation
                    self.pad_v -= 0.12 * innovation / elapsed
                    self.pad_a -= 0.005 * innovation / (elapsed * elapsed)
                    self.pad_v = np.clip(self.pad_v, -10, 10)
                    self.pad_a = np.clip(self.pad_a, -2, 2)
            if fresh:
                self.last_detection = m.time_s
                self.std = 0.03
                self.velocity_std = 0.15
            self.last_sample = m.sample_id
        self.rv = m.own_velocity - self.pad_v
        if self.cfg.level_ground_pad:
            # Known UGV kinematics, not the simulator's current state: a
            # ground carrier on a level plane cannot acquire vertical speed
            # from noisy camera innovations and coast upward during a miss.
            self.pad_v[2] = 0.0
            self.pad_a[2] = 0.0
            self.rv[2] = m.own_velocity[2]
        if fresh:
            self.last_optical_r = self.r.copy()
            self.last_optical_rv = self.rv.copy()
        self.last_t = m.time_s
        self.own = m

    def _update_reference(self, m):
        track = self.reference_track
        accepted = track.update(m, level_ground=self.cfg.level_ground_pad)
        self.initialized = track.initialized
        self.r = m.own_position-track.position if track.initialized else np.zeros(3)
        self.pad_v = track.velocity.copy()
        self.pad_a = track.acceleration.copy()
        self.rv = m.own_velocity-self.pad_v
        self.std, self.velocity_std, self.acceleration_std = np.max(track.std, axis=1)
        self.last_t, self.last_detection, self.last_sample = track.last_t, track.last_detection, track.last_sample
        self.detected = bool(m.optical_position is not None)
        self._update_disturbance(m)
        self.own = m
        self.updates += 1
        if accepted:
            self.last_optical_r = self.r.copy()
            self.last_optical_rv = self.rv.copy()

    DISTURBANCE_TIME_CONSTANT_S = 1.5

    def observe_actuation(self, applied_acceleration_m_s2):
        """Record the vehicle's own commanded acceleration for the observer.

        Own commanded state, never simulator truth and never the supervisor's
        private reasoning. Called once per policy step by the environment.
        """
        value = np.asarray(applied_acceleration_m_s2, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("applied acceleration must be a finite ENU triple")
        self.previous_applied_acceleration = value.copy()

    def _update_disturbance(self, m):
        """Low-pass the residual between realized and commanded acceleration.

        The attitude/thrust lag shows up here too, but it is transient; the
        external force is constant within an episode, so the filter separates
        them. Estimating it is what lets a policy trade authority against a
        standing bias instead of re-discovering it every step.
        """
        if self._disturbance_velocity is not None and self._disturbance_time is not None:
            dt = m.time_s - self._disturbance_time
            if 1e-6 < dt <= 2.0:
                realized = (m.own_velocity - self._disturbance_velocity) / dt
                residual = realized - self.previous_applied_acceleration
                blend = 1.0 - math.exp(-dt / self.DISTURBANCE_TIME_CONSTANT_S)
                self.disturbance += blend * (residual - self.disturbance)
                self.disturbance = np.clip(self.disturbance, -5.0, 5.0)
        self._disturbance_velocity = np.asarray(m.own_velocity, dtype=float).copy()
        self._disturbance_time = float(m.time_s)

    @property
    def age(self):
        return (
            1e3 if not self.initialized else max(0.0, self.last_t - self.last_detection)
        )


@dataclass(frozen=True)
class Safety:
    inhibited: bool
    abort: bool
    eligible: float
    reasons: tuple


def safety_status(est, cfg):
    if cfg.reference_supervisor:
        raise ValueError('v10 requires the episode-owned ReferenceSpatialSupervisor')
    lateral = float(np.linalg.norm(est.r[:2]))
    speed = float(np.linalg.norm(est.rv[:2]))
    from ..mathx import quat_to_euler_zyx

    angles = quat_to_euler_zyx(est.own.quaternion)
    tilt = float(np.linalg.norm(angles[:2]))
    # Shrinking footprint admits a physically safe corridor all the way to
    # contact; the abort hold does not replace authorized terminal descent.
    footprint = max(
        cfg.pad_half_width * 0.7,
        max(est.r[2], 0) * min(np.tan(np.asarray(cfg.fov) / 2)) * 0.65,
    )
    reasons = []
    last_r, last_v = est.last_optical_r, est.last_optical_rv
    terminal_coast = bool(
        last_r is not None
        and last_v is not None
        and 0 < last_r[2] <= 0.45
        and np.linalg.norm(last_r[:2]) <= 0.15
        and np.linalg.norm(last_v[:2]) <= 0.3
        and -0.35 <= last_v[2] <= 0.05
        and 0 < est.r[2] <= 0.45
        and est.age <= 1.5
        and lateral + 2 * est.std <= cfg.pad_half_width
        and speed <= cfg.touchdown_xy_speed
        and tilt <= cfg.touchdown_tilt
    )
    if not est.initialized or (est.age > 0.5 and not terminal_coast):
        reasons.append("visual_track_unavailable")
    if lateral > footprint:
        reasons.append("outside_visible_corridor")
    if tilt > math.radians(20):
        reasons.append("tilt")
    if est.r[2] < 1.0 and (lateral > cfg.pad_half_width or speed > 0.6):
        reasons.append("terminal_alignment")
    abort = est.age > cfg.loss_timeout or lateral > 60 or est.r[2] > 20
    eligible = float(
        np.clip(1 - lateral / max(footprint, 1e-6), 0, 1)
        * np.clip(1 - speed / 3.0, 0, 1)
        * (not reasons)
    )
    return Safety(bool(reasons), bool(abort), eligible, tuple(reasons))


def supervised_action(action, est, safety, cfg):
    if cfg.reference_supervisor:
        raise ValueError('v10 requires the episode-owned ReferenceSpatialSupervisor')
    action = np.asarray(action, dtype=float)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("finite spatial action required")
    applied = np.clip(action, -1, 1).copy()
    if safety.inhibited or safety.abort:
        # Hold/recover vertically. No privileged steering or PN teacher.
        target = max(1.0, est.r[2]) if safety.abort else max(0.12, est.r[2])
        az = np.clip(
            1.5 * (target - est.r[2]) - 1.8 * est.rv[2],
            -cfg.max_acceleration[2],
            cfg.max_acceleration[2],
        )
        applied[2] = max(applied[2], az / cfg.max_acceleration[2])
    if safety.abort and (cfg.schema == "spatial-causal-rgat/5" or cfg.direct_acceleration):
        # Reference abort_braking_hold, extended symmetrically to ENU x/y.
        # Own measured velocity is trustworthy even when target prediction is
        # not. Never chase an extrapolated, unobserved pad during an abort.
        applied[:2] = np.clip(-1.5 * est.own.own_velocity[:2] /
                             np.asarray(cfg.max_acceleration[:2]), -1., 1.)
    # Common causal braking envelope; it does not make unsafe contacts safe.
    height = max(est.r[2] - cfg.touchdown_height, 0.0)
    safe_sink = math.sqrt(cfg.touchdown_z_speed ** 2 + 2.0 * 0.7 * height)
    if est.rv[2] < -safe_sink:
        applied[2] = max(
            applied[2],
            min(1.0, (-safe_sink - est.rv[2]) / (cfg.dt * cfg.max_acceleration[2])),
        )
    return applied


@dataclass(frozen=True)
class Packet:
    values: np.ndarray
    registry_sha256: str = REGISTRY_HASH


@dataclass(frozen=True)
class Graph:
    X: np.ndarray
    schema_hash: str = REGISTRY_HASH


@dataclass(frozen=True)
class Observation:
    packet: Packet
    graph: Graph


def observation(est, safety, elapsed, cfg):
    if cfg.schema in REFERENCE_SCHEMAS:
        # One derived field list, one normalization, one plane builder per
        # axis -- see spatial/reference_contract.py and landing/packet.py.
        from .context import axis_context_graphs
        from .reference_contract import packet_values

        values = packet_values(est, safety, elapsed, cfg)
        X = axis_context_graphs(values, cfg)
        if not np.isfinite(values).all() or not np.isfinite(X).all():
            raise ValueError("nonfinite causal packet")
        return Observation(Packet(values, cfg.registry_hash),
                           Graph(X, cfg.registry_hash))
    m = est.own
    (bx, by), _ = camera_bearings(est.r, m.quaternion, cfg)
    margins = np.asarray(cfg.fov) / 2 - np.abs([bx, by])
    raw = np.r_[
        est.r,
        est.rv,
        est.pad_v,
        est.pad_a,
        m.own_velocity,
        m.quaternion,
        m.angular_rate,
        bx,
        by,
        margins,
        float(est.detected),
        m.confidence,
        min(est.age, 1000),
        float(est.initialized),
        est.std,
        est.velocity_std,
        max(0, 1 - elapsed / cfg.horizon),
        float(safety.inhibited),
        float(safety.abort),
    ]
    values = (raw / (SCALES + np.abs(raw))).astype(np.float32)
    values[DIRECT] = raw[DIRECT]
    if cfg.direct_acceleration:
        from scipy.spatial.transform import Rotation
        horizon = 0.5
        predicted_r = est.r + horizon*est.rv - 0.5*horizon**2*est.pad_a
        rotation = Rotation.from_quat(m.quaternion[[1, 2, 3, 0]])
        future_q = (rotation * Rotation.from_rotvec(horizon*m.angular_rate)).as_quat()
        predicted, depth = camera_bearings(predicted_r, future_q[[3, 0, 1, 2]], cfg)
        future = (np.r_[predicted, np.asarray(cfg.fov)/2-np.abs(predicted)]
                  if depth > 0 else np.r_[np.zeros(2), -np.asarray(cfg.fov)/2])
        if not est.initialized:
            future[:] = 0.
        scales = np.tile(np.asarray(cfg.fov)/2, 2)
        values = np.r_[values, future/(scales+np.abs(future))].astype(np.float32)
    if cfg.reference_tracking:
        values = np.r_[values, est.acceleration_std/(1+est.acceleration_std),
                       np.clip(est.previous_action, -1., 1.)].astype(np.float32)
    if cfg.reference_context:
        measured = np.zeros(2)
        valid = m.optical_position is not None
        if valid:
            measured, _ = camera_bearings(m.optical_position,
                m.quaternion if m.optical_quaternion is None else m.optical_quaternion, cfg)
        optical_age = max(0., m.time_s-(m.optical_time_s if m.optical_time_s is not None else m.time_s))
        values = np.r_[values, measured/(np.asarray(cfg.fov)/2+np.abs(measured)),
                       float(valid), optical_age/(1+optical_age)].astype(np.float32)
        from .context import reference_context_graphs
        X = reference_context_graphs(values, cfg)
        if not np.isfinite(values).all() or not np.isfinite(X).all():
            raise ValueError('nonfinite causal packet')
        return Observation(Packet(values,cfg.registry_hash),Graph(X,cfg.registry_hash))
    p = dict(zip(cfg.packet_fields, values))
    X = np.zeros((9, 12), dtype=np.float32)

    def put(node, names):
        fields = names.split()
        X[node, : len(fields)] = [p[name] for name in fields]

    # Nine named contexts, same typed edges and four grouped readouts as v2.8.
    put(0, "detected bx by mx my confidence age initialized")
    put(1, "pvx pvy pvz pax pay paz velstd confidence initialized")
    put(2, "rz vx vy vz remaining")
    put(3, "qw qx qy qz wx wy wz")
    put(4, "rx ry rz rvx rvy rvz posstd velstd initialized")
    put(5, "rx ry rvx rvy pax pay confidence")
    put(6, "age bx by mx my initialized confidence")
    X[7, 0] = safety.eligible
    X[7, 1:9] = [p[k] for k in "rx ry rvx rvy rvz rz confidence inhibited".split()]
    put(8, "inhibited abort age posstd velstd initialized")
    if cfg.direct_acceleration:
        X[0, :9] = 0
        X[6, :9] = 0
        put(0, "detected bx by pred_bx pred_by confidence age initialized")
        put(6, "age pred_bx pred_by pred_mx pred_my initialized confidence")
    if cfg.reference_tracking:
        put(1, "pvx pvy pvz pax pay paz velstd accstd initialized")
    X[:, 9] = p["remaining"]
    X[:, 10] = 1.0
    X[:, 11] = np.arange(1, 10) / 9
    if not np.isfinite(values).all() or not np.isfinite(X).all():
        raise ValueError("nonfinite causal packet")
    return Observation(Packet(values, cfg.registry_hash), Graph(X, cfg.registry_hash))
