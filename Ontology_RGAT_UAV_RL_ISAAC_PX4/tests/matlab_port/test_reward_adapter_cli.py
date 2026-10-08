import math
import subprocess
import sys

import numpy as np

from ontology_rgat.direct_policy.isaac_adapter import (
    CommandOwnership, direct_acceleration_payload, embed_planar_action,
    trajectory_setpoint_fields,
)
from ontology_rgat.direct_policy.reward import (
    TerminalEvent, TruthState, ViewMeasurement, classify_terminal, reward_v5,
)
from ontology_rgat.direct_policy.reporting import authority_metrics, export_matlab


def test_terminal_order_and_reward_components_are_finite():
    old = TruthState((0.2, 1.0), (0.1, -0.2), (0.01,), (0.02,))
    new = TruthState((0.1, 0.9), (0.05, -0.2), (0.01,), (0.02,))
    event = classify_terminal(new, elapsed_s=1, contact=False)
    value, terms = reward_v5(old, new, ViewMeasurement(True, (0.1,), (1.0,)),
                             (0.1, -0.1), 0.1, event)
    assert math.isfinite(value) and all(math.isfinite(float(v)) for v in terms.values())
    landed = classify_terminal(TruthState((0.1, 0.04), (0.1, -0.1), (0,), (0,)),
                               elapsed_s=2, contact=True)
    assert landed.reason == "SUCCESS"
    missed = classify_terminal(TruthState((2, 0.04), (0, -0.1), (0,), (0,)),
                               elapsed_s=2, contact=True)
    assert missed.reason == "MISSED_PAD_CONTACT"


def test_px4_acceleration_boundary_and_command_ownership():
    enu = embed_planar_action((1.0, 0.5))
    payload = direct_acceleration_payload(enu, heading_rad=0.2, decision_id=1,
                                          episode_id="e", policy_version=3, stamp_s=4)
    setpoint = trajectory_setpoint_fields(enu, 0.2)
    np.testing.assert_allclose(setpoint["acceleration"], [0, 1, -0.5])
    assert all(math.isnan(item) for item in setpoint["position"]+setpoint["velocity"])
    owner = CommandOwnership(); owner.reset("e"); owner.validate(payload, writer="matlab-port")
    try:
        owner.validate(payload, writer="matlab-port")
        assert False, "duplicate decision accepted"
    except RuntimeError:
        pass


def test_cli_dry_run_is_non_mutating_and_does_not_require_isaac(tmp_path, request):
    root = request.config.rootpath
    command = [sys.executable, str(root/"python/run_matlab_port.py"), "--stage", "smoke",
               "--backend", "isaac", "--dimension", "2", "--dry-run",
               "--output", str(tmp_path)]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
    assert '"status": "NOT_RUN"' in result.stdout


def test_authority_metrics_and_matlab_exports(tmp_path):
    records = [
        {"time_s": 0.0, "horizontal_distance_m": 1.0, "height_m": 2.0,
         "uav_x_m": 0.0, "uav_z_m": 2.0, "pad_x_m": 1.0, "pad_z_m": 0.0,
         "requested_ax": 1.0, "requested_az": 0.0, "applied_ax": 0.8,
         "applied_az": 0.0},
        {"time_s": 0.1, "horizontal_distance_m": 0.9, "height_m": 1.9,
         "uav_x_m": 0.1, "uav_z_m": 1.9, "pad_x_m": 1.0, "pad_z_m": 0.0,
         "requested_ax": 0.5, "requested_az": -0.1, "applied_ax": 0.5,
         "applied_az": -0.1},
    ]
    authority = authority_metrics([
        {"requested": [1, 0], "limited": [0.8, 0], "measured": [0.7, 0],
         "decision_stamp_s": 0, "command_stamp_s": 0.01},
        {"requested": [0.5, -0.1], "limited": [0.5, -0.1], "measured": [0.4, -0.1],
         "decision_stamp_s": 0.1, "command_stamp_s": 0.12},
    ])
    assert authority["intervention_rate"] == 0.5
    outputs = export_matlab(records, authority, tmp_path/"evaluation", {"split": "test"})
    assert all((tmp_path/f"evaluation.{suffix}").exists() for suffix in ("csv", "json", "mat"))
