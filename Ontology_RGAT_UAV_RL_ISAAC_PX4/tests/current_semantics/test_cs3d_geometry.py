"""Camera kinematics and the nominal v12 calibration of spatial_current_semantics_v1."""
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from ontology_rgat.current_semantics.contracts import (
    CALIBRATION_STATUS, DEPLOYMENT_PROFILE_SHA256, PROJECT_ROOT, load_config, nominal_v12,
    resolve_deployment_profile)
from ontology_rgat.current_semantics.geometry import (
    MARGIN_NAMES, camera_kinematics, rotation_world_from_body)
from ontology_rgat.current_semantics.observation import CurrentPhysical


CAL = nominal_v12(0.05)
W, H = CAL.width, CAL.height
MOUNT_DEPTH = 0.16
PROJECTION_FIELDS = ("u", "v", "u_dot", "v_dot", "xi", "eta", "xi_dot", "eta_dot")


def state(**overrides):
    values = dict(ex=0.0, ey=0.0, h=2.0, vrx=0.0, vry=0.0, vgx=0.0, vgy=0.0, vz=0.0,
                  roll=0.0, pitch=0.0, yaw=0.0, p=0.0, q=0.0, r=0.0,
                  vision_updated=True, vision_age_s=0.05, nav_valid=True, nav_age_s=0.0,
                  estimate_valid=True)
    values.update(overrides)
    return CurrentPhysical(**values)


def random_state(rng, **overrides):
    values = dict(ex=rng.uniform(-1, 1), ey=rng.uniform(-1, 1), h=rng.uniform(1.5, 4.0),
                  vrx=rng.uniform(-1, 1), vry=rng.uniform(-1, 1), vz=rng.uniform(-1, 1),
                  roll=rng.uniform(-0.5, 0.5), pitch=rng.uniform(-0.5, 0.5),
                  yaw=rng.uniform(-math.pi, math.pi),
                  p=rng.uniform(-1, 1), q=rng.uniform(-1, 1), r=rng.uniform(-1, 1))
    values.update(overrides)
    return state(**values)


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def assert_neutral(kinematics):
    assert kinematics.front_valid is False
    for name in PROJECTION_FIELDS:
        assert getattr(kinematics, name) == 0.0, name
    assert kinematics.margins == (0.0, 0.0, 0.0, 0.0)


def test_rotation_world_from_body_is_scipy_extrinsic_xyz():
    rng = np.random.default_rng(1)
    for _ in range(100):
        roll, pitch, yaw = rng.uniform([-math.pi, -math.pi / 2, -math.pi],
                                       [math.pi, math.pi / 2, math.pi])
        expected = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        np.testing.assert_allclose(rotation_world_from_body(roll, pitch, yaw), expected,
                                   atol=1e-14, rtol=0)


def test_nadir_hover_projects_to_the_principal_point():
    """The lens is 0.16 m BELOW the body origin (mount_translation_flu_m z = -0.16,
    isaac_sim/keypoint_geometry.camera_pose_in_pad adds R @ mount), so at h = 2 the
    pad is 1.84 m deep, not 2.16."""
    k = camera_kinematics(state(h=2.0), CAL)
    assert k.front_valid
    np.testing.assert_allclose(k.p_c, (0.0, 0.0, 2.0 - MOUNT_DEPTH), atol=1e-15)
    assert (k.u, k.v) == (CAL.cx, CAL.cy)
    assert k.xi == pytest.approx(1.0 / (W - 1), abs=1e-15)
    assert k.eta == pytest.approx(1.0 / (H - 1), abs=1e-15)
    assert (k.u_dot, k.v_dot, k.xi_dot, k.eta_dot) == (0.0, 0.0, 0.0, 0.0)


def test_body_forward_is_image_right_and_body_right_is_image_bottom():
    ahead = camera_kinematics(state(ex=0.5), CAL)
    assert ahead.u > CAL.cx and ahead.v == pytest.approx(CAL.cy)
    right = camera_kinematics(state(ey=-0.5), CAL)
    assert right.v > CAL.cy and right.u == pytest.approx(CAL.cx)
    turned = camera_kinematics(state(ex=0.5, yaw=math.pi / 2), CAL)
    assert turned.v > CAL.cy, "a pad east of a north-facing vehicle is on its right"


def test_rates_match_a_central_difference_of_the_rigid_motion():
    rng = np.random.default_rng(2)
    dt = 1e-5
    for _ in range(30):
        s0 = random_state(rng)
        r0 = np.array([s0.ex, s0.ey, -s0.h])
        v_rel = np.array([s0.vrx, s0.vry, -s0.vz])
        omega = np.array([s0.p, s0.q, s0.r])
        attitude0 = rotation_world_from_body(s0.roll, s0.pitch, s0.yaw)

        def at(t):
            roll, pitch, yaw = Rotation.from_matrix(
                attitude0 @ expm(skew(omega) * t)).as_euler("xyz")
            position = r0 + v_rel * t
            return camera_kinematics(replace(s0, ex=position[0], ey=position[1],
                                             h=-position[2], roll=roll, pitch=pitch,
                                             yaw=yaw), CAL)

        now, ahead, behind = at(0.0), at(dt), at(-dt)
        assert now.front_valid and ahead.front_valid and behind.front_valid
        numeric = (np.array(ahead.p_c) - np.array(behind.p_c)) / (2 * dt)
        np.testing.assert_allclose(now.p_c_dot, numeric, rtol=1e-6, atol=1e-8)
        for value, rate in (("u", "u_dot"), ("v", "v_dot"), ("xi", "xi_dot"),
                            ("eta", "eta_dot")):
            numeric = (getattr(ahead, value) - getattr(behind, value)) / (2 * dt)
            assert getattr(now, rate) == pytest.approx(numeric, rel=1e-6, abs=1e-6), rate


def test_rotational_term_uses_the_body_vector_not_the_lens_vector():
    """omega x p_C would drop the 0.16 m lever arm: 0.92 m/s instead of 1.0 m/s here."""
    roll_rate = camera_kinematics(state(h=2.0, p=0.5), CAL)
    assert roll_rate.p_c_dot == pytest.approx((0.0, 2.0 * 0.5, 0.0), abs=1e-15)
    pitch_rate = camera_kinematics(state(h=2.0, q=0.5), CAL)
    assert pitch_rate.p_c_dot == pytest.approx((2.0 * 0.5, 0.0, 0.0), abs=1e-15)

    s = random_state(np.random.default_rng(3), p=0.8, q=-0.6)
    k = camera_kinematics(s, CAL)
    r_bw = rotation_world_from_body(s.roll, s.pitch, s.yaw).T
    b = r_bw @ np.array([s.ex, s.ey, -s.h])
    omega = np.array([s.p, s.q, s.r])
    lens_variant = CAL.r_cb_matrix @ (r_bw @ np.array([s.vrx, s.vry, -s.vz])
                                      - np.cross(omega, b - CAL.t_bc_vector))
    np.testing.assert_allclose(np.asarray(k.p_c_dot) - lens_variant,
                               -CAL.r_cb_matrix @ np.cross(omega, CAL.t_bc_vector), atol=1e-12)
    assert np.linalg.norm(np.asarray(k.p_c_dot) - lens_variant) > 0.1


def test_projection_matches_the_isaac_side_keypoint_geometry(monkeypatch):
    """Independent oracle: isaac_sim/keypoint_geometry.py is numpy-only and is what
    the simulator-side labels use; same K, mount and nadir optical frame."""
    path = PROJECT_ROOT / "isaac_sim" / "keypoint_geometry.py"
    spec = importlib.util.spec_from_file_location("cs3d_keypoint_geometry_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, oracle)
    spec.loader.exec_module(oracle)
    camera = oracle.CameraModel(width=W, height=H, horizontal_fov_deg=90.0, pitch_down_deg=90.0,
                                mount_translation_flu_m=CAL.t_bc_flu)
    rng = np.random.default_rng(4)
    for _ in range(50):
        s = random_state(rng, ex=rng.uniform(-2, 2), ey=rng.uniform(-2, 2))
        x, y, z, w = Rotation.from_euler("xyz", [s.roll, s.pitch, s.yaw]).as_quat()
        pixels, depth, _ = oracle.project_pad_points(
            np.zeros((1, 3)), (-s.ex, -s.ey, s.h), (w, x, y, z), camera)
        k = camera_kinematics(s, CAL)
        assert k.p_c[2] == pytest.approx(depth[0], abs=1e-12)
        assert (k.u, k.v) == pytest.approx(tuple(pixels[0]), abs=1e-9)


def test_behind_too_near_or_missing_estimate_is_neutral_and_never_divides():
    behind = camera_kinematics(state(h=0.10), CAL)
    assert_neutral(behind)
    assert behind.p_c[2] == pytest.approx(0.10 - MOUNT_DEPTH)
    assert_neutral(camera_kinematics(state(h=MOUNT_DEPTH), CAL))
    assert_neutral(camera_kinematics(state(h=MOUNT_DEPTH + 0.04, ex=0.01, vz=-0.3), CAL))
    assert camera_kinematics(state(h=MOUNT_DEPTH + 0.06), CAL).front_valid

    missing = camera_kinematics(state(ex=0.0, ey=0.0, h=0.0, estimate_valid=False,
                                      vision_age_s=math.inf), CAL)
    assert_neutral(missing)
    assert missing.p_c == (0.0, 0.0, 0.0) and missing.p_c_dot == (0.0, 0.0, 0.0)
    assert_neutral(camera_kinematics(state(h=2.0, estimate_valid=False), CAL))


def test_fov_edges_map_to_the_registered_normalisation():
    """Pixel edges are u = 0 and u = W, so xi spans [-1, 1 + 2/(W-1)]: the
    right/bottom edge sits 2/(W-1) past +1, the same offset as the registered
    xi = 1/(W-1) on the optical axis.  The tolerance is therefore 2/(W-1)."""
    depth = 2.0
    h = depth + MOUNT_DEPTH
    half_w, half_h = math.tan(CAL.hfov_rad / 2), math.tan(CAL.vfov_rad / 2)
    left = camera_kinematics(state(h=h, ex=-depth * half_w), CAL)
    right = camera_kinematics(state(h=h, ex=depth * half_w), CAL)
    top = camera_kinematics(state(h=h, ey=depth * half_h), CAL)
    bottom = camera_kinematics(state(h=h, ey=-depth * half_h), CAL)
    assert left.xi == pytest.approx(-1.0, abs=1e-12)
    assert right.xi == pytest.approx(1.0 + 2.0 / (W - 1), abs=1e-12)
    assert top.eta == pytest.approx(-1.0, abs=1e-12)
    assert bottom.eta == pytest.approx(1.0 + 2.0 / (H - 1), abs=1e-12)
    assert abs(left.xi + 1) <= 2.0 / (W - 1) and abs(right.xi - 1) <= 2.0 / (W - 1) + 1e-12
    assert left.margin("left") == pytest.approx(0.0, abs=1e-12)
    assert abs(right.margin("right")) <= 2.0 / (W - 1) + 1e-12
    assert top.margin("top") == pytest.approx(0.0, abs=1e-12)
    assert abs(bottom.margin("bottom")) <= 2.0 / (H - 1) + 1e-12
    for k in (left, right, top, bottom):
        assert k.margins == pytest.approx((1 + k.xi, 1 - k.xi, 1 + k.eta, 1 - k.eta))
    assert MARGIN_NAMES == ("left", "right", "top", "bottom")


def test_lateral_mirror_reflects_eta_about_its_axis_value():
    eta_axis = 2.0 * CAL.cy / (H - 1) - 1.0
    rng = np.random.default_rng(5)
    for _ in range(50):
        s = random_state(rng, vgy=rng.uniform(-1, 1))
        mirrored = replace(s, ey=-s.ey, vry=-s.vry, vgy=-s.vgy, roll=-s.roll, p=-s.p,
                           yaw=-s.yaw, r=-s.r)
        a, b = camera_kinematics(s, CAL), camera_kinematics(mirrored, CAL)
        assert b.eta - eta_axis == pytest.approx(-(a.eta - eta_axis), abs=1e-12)
        assert b.eta_dot == pytest.approx(-a.eta_dot, abs=1e-12)
        assert b.xi == pytest.approx(a.xi, abs=1e-12)
        assert b.xi_dot == pytest.approx(a.xi_dot, abs=1e-12)
        assert b.p_c == pytest.approx((a.p_c[0], -a.p_c[1], a.p_c[2]), abs=1e-12)


def test_nominal_v12_calibration_is_derived_from_k_not_the_stale_profile_vfov():
    profile = resolve_deployment_profile()
    camera = profile["configuration"]["vision"]["camera"]
    assert camera["vertical_fov_deg"] == 70.0, "stale ZED value; the calibration must ignore it"
    assert (CAL.width, CAL.height) == (640, 480)
    assert CAL.fx == pytest.approx(320.0, rel=1e-12) and CAL.fy == CAL.fx
    assert (CAL.cx, CAL.cy) == (320.0, 240.0)
    assert math.degrees(CAL.hfov_rad) == pytest.approx(90.0, abs=1e-9)
    assert math.degrees(CAL.vfov_rad) == pytest.approx(73.7398, abs=1e-4)
    assert math.degrees(CAL.vfov_rad) == pytest.approx(
        math.degrees(2 * math.atan(240 / 320)), abs=1e-9)
    assert CAL.distortion == (0.0,) * 5 and not CAL.has_distortion
    assert CAL.t_bc_flu == (0.0, 0.0, -0.16)
    np.testing.assert_array_equal(CAL.r_cb_matrix, np.diag([1.0, -1.0, -1.0]))
    # The nominal formula is what Isaac flew (Kit logs), but the in-plane
    # image rotation is still only measured, never asserted by code.
    assert CAL.status == CALIBRATION_STATUS and "Kit logs" in CAL.status \
        and "not asserted" in CAL.status
    assert CAL.source.profile == "spatial-reference/12"
    assert CAL.source.sha256 == DEPLOYMENT_PROFILE_SHA256
    assert not Path(CAL.source.path).is_absolute()
    assert load_config().calibration == CAL
    with pytest.raises(NotImplementedError):
        camera_kinematics(state(), replace(CAL, distortion=(0.01, 0.0, 0.0, 0.0, 0.0)))


def test_light_profile_resolution_equals_the_spatial_runtime_contract():
    from ontology_rgat.spatial.runtime_contract import deployment_profile

    authority = deployment_profile("spatial-reference/12")
    light = resolve_deployment_profile("spatial-reference/12")
    assert authority["sha256"] == light["sha256"] == DEPLOYMENT_PROFILE_SHA256
    assert authority["resolved_scientific_configuration"] == light["configuration"]
    assert Path(authority["path"]) == PROJECT_ROOT / light["path"]
    with pytest.raises(ValueError):
        resolve_deployment_profile("spatial-reference/11")
