"""phi44 features and the 8x10 node input of spatial_current_semantics_v1 (guide 8.1)."""
from dataclasses import replace
import math
import re
from types import ModuleType

import numpy as np
import pytest
import yaml

from ontology_rgat.current_semantics import contracts, features, graph
from ontology_rgat.current_semantics.contracts import (
    DEFAULT_CONFIG_PATH, ESTIMATE_CHANNELS, NODE_ORDER, OBSERVATION_CHANNELS, OBSERVATION_NAMES,
    TRIG_PAIRS, contact_rate_rad_s, contact_tilt_rad, contract_manifest, load_config)
from ontology_rgat.current_semantics.features import (
    FEATURE_NAMES, FEATURES, NODE_FEATURES, OPEN_BOUND, bounded, build_feature_batch,
    build_features, derived_features, feature_registry, raw_from_flat, raw_from_nodes)
from ontology_rgat.current_semantics.geometry import camera_kinematics
from ontology_rgat.current_semantics.observation import CurrentPhysical, decode, encode
from ontology_rgat.direct_policy.contracts import make_contract
from ontology_rgat.direct_policy.observation import signed_scale


CONFIG = load_config()
REGISTRY = {spec.name: spec for spec in feature_registry(CONFIG)}
FI = {name: index for index, name in enumerate(FEATURE_NAMES)}
OI = {name: index for index, name in enumerate(OBSERVATION_NAMES)}
TRANSFORM = {spec.name: spec.transform for spec in FEATURES}

EXPECTED = {
    "ObservationQuality": ("ugv_visionUpdated", "ugv_visionAge", "drone_navigationValid",
                           "drone_navigationAge"),
    "RelativeGeometry": ("relative_x", "relative_y", "relative_height"),
    "MotionState": ("relative_vx", "relative_vy", "ugv_vx", "ugv_vy", "drone_vz",
                    "own_vx", "own_vy"),
    "AttitudeState": ("drone_sinPhi", "drone_cosPhi", "drone_sinTheta", "drone_cosTheta",
                      "drone_sinPsi", "drone_cosPsi", "drone_rollRate", "drone_pitchRate",
                      "drone_yawRate"),
    "ApproachSituation": ("approach_ex", "approach_ey", "approach_vx", "approach_vy",
                          "approach_closure"),
    "ViewGeometry": ("view_xi", "view_eta", "view_xi_rate", "view_eta_rate",
                     "view_margin_left", "view_margin_right", "view_margin_top",
                     "view_margin_bottom", "view_front_valid"),
    "ContactCondition": ("contact_ex", "contact_ey", "contact_vx", "contact_vy", "contact_vz",
                         "contact_tilt", "contact_rate"),
    "LandingContext": (),
}
APPROACH, VIEW, CONTACT = (EXPECTED[n] for n in ("ApproachSituation", "ViewGeometry",
                                                 "ContactCondition"))
NEEDS_ESTIMATE = ("contact_ex", "contact_ey", "contact_vx", "contact_vy")
OWN_STATE = ("contact_vz", "contact_tilt", "contact_rate")


def physical(**overrides):
    values = dict(ex=0.0, ey=0.0, h=2.0, vrx=0.0, vry=0.0, vgx=0.0, vgy=0.0, vz=0.0,
                  roll=0.0, pitch=0.0, yaw=0.0, p=0.0, q=0.0, r=0.0,
                  vision_updated=True, vision_age_s=0.05, nav_valid=True, nav_age_s=0.0,
                  estimate_valid=True)
    values.update(overrides)
    return CurrentPhysical(**values)


def random_physical(rng, **overrides):
    values = dict(ex=rng.uniform(-6, 6), ey=rng.uniform(-6, 6), h=rng.uniform(-1, 12),
                  vrx=rng.uniform(-4, 4), vry=rng.uniform(-4, 4),
                  vgx=rng.uniform(-4, 4), vgy=rng.uniform(-4, 4), vz=rng.uniform(-2, 2),
                  roll=rng.uniform(-0.6, 0.6), pitch=rng.uniform(-0.6, 0.6),
                  yaw=rng.uniform(-3.1, 3.1), p=rng.uniform(-3, 3), q=rng.uniform(-3, 3),
                  r=rng.uniform(-3, 3), vision_updated=bool(rng.integers(2)),
                  vision_age_s=rng.uniform(0, 6), nav_valid=bool(rng.integers(2)),
                  nav_age_s=rng.uniform(0, 6), estimate_valid=bool(rng.random() > 0.2))
    values.update(overrides)
    return physical(**values)


NEAR_ONE = float(np.nextafter(1.0, 0.0))
EXTREME_SIGNED = (0.0, 1e-300, 0.5, 1.0 - 1e-12, NEAR_ONE)
EXTREME_ANGLES = (0.0, math.pi / 2, -math.pi / 2, math.pi, 1e-9)


def raw_space_vector(rng, extreme):
    """A decodable vector drawn in encoded space, including |u| one ulp below 1."""
    vector = np.zeros(21)
    for channel in OBSERVATION_CHANNELS:
        i = OI[channel.name]
        if channel.transform == "signed":
            magnitude = rng.choice(EXTREME_SIGNED) if extreme else rng.uniform(0.0, 1.0)
            vector[i] = rng.choice((-1.0, 1.0)) * magnitude
        elif channel.transform == "flag":
            vector[i] = float(rng.integers(2))
        elif channel.transform == "age":
            vector[i] = rng.choice((0.0, 0.5, NEAR_ONE, 1.0)) if extreme else rng.uniform(0, 1)
    for _, sin_name, cos_name in TRIG_PAIRS:
        angle = rng.choice(EXTREME_ANGLES) if extreme else rng.uniform(-math.pi, math.pi)
        vector[OI[sin_name]], vector[OI[cos_name]] = math.sin(angle), math.cos(angle)
    if vector[OI["ugv_visionAge"]] == 1.0:
        vector[[OI[name] for name in ESTIMATE_CHANNELS]] = 0.0
    return vector


def sample_vectors(seed, count=200):
    rng = np.random.default_rng(seed)
    vectors = [encode(random_physical(rng)) for _ in range(count // 2)]
    vectors += [raw_space_vector(rng, extreme=bool(i % 2)) for i in range(count - count // 2)]
    return vectors


def test_registry_is_44_unique_names_grouped_q4_g3_m7_a9_p5_o9_c7_l0():
    assert len(FEATURES) == 44 == len(set(FEATURE_NAMES))
    assert features.NODE_ORDER is contracts.NODE_ORDER
    assert tuple(NODE_FEATURES) == NODE_ORDER
    assert [len(NODE_FEATURES[node]) for node in NODE_ORDER] == [4, 3, 7, 9, 5, 9, 7, 0]
    assert dict(NODE_FEATURES) == EXPECTED
    assert FEATURE_NAMES == tuple(name for node in NODE_ORDER for name in EXPECTED[node])
    assert sorted(set(FEATURE_NAMES) & set(OBSERVATION_NAMES)) == sorted(OBSERVATION_NAMES)
    for spec in FEATURES:
        assert spec.node == next(node for node, names in EXPECTED.items() if spec.name in names)


def test_resolved_scales_come_from_the_observation_contract_and_the_config():
    scales = CONFIG.feature_scales
    observation = {channel.name: channel for channel in OBSERVATION_CHANNELS}
    expected = {
        "own_vx": observation["ugv_vx"].scale, "own_vy": observation["ugv_vy"].scale,
        "approach_ex": scales.approach_position_m, "approach_ey": scales.approach_position_m,
        "approach_vx": scales.approach_velocity_m_s, "approach_vy": scales.approach_velocity_m_s,
        "approach_closure": scales.approach_closure_m2_s,
        "view_xi_rate": scales.image_rate_per_s, "view_eta_rate": scales.image_rate_per_s,
        "contact_ex": 0.5, "contact_ey": 0.5, "contact_vx": 0.35, "contact_vy": 0.35,
        "contact_vz": 0.30, "contact_tilt": math.radians(5.0),
        "contact_rate": math.radians(10.0)}
    expected.update({name: scales.image_coordinate for name in VIEW
                     if name not in expected and name != "view_front_valid"})
    for name, scale in expected.items():
        assert REGISTRY[name].scale == pytest.approx(scale, rel=0, abs=1e-15), name
        assert REGISTRY[name].transform == "signed", name
    # observation/2 rescaled the velocity channels to 1 m/s; own velocity follows them.
    assert REGISTRY["own_vx"].scale == observation["relative_vx"].scale == 1.0
    assert REGISTRY["view_front_valid"].transform == "flag"
    assert REGISTRY["view_front_valid"].scale is None
    for name in OBSERVATION_NAMES:
        assert (REGISTRY[name].transform, REGISTRY[name].scale) == (
            observation[name].transform, observation[name].scale)
    template = {spec.name: spec for spec in FEATURES}
    assert template["approach_ex"].scale is None and template["own_vx"].scale == 1.0
    assert all(spec.scale is None or spec.scale > 0 for spec in REGISTRY.values())


def test_nodes_are_8x10_with_id_column_and_zero_padding():
    for vector in sample_vectors(1, 40):
        frame = build_features(vector, CONFIG)
        assert frame.flat.shape == (44,) and frame.nodes.shape == (8, 10)
        np.testing.assert_array_equal(frame.nodes[:, 9], (np.arange(8) + 1) / 8)
        offset = 0
        for row, node in enumerate(NODE_ORDER):
            count = len(NODE_FEATURES[node])
            np.testing.assert_array_equal(frame.nodes[row, :count],
                                          frame.flat[offset:offset + count])
            assert np.all(frame.nodes[row, count:9] == 0.0)
            offset += count
        assert offset == 44
        assert np.all(frame.nodes[NODE_ORDER.index("LandingContext"), :9] == 0.0)


def test_raw21_is_recovered_exactly_by_name_from_flat_and_nodes():
    vectors = sample_vectors(2, 300)
    for vector in vectors:
        frame = build_features(vector, CONFIG)
        np.testing.assert_array_equal(raw_from_flat(frame.flat), vector)
        np.testing.assert_array_equal(raw_from_nodes(frame.nodes), vector)
        np.testing.assert_array_equal(frame.raw, vector)
        for name in OBSERVATION_NAMES:
            assert frame.flat[FI[name]] == vector[OI[name]], name
            node = REGISTRY[name].node
            row, slot = NODE_ORDER.index(node), NODE_FEATURES[node].index(name)
            assert frame.nodes[row, slot] == vector[OI[name]], name
    flat, nodes = build_feature_batch(np.stack(vectors), CONFIG)
    np.testing.assert_array_equal(raw_from_flat(flat), np.stack(vectors))
    np.testing.assert_array_equal(raw_from_nodes(nodes), np.stack(vectors))
    with pytest.raises(ValueError):
        raw_from_flat(np.zeros(43))
    with pytest.raises(ValueError):
        raw_from_nodes(np.zeros((8, 9)))


def test_build_is_pure_and_holds_no_module_level_mutable_state():
    first, second, third = sample_vectors(3, 3)
    original = first.copy()
    a = build_features(first, CONFIG)
    build_features(second, CONFIG)
    build_feature_batch(np.stack([third, second]), CONFIG)
    build_features(third, CONFIG)
    b = build_features(first, CONFIG)
    np.testing.assert_array_equal(first, original)
    for field in ("flat", "nodes", "raw"):
        assert getattr(a, field).tobytes() == getattr(b, field).tobytes(), field
        assert not getattr(a, field).flags.writeable, field
    assert a.camera == b.camera and a.estimate_valid is b.estimate_valid
    with pytest.raises(ValueError):
        a.flat[0] = 0.5

    for module in (features, graph):
        for name, value in vars(module).items():
            if name.startswith("__") or isinstance(value, (ModuleType, type)) or callable(value):
                continue
            assert not isinstance(value, (list, dict, set, bytearray)), (module.__name__, name)
            if isinstance(value, np.ndarray):
                assert not value.flags.writeable, (module.__name__, name)


def test_invalid_estimate_zeroes_estimate_entries_but_keeps_own_state():
    rng = np.random.default_rng(4)
    gated = {spec.name for spec in FEATURES if spec.valid_when != "always"}
    assert gated >= set(APPROACH) | set(VIEW) - {"view_front_valid"} | set(NEEDS_ESTIMATE)
    assert gated.isdisjoint(OWN_STATE) and "view_front_valid" not in gated
    for _ in range(50):
        state = random_physical(rng, estimate_valid=False, vz=rng.uniform(0.1, 2.0),
                                roll=rng.uniform(0.05, 0.5), p=rng.uniform(0.1, 2.0))
        frame = build_features(encode(state), CONFIG)
        assert frame.estimate_valid is False and frame.camera.front_valid is False
        for name in gated | {"view_front_valid"}:
            assert frame.flat[FI[name]] == 0.0, name
        for name in APPROACH + VIEW + NEEDS_ESTIMATE + ("own_vx", "own_vy"):
            assert frame.flat[FI[name]] == 0.0, name
        decoded = decode(frame.raw)
        tilt = contact_tilt_rad(decoded.roll, decoded.pitch)
        rate = contact_rate_rad_s(decoded.p, decoded.q)
        assert frame.flat[FI["contact_tilt"]] == bounded(float(tilt), math.radians(5.0)) != 0.0
        assert frame.flat[FI["contact_rate"]] == bounded(float(rate), math.radians(10.0)) != 0.0
        assert frame.flat[FI["contact_vz"]] == bounded(decoded.vz, 0.30) != 0.0


def test_estimate_valid_but_behind_the_lens_is_neutral_view_only():
    frame = build_features(encode(physical(ex=0.4, ey=0.2, h=0.10, vrx=0.3)), CONFIG)
    assert frame.estimate_valid and not frame.camera.front_valid
    assert all(frame.flat[FI[name]] == 0.0 for name in VIEW)
    assert frame.flat[FI["approach_ex"]] != 0.0 and frame.flat[FI["contact_ex"]] != 0.0


def test_view_entries_are_the_bounded_camera_kinematics():
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(100):
        state = random_physical(rng, estimate_valid=True, h=rng.uniform(1.0, 8.0))
        frame = build_features(encode(state), CONFIG)
        k = camera_kinematics(decode(frame.raw), CONFIG.calibration)
        assert k == frame.camera
        if not k.front_valid:
            continue
        checked += 1
        expected = {"view_xi": k.xi, "view_eta": k.eta, "view_xi_rate": k.xi_dot,
                    "view_eta_rate": k.eta_dot, "view_margin_left": k.margin("left"),
                    "view_margin_right": k.margin("right"), "view_margin_top": k.margin("top"),
                    "view_margin_bottom": k.margin("bottom")}
        for name, value in expected.items():
            assert frame.flat[FI[name]] == bounded(value, REGISTRY[name].scale), name
        assert frame.flat[FI["view_front_valid"]] == 1.0
    assert checked > 50
    hover = build_features(encode(physical(h=2.0)), CONFIG)
    # fx = (W/2)/tan(45 deg) carries one ulp of rounding, so the axis value is approximate.
    assert hover.flat[FI["view_xi"]] == pytest.approx(
        bounded(1.0 / (CONFIG.calibration.width - 1), 1.0), abs=1e-15)


def test_non_flag_entries_are_bounded_on_a_sweep_with_extreme_values():
    """Every B entry is strictly inside (-1, 1); raw sin/cos keep their closed
    [-1, 1] (cos 0 = 1 is the level-attitude value) and raw ages reach 1 only
    as the registered infinite-age sentinel (guide 6.1)."""
    rng = np.random.default_rng(6)
    signed = [FI[spec.name] for spec in FEATURES if spec.transform == "signed"]
    trig = [FI[spec.name] for spec in FEATURES if spec.transform == "trig"]
    ages = [FI[spec.name] for spec in FEATURES if spec.transform == "age"]
    flags = [FI[spec.name] for spec in FEATURES if spec.transform == "flag"]
    assert (len(signed), len(trig), len(ages), len(flags)) == (33, 6, 2, 3)
    vectors = np.stack([raw_space_vector(rng, extreme=i % 3 != 0) for i in range(600)])
    flat, _ = build_feature_batch(vectors, CONFIG)
    assert np.all(np.isfinite(flat))
    assert np.all(np.abs(flat[:, signed]) < 1.0)
    assert np.max(np.abs(flat[:, signed])) == OPEN_BOUND
    assert np.all(np.abs(flat[:, trig]) <= 1.0)
    assert np.all((flat[:, ages] >= 0.0) & (flat[:, ages] <= 1.0))
    for column in ages:
        name = FEATURE_NAMES[column]
        sentinel = flat[:, column] == 1.0
        assert sentinel.any(), name
        assert all(decode(v).__getattribute__(
            {"ugv_visionAge": "vision_age_s", "drone_navigationAge": "nav_age_s"}[name])
            == math.inf for v in vectors[sentinel])
    assert set(np.unique(flat[:, flags])) <= {0.0, 1.0}


def test_bounded_is_the_registered_map_and_saturates_only_at_float_resolution():
    for x, s in ((0.3, 1.0), (-2.0, 3.0), (1e-300, 0.5), (0.0, 1.0)):
        assert bounded(x, s) == x / (abs(x) + s) == float(signed_scale(x, s))
    assert bounded(2.7e16, 1.0) == OPEN_BOUND < 1.0
    assert bounded(-2.7e16, 1.0) == -OPEN_BOUND


def test_contact_entries_are_one_half_at_the_registered_limits():
    limits = CONFIG.contact
    registered = {"contact_ex": limits.pad_radius_m, "contact_ey": limits.pad_radius_m,
                  "contact_vx": limits.xy_speed_m_s, "contact_vy": limits.xy_speed_m_s,
                  "contact_vz": limits.z_speed_m_s, "contact_tilt": limits.tilt_rad,
                  "contact_rate": limits.rate_rad_s}
    for name, limit in registered.items():
        assert REGISTRY[name].scale == limit, name
        assert bounded(limit, REGISTRY[name].scale) == 0.5, name
        assert bounded(-limit, REGISTRY[name].scale) == -0.5, name

    at_limit = physical(ex=limits.pad_radius_m, ey=-limits.pad_radius_m,
                        vrx=limits.xy_speed_m_s, vry=-limits.xy_speed_m_s,
                        vz=-limits.z_speed_m_s, p=0.0, q=-limits.rate_rad_s)
    values = derived_features(at_limit, camera_kinematics(at_limit, CONFIG.calibration), CONFIG)
    assert [values[name] for name in ("contact_ex", "contact_ey", "contact_vx", "contact_vy",
                                      "contact_vz", "contact_rate")] == [
        0.5, -0.5, 0.5, -0.5, -0.5, 0.5]
    # acos(cos(t)) is not t in float64 for t = 5 deg (no roll within 200 ulps
    # gives it back exactly), so the tilt is checked end to end to rounding.
    tilted = physical(roll=limits.tilt_rad)
    tilt = derived_features(tilted, camera_kinematics(tilted, CONFIG.calibration),
                            CONFIG)["contact_tilt"]
    assert abs(tilt - 0.5) <= 1e-15


def test_approach_closure_is_negative_while_closing_on_the_pad():
    cases = {
        "own_moves_toward_pad": (dict(ex=2.0, ey=1.0, vgx=0.0, vgy=0.0, vrx=-1.0, vry=-0.5), -1),
        "pad_moves_toward_own": (dict(ex=2.0, ey=0.0, vgx=-1.0, vgy=0.0, vrx=-1.0, vry=0.0), -1),
        "own_moves_away": (dict(ex=2.0, ey=1.0, vgx=0.0, vgy=0.0, vrx=1.0, vry=0.5), 1),
        "tangential": (dict(ex=2.0, ey=0.0, vgx=0.0, vgy=0.0, vrx=0.0, vry=1.0), 0),
    }
    for label, (state, sign) in cases.items():
        frame = build_features(encode(physical(**state)), CONFIG)
        decoded = decode(frame.raw)
        closure = frame.flat[FI["approach_closure"]]
        assert np.sign(closure) == sign, label
        assert closure == bounded(decoded.ex * decoded.vrx + decoded.ey * decoded.vry, 1.0)
    # own velocity toward the pad = vg - vr points along e.
    toward = decode(build_features(encode(physical(**cases["own_moves_toward_pad"][0])),
                                   CONFIG).raw)
    assert toward.own_vx * toward.ex + toward.own_vy * toward.ey > 0


def test_own_velocity_is_bounded_pad_minus_relative_at_the_raw_velocity_scale():
    rng = np.random.default_rng(7)
    scale = {channel.name: channel.scale for channel in OBSERVATION_CHANNELS}["ugv_vx"]
    for vector in sample_vectors(7, 100):
        frame = build_features(vector, CONFIG)
        decoded = decode(vector)
        if not decoded.estimate_valid:
            assert frame.flat[FI["own_vx"]] == frame.flat[FI["own_vy"]] == 0.0
            continue
        assert frame.flat[FI["own_vx"]] == bounded(decoded.vgx - decoded.vrx, scale)
        assert frame.flat[FI["own_vy"]] == bounded(decoded.vgy - decoded.vry, scale)
    for _ in range(50):
        own, pad = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
        state = physical(vgx=pad[0], vgy=pad[1], vrx=pad[0] - own[0], vry=pad[1] - own[1])
        frame = build_features(encode(state), CONFIG)
        assert frame.flat[FI["own_vx"]] == pytest.approx(float(signed_scale(own[0], scale)),
                                                          abs=1e-12)
        assert frame.flat[FI["own_vy"]] == pytest.approx(float(signed_scale(own[1], scale)),
                                                          abs=1e-12)


def test_batch_builder_equals_the_per_sample_builder():
    vectors = np.stack(sample_vectors(8, 64))
    flat, nodes = build_feature_batch(vectors, CONFIG)
    assert flat.shape == (64, 44) and nodes.shape == (64, 8, 10)
    for row, vector in enumerate(vectors):
        frame = build_features(vector, CONFIG)
        assert flat[row].tobytes() == frame.flat.tobytes()
        assert nodes[row].tobytes() == frame.nodes.tobytes()
    empty_flat, empty_nodes = build_feature_batch(np.zeros((0, 21)), CONFIG)
    assert empty_flat.shape == (0, 44) and empty_nodes.shape == (0, 8, 10)


def test_stable_24d_and_malformed_inputs_are_rejected():
    stable = make_contract(3, backend="local", training_profile="stable").observation
    assert stable.dimension == 24
    vector = np.r_[encode(physical()), 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="24D"):
        build_features(vector, CONFIG)
    with pytest.raises(ValueError, match="24D"):
        build_feature_batch(vector[None, :], CONFIG)
    with pytest.raises(ValueError):
        build_features(encode(physical())[None, :], CONFIG)
    with pytest.raises(ValueError):
        build_feature_batch(encode(physical()), CONFIG)
    bad = encode(physical())
    bad[OI["relative_x"]] = 1.0
    with pytest.raises(ValueError):
        build_feature_batch(np.stack([encode(physical()), bad]), CONFIG)


def _variant(tmp_path, name, edit):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return contract_manifest(load_config(path))


def test_manifest_is_complete_and_feature_hash_tracks_scales_calibration_and_names(
        tmp_path, monkeypatch):
    base = contract_manifest(CONFIG)
    assert base["complete"] is True and base["missing_parts"] == []
    for key in ("feature_hash", "graph_hash", "algorithm_hash"):
        assert re.fullmatch(r"[0-9a-f]{64}", base[key]), key
    assert [item["name"] for item in base["features"]["features"]] == list(FEATURE_NAMES)
    assert base["features"]["features"][FI["approach_ex"]]["scale"] == 1.0

    edits = {
        "position_scale": lambda d: d["feature_scales"].update(approach_position_m=2.0),
        "image_rate_scale": lambda d: d["feature_scales"].update(image_rate_per_s=4.0),
        "pad_radius": lambda d: d["contact"].update(pad_radius_m=0.6),
        "z_near": lambda d: d["camera"].update(z_near_m=0.06),
    }
    for name, edit in edits.items():
        variant = _variant(tmp_path, name, edit)
        assert variant["feature_hash"] != base["feature_hash"], name
        assert variant["algorithm_hash"] != base["algorithm_hash"], name
        assert variant["graph_hash"] == base["graph_hash"], name
        assert variant["observation_hash"] == base["observation_hash"], name
        assert (variant["calibration_hash"] != base["calibration_hash"]) is (name == "z_near")

    renamed = (replace(FEATURES[FI["approach_closure"]], name="approach_dot"),)
    monkeypatch.setattr(features, "FEATURES", FEATURES[:FI["approach_closure"]] + renamed
                        + FEATURES[FI["approach_closure"] + 1:])
    changed = contract_manifest(CONFIG)
    assert changed["feature_hash"] != base["feature_hash"]
    assert changed["graph_hash"] == base["graph_hash"]
