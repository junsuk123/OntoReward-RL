"""The 21D current observation and the registered contract of spatial_current_semantics_v1."""
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from ontology_rgat.current_semantics import SCHEMA_ID
from ontology_rgat.current_semantics import contracts
from ontology_rgat.current_semantics.contracts import (
    DEFAULT_CONFIG_PATH, DESIGN_PARAMETER_TOTALS, ESTIMATE_CHANNELS, OBSERVATION_CHANNELS,
    OBSERVATION_NAMES, TRANSFORM_BY_NAME, TRIG_PAIRS, contact_rate_rad_s, contact_tilt_rad,
    contract_manifest, design_parameter_counts, load_config)
from ontology_rgat.current_semantics.observation import (
    CurrentPhysical, decode, encode, from_source, raw_by_name)
from ontology_rgat.direct_policy.contracts import SPATIAL_FIELDS, make_contract
from ontology_rgat.direct_policy.observation import SpatialEstimate, spatial_vector


PYTHON_DIR = Path(contracts.__file__).resolve().parents[2]
INDEX = {name: i for i, name in enumerate(OBSERVATION_NAMES)}
SIGNED = [name for name, kind in TRANSFORM_BY_NAME.items() if kind == "signed"]
FLAGS = [name for name, kind in TRANSFORM_BY_NAME.items() if kind == "flag"]
AGES = [name for name, kind in TRANSFORM_BY_NAME.items() if kind == "age"]
PHYSICAL_FIELDS = tuple(CurrentPhysical.__dataclass_fields__)


def random_physical(rng, **overrides):
    values = dict(
        ex=rng.uniform(-6, 6), ey=rng.uniform(-6, 6), h=rng.uniform(-1, 15),
        vrx=rng.uniform(-4, 4), vry=rng.uniform(-4, 4),
        vgx=rng.uniform(-4, 4), vgy=rng.uniform(-4, 4), vz=rng.uniform(-2, 2),
        roll=rng.uniform(-3.1, 3.1), pitch=rng.uniform(-1.55, 1.55), yaw=rng.uniform(-3.1, 3.1),
        p=rng.uniform(-3, 3), q=rng.uniform(-3, 3), r=rng.uniform(-3, 3),
        vision_updated=bool(rng.integers(2)), vision_age_s=rng.uniform(0, 6),
        nav_valid=bool(rng.integers(2)), nav_age_s=rng.uniform(0, 6), estimate_valid=True)
    values.update(overrides)
    return CurrentPhysical(**values)


def assert_physical_close(actual, expected, tol=1e-12):
    for name in PHYSICAL_FIELDS:
        a, e = getattr(actual, name), getattr(expected, name)
        if isinstance(e, bool):
            assert a is e, name
        elif math.isinf(e):
            assert a == e, name
        else:
            assert math.isclose(a, e, rel_tol=tol, abs_tol=tol), (name, a, e)


def test_channels_are_the_direct_policy_spatial_fields():
    source = make_contract(3, backend="local").observation
    assert source.fields == SPATIAL_FIELDS and source.dimension == 21
    assert OBSERVATION_NAMES == tuple(item.name for item in SPATIAL_FIELDS)
    for channel, field in zip(OBSERVATION_CHANNELS, SPATIAL_FIELDS):
        assert (channel.name, channel.source_scale, channel.meaning) == (
            field.name, field.scale, field.source)
        assert channel.scale == contracts.OBSERVATION_SCALE_OVERRIDES.get(field.name, field.scale)
        assert channel.transform == TRANSFORM_BY_NAME[field.name]
    assert ESTIMATE_CHANNELS == OBSERVATION_NAMES[:7]
    assert {channel.node for channel in OBSERVATION_CHANNELS} == set(contracts.NODE_ORDER[:4])


def test_stable_24d_profile_is_rejected():
    stable = make_contract(3, backend="local", training_profile="stable").observation
    assert stable.dimension == 24
    vector = np.r_[encode(random_physical(np.random.default_rng(0))), 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="24D"):
        decode(vector)


def test_round_trip_on_random_physical_states():
    rng = np.random.default_rng(20261028)
    for _ in range(300):
        physical = random_physical(rng)
        assert_physical_close(decode(encode(physical)), physical)
    physical = random_physical(rng, nav_valid=False, nav_age_s=math.inf)
    decoded = decode(encode(physical))
    assert decoded.nav_age_s == math.inf and decoded.nav_valid is False


def test_decode_and_encode_agree_with_direct_policy_spatial_vector():
    rng = np.random.default_rng(7)
    for _ in range(200):
        pad, own = rng.uniform(-20, 20, 3), rng.uniform(-20, 20, 3)
        pad_v, own_v = np.r_[rng.uniform(-4, 4, 2), 0.0], rng.uniform(-3, 3, 3)
        angles, rates = rng.uniform([-3, -1.5, -3], [3, 1.5, 3]), rng.uniform(-3, 3, 3)
        flags, ages = rng.integers(2, size=2), rng.uniform(0, 5, 2)
        estimate = SpatialEstimate(tuple(pad), tuple(pad_v), tuple(own), tuple(own_v),
                                   tuple(angles), tuple(rates), bool(flags[0]), ages[0],
                                   bool(flags[1]), ages[1])
        vector = from_source(spatial_vector(estimate))
        expected = CurrentPhysical(
            ex=pad[0] - own[0], ey=pad[1] - own[1], h=own[2] - pad[2],
            vrx=pad_v[0] - own_v[0], vry=pad_v[1] - own_v[1], vgx=pad_v[0], vgy=pad_v[1],
            vz=own_v[2], roll=angles[0], pitch=angles[1], yaw=angles[2],
            p=rates[0], q=rates[1], r=rates[2], vision_updated=bool(flags[0]),
            vision_age_s=ages[0], nav_valid=bool(flags[1]), nav_age_s=ages[1],
            estimate_valid=True)
        assert_physical_close(decode(vector), expected, tol=1e-9)
        np.testing.assert_allclose(encode(expected), vector, rtol=0, atol=1e-14)
        np.testing.assert_allclose(encode(decode(vector)), vector, rtol=0, atol=1e-14)


def test_source_rescaling_changes_only_the_overridden_signed_channels():
    # observation/2: the landing-regime scales put the 0.30 m/s touchdown limit
    # at 0.23 of a relative-velocity channel instead of 0.03.
    physical = CurrentPhysical(
        ex=0.2, ey=-0.1, h=0.3, vrx=0.3, vry=-0.1, vgx=0.5, vgy=0.2, vz=-0.3, roll=0.05,
        pitch=-0.04, yaw=0.3, p=0.1, q=-0.1, r=0.0, vision_updated=True, vision_age_s=0.1,
        nav_valid=True, nav_age_s=0.0, estimate_valid=True)
    estimate = SpatialEstimate((1.0, 2.0, 0.0), (0.5, 0.2, 0.0), (0.8, 2.1, 0.3),
                               (0.2, 0.3, -0.3), (0.05, -0.04, 0.3), (0.1, -0.1, 0.0),
                               True, 0.1, True, 0.0)
    source, vector = spatial_vector(estimate), from_source(spatial_vector(estimate))
    np.testing.assert_allclose(vector, encode(physical), rtol=0, atol=1e-14)
    for name in OBSERVATION_NAMES:
        moved = name in contracts.OBSERVATION_SCALE_OVERRIDES
        assert (vector[INDEX[name]] != source[INDEX[name]]) == moved, name
    assert vector[INDEX["relative_vx"]] == pytest.approx(0.3 / 1.3)
    assert source[INDEX["relative_vx"]] == pytest.approx(0.3 / 10.3)
    with pytest.raises(ValueError, match="signed"):
        from_source(np.r_[1.0, source[1:]])


def test_uninitialised_estimate_is_zeros_plus_age_sentinel_and_reads_invalid():
    estimate = SpatialEstimate((9, 9, 9), (2, 2, 0), (1, 2, 5), (0.5, -0.3, -0.2),
                               (0.1, -0.2, 0.3), (0.1, 0.2, 0.3), False, math.inf, True, 0.02,
                               estimate_initialized=False)
    vector = from_source(spatial_vector(estimate))
    assert vector[INDEX["ugv_visionAge"]] == 1.0
    physical = decode(vector)
    assert physical.estimate_valid is False and physical.vision_age_s == math.inf
    assert (physical.ex, physical.ey, physical.h, physical.vrx, physical.vry,
            physical.vgx, physical.vgy) == (0.0,) * 7
    assert physical.vz == pytest.approx(-0.2) and physical.roll == pytest.approx(0.1)
    np.testing.assert_allclose(encode(physical), vector, rtol=0, atol=1e-15)

    ignored = random_physical(np.random.default_rng(3), estimate_valid=False)
    encoded = encode(ignored)
    np.testing.assert_array_equal(encoded[:7], 0.0)
    assert encoded[INDEX["ugv_visionAge"]] == 1.0


def test_non_zero_estimate_with_sentinel_age_is_rejected():
    vector = encode(random_physical(np.random.default_rng(4)))
    vector[INDEX["ugv_visionAge"]] = 1.0
    with pytest.raises(ValueError, match="sentinel"):
        decode(vector)
    for name in ESTIMATE_CHANNELS:
        single = np.zeros(21)
        single[[INDEX[n] for n in ("drone_cosPhi", "drone_cosTheta", "drone_cosPsi")]] = 1.0
        single[INDEX["ugv_visionAge"]] = 1.0
        single[INDEX[name]] = 0.01
        with pytest.raises(ValueError, match="sentinel"):
            decode(single)


def malformed_cases():
    base = encode(random_physical(np.random.default_rng(5)))

    def mutate(**values):
        vector = base.copy()
        for name, value in values.items():
            vector[INDEX[name]] = value
        return vector

    cases = [("shape20", base[:20]), ("shape1x21", base[None, :]),
             ("nan", mutate(relative_x=math.nan)), ("inf", mutate(drone_vz=math.inf))]
    cases += [(f"{name}={value}", mutate(**{name: value}))
              for name in SIGNED for value in (1.0, -1.0, 1.5)]
    cases += [(f"{name}={value}", mutate(**{name: value}))
              for name in FLAGS for value in (0.5, 2.0, -1.0)]
    cases += [(f"{name}={value}", mutate(**{name: value}))
              for name in AGES for value in (-0.1, 1.1)]
    cases += [(f"{sin_name}/{cos_name}", mutate(**{sin_name: 0.6, cos_name: 0.6}))
              for _, sin_name, cos_name in TRIG_PAIRS]
    cases += [("trig_both_one", mutate(drone_sinPsi=1.0, drone_cosPsi=1.0))]
    return cases


@pytest.mark.parametrize("vector", [pytest.param(vector, id=label)
                                    for label, vector in malformed_cases()])
def test_malformed_vectors_are_rejected(vector):
    with pytest.raises(ValueError):
        decode(vector)


def test_trig_pairs_within_tolerance_are_accepted():
    vector = encode(random_physical(np.random.default_rng(6), roll=0.3))
    vector[INDEX["drone_sinPhi"]] += 4e-7
    assert decode(vector).roll == pytest.approx(0.3, abs=1e-6)


def test_encode_refuses_states_the_vector_cannot_represent():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match="sentinel"):
        encode(random_physical(rng, vision_age_s=math.inf))
    with pytest.raises(ValueError, match="age"):
        encode(random_physical(rng, nav_age_s=-0.1))
    with pytest.raises(ValueError, match="age"):
        encode(random_physical(rng, vision_age_s=math.nan))
    with pytest.raises(ValueError, match="finite"):
        encode(random_physical(rng, p=math.inf))


def test_raw_by_name_is_the_registered_order():
    vector = encode(random_physical(np.random.default_rng(9)))
    by_name = raw_by_name(vector)
    assert tuple(by_name) == OBSERVATION_NAMES
    assert [by_name[name] for name in OBSERVATION_NAMES] == vector.tolist()
    with pytest.raises(ValueError):
        raw_by_name(np.zeros(24))


def test_package_import_is_torch_free_and_contracts_never_load_cv2():
    code = (f"import sys; sys.path.insert(0, {str(PYTHON_DIR)!r})\n"
            "import ontology_rgat.current_semantics as cs\n"
            "print('torch' in sys.modules)\n"
            "from ontology_rgat.current_semantics import contracts\n"
            "contracts.contract_manifest(contracts.load_config())\n"
            "print('cv2' in sys.modules, 'keypoint_geometry' in sys.modules)\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=True, timeout=60)
    assert result.stdout.split() == ["False", "False", "False"]


def test_config_loads_the_registered_contract():
    config = load_config()
    assert config.schema_id == SCHEMA_ID and config.deployment_profile == "spatial-reference/12"
    assert config.z_near_m == 0.05
    assert config.contact.pad_radius_m == 0.5
    assert (config.contact.xy_speed_m_s, config.contact.z_speed_m_s) == (0.35, 0.30)
    assert config.contact.tilt_rad == pytest.approx(math.radians(5.0))
    assert config.contact.rate_rad_s == pytest.approx(math.radians(10.0))
    assert config.action.maximum_m_s2 == (2.5, 2.5, 2.0) and config.action.frame == "world_enu"
    assert [config.model.head(arm) for arm in contracts.ARMS] == [110, 57, 74, 66]
    assert config.experiment_proposal.registered is False
    counts = design_parameter_counts(config.model)
    assert {arm: c["total"] for arm, c in counts.items()} == DESIGN_PARAMETER_TOTALS == {
        "raw_ppo": 4022, "semantic_mlp_ppo": 4008,
        "untyped_gnn_ppo": 4020, "ontology_rgat_ppo": 4016}
    assert counts["untyped_gnn_ppo"]["encoder"] == 306
    assert counts["ontology_rgat_ppo"]["encoder"] == 566
    assert {c["critic"] + c["log_std"] for c in counts.values()} == {1269}


def _write_variant(tmp_path, edit):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("edit,match", [
    (lambda d: d.update(surprise=1), "unknown keys"),
    (lambda d: d["contact"].update(pad_half_width_m=0.5), "unknown keys"),
    (lambda d: d["model"].pop("critic_hidden"), "missing keys"),
    (lambda d: d["feature_scales"].pop("image_rate_per_s"), "missing keys"),
    (lambda d: d.update(schema_id="current_semantics_v1"), "schema_id"),
    (lambda d: d.update(deployment_profile="spatial-reference/11"), "deployment_profile"),
    (lambda d: d["action"].update(frame="body_flu"), "world_enu"),
    (lambda d: d["action"].update(maximum_m_s2=[2.5, 2.0]), "3 values"),
    (lambda d: d["contact"].update(tilt_deg=-5.0), "positive"),
    (lambda d: d["model"]["heads"].update(raw_ppo=111), "budget"),
    (lambda d: d["experiment_proposal"].update(registered=True), "registered"),
])
def test_config_refuses_unknown_missing_and_unregistered_values(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write_variant(tmp_path, edit))


def test_manifest_is_plain_json_and_incomplete_parts_carry_no_hash():
    manifest = contract_manifest(load_config())
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["schema_id"] == SCHEMA_ID
    assert len(manifest["observation"]["channels"]) == 21
    assert manifest["provenance"]["baseline_3d_sha"] == contracts.BASELINE_3D_SHA
    for key in ("observation_hash", "calibration_hash", "action_hash"):
        assert re.fullmatch(r"[0-9a-f]{64}", manifest[key])
    assert manifest["complete"] is (not manifest["missing_parts"])
    assert set(manifest["missing_parts"]) <= set(contracts.PLUGGABLE_PARTS)
    if "features" in manifest["missing_parts"]:
        assert manifest["features"] is None and manifest["feature_hash"] is None
    if "graph" in manifest["missing_parts"]:
        assert manifest["graph"] is None and manifest["graph_hash"] is None
    if not manifest["complete"]:
        assert manifest["algorithm_hash"] is None
    again = contract_manifest(load_config())
    assert {k: v for k, v in again.items() if k.endswith("_hash")} == \
        {k: v for k, v in manifest.items() if k.endswith("_hash")}


def test_registered_contact_definitions_are_thrust_axis_tilt_and_roll_pitch_rate():
    roll, pitch, yaw = 0.4, -0.3, 1.1
    thrust_axis = Rotation.from_euler("xyz", [roll, pitch, yaw]).apply([0.0, 0.0, 1.0])
    assert contact_tilt_rad(roll, pitch) == pytest.approx(math.acos(thrust_axis[2]), abs=1e-12)
    assert abs(contact_tilt_rad(roll, pitch) - math.hypot(roll, pitch)) > 1e-3
    assert contact_rate_rad_s(0.3, -0.4) == pytest.approx(0.5)
    batch = contact_tilt_rad(np.array([0.0, 0.1]), np.array([0.2, 0.0]))
    np.testing.assert_allclose(batch, [0.2, 0.1], atol=1e-12)
