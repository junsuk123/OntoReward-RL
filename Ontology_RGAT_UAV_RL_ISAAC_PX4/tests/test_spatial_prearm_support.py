"""Physical setup support regressions; no policy-time assistance."""
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac_sim"))
from prearm_support import leveling_torque


def test_no_setup_torque_under_autopilot_control():
    np.testing.assert_array_equal(
        leveling_torque([0.5, 0.5, 0.5, 0.5], [2, 3, 4], controlled=True), np.zeros(3)
    )


@pytest.mark.parametrize("degrees", [(15, 0, 0), (0, -40, 80), (85, 35, 10)])
def test_tilted_disarmed_body_levels_with_physical_torque(degrees):
    rotation = Rotation.from_euler("xyz", degrees, degrees=True)
    rate = np.zeros(3)
    inertia = np.array([0.03, 0.03, 0.055])
    for _ in range(2500):
        torque = leveling_torque(rotation.as_quat(), rate, controlled=False)
        assert np.linalg.norm(torque) <= 0.4 + 1e-12
        rate += 0.004 * (torque - np.cross(rate, inertia * rate)) / inertia
        rotation = rotation * Rotation.from_rotvec(0.004 * rate)
    assert np.linalg.norm(rotation.as_euler("xyz")[:2]) < np.deg2rad(0.1)
    assert np.linalg.norm(rate) < 0.005


def test_level_body_has_no_artificial_torque():
    q = Rotation.from_euler("z", 1.2).as_quat()
    np.testing.assert_allclose(
        leveling_torque(q, [0, 0, 0], controlled=False), 0, atol=1e-12
    )


def test_armed_auto_land_never_reenables_prearm_hover_support():
    import ast
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / "isaac_sim/landing_world.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_hold_prearm_start"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    # Deliberately no vehicle/force API: AUTO.LAND must return before it can
    # apply any support or inspect physical pose.
    state = SimpleNamespace(
        start_airborne=True, autopilot_armed=True, autopilot_flying=False,
        has_policy_handover=True
    )
    namespace["_hold_prearm_start"](state, 0.004)


def test_armed_preentry_gap_retains_support_without_policy_assistance():
    import ast
    from types import SimpleNamespace
    source = Path(__file__).resolve().parents[1] / "isaac_sim/landing_world.py"
    method = next(node for node in ast.walk(ast.parse(source.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == '_hold_prearm_start')
    ns = dict(np=np, Rotation=Rotation, GRAVITY_M_S2=9.80665,
              leveling_torque=leveling_torque)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'), ns)
    forces = []
    vehicle = SimpleNamespace(state=SimpleNamespace(position=np.array([0.,0.,2.]),
        linear_velocity=np.zeros(3), attitude=np.array([0.,0.,0.,1.]),
        angular_velocity=np.zeros(3)), apply_force=lambda f,**_:forces.append(f),
        apply_torque=lambda *a,**kw:pytest.fail('no setup torque while armed'))
    state = SimpleNamespace(start_airborne=True, autopilot_armed=True,
        autopilot_flying=False, has_policy_handover=False, vehicle=vehicle,
        deck=SimpleNamespace(world_from_pad=lambda p:np.asarray(p),velocity=np.zeros(3)),
        hover_start_pad_m=[0.,0.,2.],hover_hold_mass_kg=1.5,spatial_clock_pub=object())
    ns['_hold_prearm_start'](state,.004)
    np.testing.assert_allclose(forces, [[0.,0.,1.5*9.80665]])
    state.has_policy_handover = True
    ns['_hold_prearm_start'](state,.004)
    assert len(forces) == 1
