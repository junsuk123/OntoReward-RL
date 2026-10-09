from pathlib import Path

import numpy as np
import pytest

from ontology_rgat.direct_policy.isaac_adapter import CausalIsaacObservation
from ontology_rgat.direct_policy.backends import DirectIsaacBackend, backend_factory
from ontology_rgat.direct_policy.contracts import make_contract
from ontology_rgat.spatial.environment import Truth
from ontology_rgat.spatial.core import Measurement
from ontology_rgat.spatial.runtime_contract import deployment_manifest
from ontology_rgat.spatial.scenarios import sample_planar_scenario, sample_spatial_scenario


def measurement(t=1.0, sample=1, optical=True):
    return Measurement(
        time_s=t, own_position=np.array([2.0, 0.0, 3.0]),
        own_velocity=np.array([0.5, 0.0, -0.1]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate=np.zeros(3),
        optical_position=(np.array([1.0, 0.0, 2.4]) if optical else None),
        confidence=0.8 if optical else 0.0, sample_id=sample,
        optical_time_s=t-0.05 if optical else None,
        optical_own_position=np.array([2.0, 0.0, 3.0]) if optical else None,
        optical_quaternion=np.array([1.0, 0.0, 0.0, 0.0]) if optical else None)


def test_deployed_observation_is_causal_and_episode_scoped():
    adapter = CausalIsaacObservation(2)
    adapter.reset("a")
    adapter.ingest(measurement())
    vector, provenance = adapter.vector()
    assert vector.shape == (12,)
    assert provenance["capture_stamp_s"] == 0.95
    assert vector[8] == 1.0
    adapter.ingest(measurement(t=1.1, sample=1, optical=True))
    stale, _ = adapter.vector()
    assert stale[8] == 0.0
    adapter.reset("b")
    assert adapter.last_sample_id == -1


def test_planar_scenario_preserves_draws_but_projects_heading():
    spatial = sample_spatial_scenario(42)
    planar = sample_planar_scenario(42)
    assert (planar.t1, planar.t2, planar.v0, planar.a2) == (
        spatial.t1, spatial.t2, spatial.v0, spatial.a2)
    assert planar.heading == 0.0


def test_planar_profile_is_explicit_and_resolved():
    root = Path(__file__).resolve().parents[2]
    deployment = deployment_manifest(root/"config"/"matlab-port-planar-isaac.yaml")
    benchmark = deployment["resolved_scientific_configuration"]["benchmark"]
    assert benchmark["spatial_scenario"] == "matlab_planar_cv_ca_cv"
    assert benchmark["planar_entry"] is True


def test_backend_factory_fails_closed_instead_of_substituting_local(tmp_path):
    with pytest.raises(ValueError, match="deployment manifest"):
        backend_factory(make_contract(3, backend="isaac"))
    with pytest.raises(ValueError, match="replay-fixture"):
        backend_factory(make_contract(2, backend="replay"))
    local = backend_factory(make_contract(2, backend="local"))()
    assert local.name == "local-reference"
    local.close()


class FakeFlight:
    def __init__(self):
        self.m = measurement()
        self.finished = False

    def reset(self, seed):
        return self.m, Truth(np.array([1.0, 0.0, 2.4]), np.zeros(3),
                             np.zeros(2), np.zeros(3), False)

    def advance(self, command, callback):
        self.m = measurement(t=self.m.time_s+0.1,
                             sample=self.m.sample_id+1, optical=True)
        callback(self.m)
        return Truth(np.array([0.9, 0.0, 2.3]), np.zeros(3),
                     np.zeros(2), np.zeros(3), False)

    def finish(self):
        self.finished = True
        return True

    def close(self):
        pass


def test_authority_uses_the_observation_that_caused_the_command(tmp_path):
    root = Path(__file__).resolve().parents[2]
    deployment = deployment_manifest(root/"config"/"matlab-port-planar-isaac.yaml")
    flight = FakeFlight()
    backend = DirectIsaacBackend(make_contract(2, backend="isaac"),
                                 deployment=deployment, flight=flight)
    backend.set_policy_version(7)
    backend.reset(10000)
    result = backend.step(np.zeros(2))
    authority = result.info["authority"]
    assert result.info["transition_s"] > 0.0
    assert authority["policy_version"] == 7
    assert authority["capture_stamp_s"] == 0.95
    assert authority["capture_stamp_s"] <= authority["receive_stamp_s"]
    assert authority["receive_stamp_s"] <= authority["decision_stamp_s"]
    backend.close()
    assert flight.finished, "truncated/failed rollouts must still invoke owned cleanup"
