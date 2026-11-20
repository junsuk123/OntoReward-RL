"""The monitor's encoder trace (encoding_view.py) is the encoder's own arithmetic."""
from pathlib import Path

import numpy as np
import pytest
import torch

from ontology_rgat.current_semantics import encoding_view as ev
from ontology_rgat.current_semantics.contracts import (AUDITED_SEED_RANGES,
                                                        DIAGNOSTIC_SEED_RANGES, load_config)
from ontology_rgat.current_semantics.models import CurrentActorCritic

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "config" / "current_semantics" / "spatial_view_contact_v1.yaml"


@pytest.mark.parametrize("arm", ev.GRAPH_ARMS)
def test_trace_matches_the_encoder_and_attention_is_a_softmax(arm):
    model = CurrentActorCritic(arm, load_config(CONFIG), seed=3)
    nodes = torch.randn(5, 8, 10)
    out = ev.trace_encoder(model.encoder, nodes)
    assert np.allclose(out["g"], model.encoder(nodes).detach().numpy(), atol=1e-6)
    hops, edges = model.encoder.hops, len(model.encoder.spec.edges)
    assert out["alpha"].shape == (hops, 5, edges) and out["norm"].shape == (hops + 1, 5, 8)
    target = model.encoder.target.numpy()
    for hop in range(hops):
        sums = np.zeros((5, 8))
        np.add.at(sums.T, target, out["alpha"][hop].T)
        assert np.allclose(sums, 1.0, atol=1e-5)          # every node has a self loop


def test_the_probe_seed_is_outside_every_registered_range():
    for _, first, last, *_ in list(AUDITED_SEED_RANGES) + list(DIAGNOSTIC_SEED_RANGES):
        assert not first <= ev.PROBE_SEED <= last


def test_view_refuses_paths_outside_results_and_non_graph_arms(tmp_path):
    with pytest.raises(ValueError):
        ev.view(tmp_path, "../x/ontology_rgat_ppo")
    (tmp_path / "r" / "raw_ppo").mkdir(parents=True)
    with pytest.raises(ValueError, match="graph-arm"):
        ev.view(tmp_path, "r/raw_ppo")
    (tmp_path / "r" / "ontology_rgat_ppo").mkdir(parents=True)
    assert ev.view(tmp_path, "r/ontology_rgat_ppo")["checkpoint"] is None


def _orl(seed=3):
    from ontology_rgat.current_semantics import orl
    from ontology_rgat.current_semantics.models import build_model
    return build_model(orl.ARMS["orl/1"], load_config(CONFIG), seed=seed).eval()


def test_the_orl_trace_is_the_encoder_and_its_outputs_are_the_actors():
    """ORL (the proposed policy): the traced node states equal ORLEncoder.node_states, every
    node's incoming attention sums to 1, and the shown outputs are the actor's mirror-averaged
    mean with mu_z = hold + gated."""
    from ontology_rgat.current_semantics import orl
    model = _orl()
    flat = torch.randn(4, 44) * 0.5
    out = ev.trace_orl(model, flat)
    edges = len(orl.GRAPHS["orl/1"].edges)
    assert out["alpha"].shape == (orl.HOPS, 4, edges)
    assert out["norm"].shape == (orl.HOPS + 1, 4, len(orl.NODES))
    target = model.encoder.target.numpy()
    for hop in range(orl.HOPS):
        sums = np.zeros((4, len(orl.NODES)))
        np.add.at(sums.T, target, out["alpha"][hop].T)
        assert np.allclose(sums, 1.0, atol=1e-5)
    with torch.no_grad():
        mean = model(torch.zeros(4, 21), flat, torch.zeros(4, 8, 10)).numpy()
    terms = out["terms"]
    assert np.allclose(mean[:, 2], terms["hold"] + terms["gated"], atol=1e-5)


def test_an_orl_folder_is_a_graph_run_and_live_weights_are_viewable(tmp_path):
    """A learning-speed ORL folder (orl_ppo_<hash>) resolves to its arm; the trainer's live.pt
    snapshot is a checkpoint the view loads, with its update and difficulty."""
    from ontology_rgat.current_semantics import orl
    from ontology_rgat.current_semantics.contracts import contract_manifest
    from ontology_rgat.current_semantics.models import save_checkpoint
    arm = orl.ARMS["orl/1"]
    folder = arm.replace(":", "_")
    assert ev.arm_of(folder) == arm and ev.arm_of("raw_ppo") is None
    run = tmp_path / "spatial_current_semantics_v1" / "learning_speed" / "seed40" / folder
    run.mkdir(parents=True)
    config = load_config(CONFIG)
    save_checkpoint(run / "live.pt", _orl(seed=40), manifest=contract_manifest(config),
                    extra={"update": 12, "model_seed": 40, "live": True, "difficulty": 0.3})
    assert ev.newest_checkpoint(run).name == "live.pt"
    payload = ev.view(tmp_path, str(run.relative_to(tmp_path)))
    assert payload["family"] == "orl" and payload["live"] and payload["checkpoint_update"] == 12
    assert len(payload["nodes"]) == len(orl.NODES) == len(payload["layout"])
    assert len(payload["steps"][0]["alpha"]) == orl.HOPS
    assert len(payload["steps"][0]["g"]) == len(payload["g_labels"]) == 6


def test_the_live_snapshot_reads_no_random_state():
    """pilot's live.pt is a save_checkpoint of a state-dict copy: the global and a private
    generator are untouched, so training with snapshots is bit-identical to training without."""
    import tempfile
    from ontology_rgat.current_semantics import pilot
    from ontology_rgat.current_semantics.contracts import contract_manifest
    from ontology_rgat.current_semantics.models import save_checkpoint
    import inspect
    assert inspect.signature(pilot.train_seed).parameters["live_every"].default == pilot.LIVE_EVERY
    model = _orl()
    generator = torch.Generator().manual_seed(7)
    before = (torch.get_rng_state().clone(), generator.get_state().clone(),
              {k: v.clone() for k, v in model.state_dict().items()})
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(Path(tmp) / pilot.LIVE_CHECKPOINT, model,
                        manifest=contract_manifest(load_config(CONFIG)), extra={"update": 0})
    assert torch.equal(torch.get_rng_state(), before[0])
    assert torch.equal(generator.get_state(), before[1])
    assert all(torch.equal(v, before[2][k]) for k, v in model.state_dict().items())
