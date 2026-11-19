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
