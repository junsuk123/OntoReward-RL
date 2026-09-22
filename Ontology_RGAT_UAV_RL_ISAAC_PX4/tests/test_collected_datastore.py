"""What the accumulating store may reuse, and what it must never reuse."""
from pathlib import Path

import numpy as np
import pytest

from config_loader import load_config
from ontology_rgat.datastore import (KIND_FOV_RISK, KIND_KEYPOINT_CALIBRATION,
                                     ON_POLICY_KINDS, CollectedDataStore,
                                     data_fingerprint, open_datastore,
                                     split_episodes)
from ontology_rgat.rgat import (build_fov_risk_dataset,
                               fov_risk_data_fingerprint, split_by_episode,
                               split_by_episode_ids)
from ontology_rgat.rgat.fov_graph import FOV_GRAPH_INPUT_DIM, FOV_NODE_NAMES


ROOT = Path(__file__).resolve().parents[1]


def _payload(samples=4, value=0.25):
    return {"graph_X": np.full((samples, 3), float(value), dtype=np.float32),
            "geometric_in_fov": np.ones(samples, dtype=bool)}


def _store(tmp_path, name="collected.sqlite3", run_id="run-a"):
    return CollectedDataStore(tmp_path / name, run_id=run_id)


def test_on_policy_data_is_refused_rather_than_accumulated(tmp_path):
    """PPO's update is only valid for data drawn from the policy it updates.

    Replaying a stored rollout into it is wrong however the samples are
    weighted, so the store refuses instead of leaving that to a caller.
    """
    assert "ppo_rollout" in ON_POLICY_KINDS
    with _store(tmp_path) as store:
        for kind in sorted(ON_POLICY_KINDS):
            with pytest.raises(ValueError, match="on-policy"):
                store.store_episode(kind, "fp", seed=1, payload=_payload(),
                                    identity={"seed": 1})
        assert store.episodes("ppo_rollout", "fp") == []


def test_recollecting_an_episode_does_not_add_a_second_copy(tmp_path):
    """Resuming an interrupted collection must be idempotent, not double-weighted."""
    with _store(tmp_path) as store:
        identity = {"seed": 70001, "source_checkpoint_sha256": "abc"}
        first, inserted = store.store_episode(
            KIND_FOV_RISK, "fp", seed=70001, payload=_payload(),
            identity=identity, environment_steps=300)
        assert inserted is True
        again, inserted = store.store_episode(
            KIND_FOV_RISK, "fp", seed=70001, payload=_payload(value=0.9),
            identity=identity, environment_steps=300)
        assert again == first and inserted is False
        assert len(store.episodes(KIND_FOV_RISK, "fp")) == 1
        # A genuinely different flight of the same seed -- another policy --
        # is a new episode, not a duplicate.
        _, inserted = store.store_episode(
            KIND_FOV_RISK, "fp", seed=70001, payload=_payload(),
            identity={"seed": 70001, "source_checkpoint_sha256": "def"})
        assert inserted is True
        assert len(store.episodes(KIND_FOV_RISK, "fp")) == 2


def test_the_split_is_frozen_on_the_seed_and_does_not_move_as_data_arrives(tmp_path):
    """A split that is redrawn as the store grows trains on yesterday's holdout.

    The validation loss that selects the frozen artifact would then be
    measured on data its own lineage had already fitted -- the same bias as
    replaying data onto a resumed checkpoint.
    """
    with _store(tmp_path) as store:
        for seed in range(70000, 70040):
            store.store_episode(
                KIND_FOV_RISK, "fp", seed=seed, payload=_payload(),
                identity={"seed": seed, "source_checkpoint_sha256": "abc"})
        early = {episode.seed: episode.split(0.25)
                 for episode in store.episodes(KIND_FOV_RISK, "fp")}
        assert 0 < list(early.values()).count("validation") < len(early)

        # Ten more episodes, and a second flight of every original seed.
        for seed in range(70040, 70050):
            store.store_episode(
                KIND_FOV_RISK, "fp", seed=seed, payload=_payload(),
                identity={"seed": seed, "source_checkpoint_sha256": "abc"})
        for seed in range(70000, 70040):
            store.store_episode(
                KIND_FOV_RISK, "fp", seed=seed, payload=_payload(),
                identity={"seed": seed, "source_checkpoint_sha256": "def"})

        later = store.episodes(KIND_FOV_RISK, "fp")
        for episode in later:
            if episode.seed in early:
                # Same side as before, and both flights of one initial
                # condition on the same side as each other.
                assert episode.split(0.25) == early[episode.seed]
        training, validation = split_episodes(later, validation_fraction=0.25)
        assert not ({episode.seed for episode in training}
                    & {episode.seed for episode in validation})


def test_a_different_architecture_sees_none_of_the_old_accumulation(tmp_path):
    with _store(tmp_path) as store:
        store.store_episode(KIND_FOV_RISK, "old-fp", seed=1,
                            payload=_payload(), identity={"seed": 1})
        assert len(store.episodes(KIND_FOV_RISK, "old-fp")) == 1
        assert store.episodes(KIND_FOV_RISK, "new-fp") == []
        # The row is kept for audit rather than deleted.
        assert store.summary(KIND_FOV_RISK, "old-fp")["episodes"] == 1


def test_a_payload_round_trips_and_the_summary_counts_what_was_flown(tmp_path):
    with _store(tmp_path) as store:
        store.record_run(experiment="two_pipeline_fov_risk", config_hash="cfg")
        store.store_episode(
            KIND_FOV_RISK, "fp", seed=7, payload=_payload(samples=5),
            identity={"seed": 7}, environment_steps=300,
            provenance={"source_pipeline": "shin_se_fixed"})
        episode, = store.episodes(KIND_FOV_RISK, "fp")
        payload = episode.payload()
        np.testing.assert_allclose(payload["graph_X"], _payload(5)["graph_X"])
        assert payload["geometric_in_fov"].dtype == bool
        assert episode.provenance["source_pipeline"] == "shin_se_fixed"
        summary = store.summary(KIND_FOV_RISK, "fp")
        assert summary["episodes"] == 1 and summary["samples"] == 5
        assert summary["environment_steps"] == 300 and summary["seeds"] == [7]


def test_the_consumption_ledger_records_how_an_artifact_was_trained(tmp_path):
    """An artifact fine-tuned on an accumulation has seen its old data more.

    That cannot be undone after the fact, so it is at least written down.
    """
    with _store(tmp_path) as store:
        for seed in range(3):
            store.store_episode(KIND_FOV_RISK, "fp", seed=seed,
                                payload=_payload(), identity={"seed": seed})
        episodes = store.episodes(KIND_FOV_RISK, "fp")
        assert store.record_consumption(
            "artifact-sha", episodes, validation_fraction=0.25,
            trained_from_scratch=True) == 3
        assert store.consumed_by("artifact-sha") == {
            episode.uid for episode in episodes}
        assert store.consumed_by("another-artifact") == set()


def test_a_second_store_reopens_the_same_accumulation(tmp_path):
    path = tmp_path / "collected.sqlite3"
    with CollectedDataStore(path, run_id="run-a") as first:
        first.store_episode(KIND_FOV_RISK, "fp", seed=5, payload=_payload(),
                            identity={"seed": 5})
    with CollectedDataStore(path, run_id="run-b") as second:
        assert len(second.episodes(KIND_FOV_RISK, "fp")) == 1
        _, inserted = second.store_episode(
            KIND_FOV_RISK, "fp", seed=5, payload=_payload(), identity={"seed": 5})
        assert inserted is False
        assert second.summary(KIND_FOV_RISK, "fp")["runs"] == ["run-a"]
    assert open_datastore(None) is None


DECKS = ("straight_escape_burst",)


def test_the_fov_fingerprint_tracks_meaning_not_optimizer_settings():
    """Reuse must survive a learning-rate edit and must not survive a new camera."""
    system = load_config(ROOT / "config/shin2026-system.yaml")
    base = fov_risk_data_fingerprint(
        system, prediction_steps=10, control_hz=10.0, scenarios=DECKS,
        keypoint_implementation="encoder-v3")

    tuned = {**system, "ppo": {"learning_rate": 1e-9},
             "training": {"episodes_full": 1}}
    assert fov_risk_data_fingerprint(
        tuned, prediction_steps=10, control_hz=10.0, scenarios=DECKS,
        keypoint_implementation="encoder-v3") == base

    for changed in (
            {"prediction_steps": 20, "control_hz": 10.0},
            {"prediction_steps": 10, "control_hz": 20.0}):
        assert fov_risk_data_fingerprint(
            system, scenarios=DECKS, keypoint_implementation="encoder-v3",
            **changed) != base
    assert fov_risk_data_fingerprint(
        system, prediction_steps=10, control_hz=10.0, scenarios=DECKS,
        keypoint_implementation="encoder-v4") != base

    camera = {**system, "vision": {**system["vision"], "camera": {
        **system["vision"]["camera"], "pitch_down_deg": 45.0}}}
    assert fov_risk_data_fingerprint(
        camera, prediction_steps=10, control_hz=10.0, scenarios=DECKS,
        keypoint_implementation="encoder-v3") != base

    deck = {**system, "pad": {**system["pad"], "speed_max_m_s": 99.0}}
    assert fov_risk_data_fingerprint(
        deck, prediction_steps=10, control_hz=10.0, scenarios=DECKS,
        keypoint_implementation="encoder-v3") != base


def test_a_different_deck_rotation_retires_the_accumulation():
    """The deck is the trajectory distribution, not a tuning knob.

    An episode on a pad that stays in frame teaches the future-FOV target
    something different from one that drives out of it, so a burst-only
    collection must not silently extend a six-deck accumulation. Order matters
    because the rotation is indexed by episode.
    """
    system = load_config(ROOT / "config/shin2026-system.yaml")

    def fingerprint(scenarios):
        return fov_risk_data_fingerprint(
            system, prediction_steps=10, control_hz=10.0,
            scenarios=scenarios, keypoint_implementation="encoder-v4")

    burst = fingerprint(("straight_escape_burst",))
    assert fingerprint(["straight_escape_burst"]) == burst   # list or tuple
    assert fingerprint(("training_random_walk",)) != burst
    assert fingerprint(("straight_escape_burst", "circle")) != burst
    assert fingerprint(("circle", "straight_escape_burst")) != fingerprint(
        ("straight_escape_burst", "circle"))


def test_a_fingerprint_needs_properties_and_ignores_key_order():
    assert data_fingerprint({"a": 1, "b": [2, 3]}) == data_fingerprint(
        {"b": [2, 3], "a": 1})
    with pytest.raises(ValueError, match="at least one property"):
        data_fingerprint({})


def _fov_dataset(episode_seeds, *, steps=14):
    rng = np.random.default_rng(3)
    episodes = []
    for index, seed in enumerate(episode_seeds, start=1):
        visible = rng.random(steps) > (0.3 if index % 2 else 0.9)
        episodes.append({
            "episode_id": index, "seed": int(seed),
            "samples": [{
                "graph_X": rng.random(
                    (FOV_GRAPH_INPUT_DIM, len(FOV_NODE_NAMES))).astype(np.float32),
                "geometric_in_fov": bool(value)} for value in visible],
        })
    return build_fov_risk_dataset(episodes, prediction_steps=3)


def test_an_explicit_split_holds_the_named_episodes_out():
    dataset = _fov_dataset([70000, 70001, 70002, 70003])
    training, validation = split_by_episode_ids(dataset, [2, 4])
    meta = np.asarray(dataset["meta"])
    assert set(meta[validation, 0]) == {2, 4}
    assert set(meta[training, 0]) == {1, 3}
    assert not (training & validation).any()
    # The seeded per-run draw stays available for a one-off dataset.
    seeded_training, seeded_validation = split_by_episode(
        dataset, validation_fraction=0.25, seed=1)
    assert seeded_training.any() and seeded_validation.any()


def test_an_explicit_split_refuses_to_leave_a_side_empty():
    dataset = _fov_dataset([70000, 70001])
    with pytest.raises(ValueError, match="leaves one side empty"):
        split_by_episode_ids(dataset, [])
    with pytest.raises(ValueError, match="deliberately not redrawn"):
        split_by_episode_ids(dataset, [1, 2])
    with pytest.raises(ValueError, match="not in this dataset"):
        split_by_episode_ids(dataset, [99])


def test_a_stored_fov_episode_rebuilds_the_labeller_input():
    """The store keeps the observation, not the label.

    A horizon change must re-derive the target rather than reuse one that no
    longer means what it says.
    """
    import run_three_pipeline as runner

    dataset = _fov_dataset([70000, 70001])
    episodes = []
    rng = np.random.default_rng(11)
    for episode_id, seed in enumerate((70000, 70001), start=1):
        samples = [{"graph_X": rng.random(
            (FOV_GRAPH_INPUT_DIM, len(FOV_NODE_NAMES))).astype(np.float32),
            "geometric_in_fov": bool(index % 3)} for index in range(14)]
        payload = runner._fov_episode_payload(samples)
        restored = runner._fov_episode_from_payload(
            payload, episode_id=episode_id, seed=seed)
        assert restored["seed"] == seed and len(restored["samples"]) == 14
        for original, round_tripped in zip(samples, restored["samples"]):
            np.testing.assert_allclose(original["graph_X"],
                                       round_tripped["graph_X"])
            assert original["geometric_in_fov"] == round_tripped["geometric_in_fov"]
        episodes.append(restored)

    short = build_fov_risk_dataset(episodes, prediction_steps=3)
    long = build_fov_risk_dataset(episodes, prediction_steps=6)
    assert short["y"].shape == long["y"].shape
    assert int(short["valid"].sum()) > int(long["valid"].sum())
    assert dataset["X"].shape[1:] == short["X"].shape[1:]
