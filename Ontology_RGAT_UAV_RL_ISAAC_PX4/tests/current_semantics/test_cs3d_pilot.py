"""G4 pilot end to end on a tiny registered variant (never the real budget in tests)."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from ontology_rgat.current_semantics.contracts import DEFAULT_CONFIG_PATH, load_config
from ontology_rgat.current_semantics.contracts import SuccessGatedCurriculum
from ontology_rgat.current_semantics.pilot import Curriculum


PROJECT = Path(__file__).resolve().parents[2]
RUN_SH = PROJECT.parent / "run.sh"
CONFIG = load_config()


def test_curriculum_promotes_only_on_a_full_window_of_success_and_never_demotes():
    c = Curriculum(SuccessGatedCurriculum(start_difficulty=0.0, step=0.1, window_updates=3,
                                          promote_at_success=0.5))
    assert [c.record(s) for s in (0.9, 0.9)] == [False, False] and c.difficulty == 0.0
    assert c.record(0.0) is True and c.difficulty == pytest.approx(0.1)   # mean 0.6
    assert [c.record(s) for s in (0.2, 0.2, 0.2)] == [False] * 3 and c.difficulty == 0.1
    assert [c.record(s) for s in (1.0, 1.0)] == [False, True]             # window slides
    assert c.difficulty == pytest.approx(0.2)
    for _ in range(40):
        c.record(1.0)
    assert c.difficulty == 1.0 and c.record(0.0) is False and c.difficulty == 1.0
    assert c.record(None) is False


def test_registered_protocol_is_pilot_7():
    g4 = CONFIG.g4
    assert g4.protocol == "g4-pilot/7" and g4.updates == 600
    assert g4.training_randomization.handover_own_state_error.horizontal_max_m_s == 1.5
    assert g4.training_randomization.vision_blackout.probability == 0.5
    assert (g4.curriculum.step, g4.curriculum.window_updates,
            g4.curriculum.promote_at_success) == (0.1, 5, 0.5)
    assert (g4.criteria.deterministic_success_min, g4.criteria.sampled_success_min) \
        == (0.5, 0.25)


def test_registered_seeds_are_disjoint_and_outside_recorded_use():
    g4 = CONFIG.g4
    train = {g4.train_seed(m, u, e) for m in g4.model_seeds for u in range(g4.updates)
             for e in range(g4.episodes_per_update)}
    assert len(train) == len(g4.model_seeds) * g4.updates * g4.episodes_per_update
    for family in (g4.validation_seeds, g4.test_seeds, CONFIG.g3.pilot_seeds):
        assert not train & set(family)
    assert not set(g4.validation_seeds) & set(g4.test_seeds)


def tiny(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    data["g4_pilot"].update(updates=3, episodes_per_update=2, validate_every=1,
                            validation_seeds=[992001], test_seeds=[993001], workers=2,
                            model_seeds=[0])
    data["g4_pilot"]["curriculum"].update(step=0.5, window_updates=1, promote_at_success=0.0)
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_learnability_is_refused_until_g2_and_g3_pass(tmp_path):
    done = subprocess.run(["bash", str(RUN_SH), "current-semantics-3d", "--stage",
                           "learnability", "--run-root", str(tmp_path / "x")],
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 3 and "G3" in done.stdout
    assert not (tmp_path / "x").exists()


def test_tiny_pilot_trains_validates_tests_and_judges_through_the_cli(tmp_path):
    config = tiny(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    # G3 artifacts of the real run root carry this config's g3_hash (g4 changes only).
    real = PROJECT / "results/current_semantics_3d/spatial_current_semantics_v1"
    for name in ("coupling_local.json", "coupling_isaac.json"):
        if not (real / name).exists():
            pytest.skip("needs the recorded G3 artifacts")
        shutil.copy(real / name, root / name)
    done = subprocess.run(["bash", str(RUN_SH), "current-semantics-3d", "--stage",
                           "learnability", "--config", str(config), "--run-root", str(root)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    report = json.loads((root / "g4_pilot.json").read_text())
    assert report["protocol_complete"] and report["arm"] == "raw_ppo"
    assert set(report["tests"]["0"]) >= {"deterministic", "sampled"}
    seed = root / "g4_pilot" / "seed0"
    assert (seed / "best.pt").exists() and (seed / "final.pt").exists()
    history = [json.loads(line) for line in (seed / "history.jsonl").read_text().splitlines()]
    assert [row["update"] for row in history] == [0, 1, 2]
    assert [row["difficulty"] for row in history] == [0.0, 0.5, 1.0]
    assert [row["promoted_to"] for row in history] == [0.5, 1.0, None]
    validation = [json.loads(line) for line in (seed / "validation.jsonl").read_text().splitlines()]
    assert validation and all({"deterministic", "sampled"} <= set(v) for v in validation)
    report_out = subprocess.run(["bash", str(RUN_SH), "current-semantics-3d", "--stage", "report",
                                 "--config", str(config), "--run-root", str(root)],
                                capture_output=True, text=True, timeout=180)
    gates = json.loads(report_out.stdout[report_out.stdout.index("{"):])["gates"]
    assert gates["G4"]["status"] in ("passed", "failed")


def test_only_training_episodes_carry_the_randomisation(monkeypatch):
    from ontology_rgat.current_semantics import pilot
    seen = []
    monkeypatch.setattr(pilot, "run_episode", lambda *a, **k: seen.append(k["train"]) or {})
    pilot._WORKER["config"] = CONFIG
    pilot._work(("raw_ppo", {}, 1, 1.0, True, 1, False))
    pilot._work(("raw_ppo", {}, 1, 0.3, False, 1, True))
    assert seen == [False, True]

    class Pool:
        def map(self, function, tasks):
            self.tasks = list(tasks)
            return [{"seed": t[2], "status": "SUCCESS", "success": True, "return": 0.0,
                     "decisions": 1} for t in self.tasks]
    pool = Pool()
    pilot.evaluate(pool, "raw_ppo", {}, (992001, 992002), deterministic=True, offset=0)
    assert all(task[-1] is False for task in pool.tasks)
