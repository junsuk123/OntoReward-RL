"""Run one episode as its own set of processes, then take it all down.

An episode is a complete life: a simulator start, a collector and a rosbag
attached to it, a fixed duration, and a full shutdown. Nothing is reused between
episodes -- not the stage, not the ROS graph, not DDS discovery -- because the
point of the plan is that each scenario is observed in isolation.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

from simlab.config.loader import deep_merge, read_yaml
from simlab.config.schema import SceneConfig
from simlab.utils.paths import PROJECT_ROOT, resolve_path

#: Recorded so an episode can be replayed without the simulator.
BAG_TOPICS: Sequence[str] = (
    "/clock", "/tf",
    "/simlab/dataset/front_near/image", "/simlab/front_near/camera_info",
    "/simlab/dataset/front_far/image", "/simlab/front_far/camera_info",
    "/simlab/dataset/satellite_nadir/image", "/simlab/satellite_nadir/camera_info",
)


@dataclass(frozen=True)
class EpisodePlan:
    """One line of the plan, resolved to concrete paths and a seed."""

    index: int
    scenario: str
    environment: str
    seed: int
    duration_s: float
    episode_id: str
    config_path: Path
    raw_dir: Path
    bag_dir: Path
    run_dir: Path

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "scenario": self.scenario,
            "environment": self.environment,
            "seed": self.seed,
            "duration_s": self.duration_s,
            "episode_id": self.episode_id,
            "config": str(self.config_path),
            "raw": str(self.raw_dir),
            "bag": str(self.bag_dir),
        }


def stop(process: subprocess.Popen | None, grace_s: float = 15.0) -> None:
    """Ask politely, then insist. A half-dead collector poisons the next run."""
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def build_plan(scene: SceneConfig, stamp: str, base_seed: int) -> List[EpisodePlan]:
    """Expand ``scenarios.plan`` into episodes with stable ids and seeds."""
    run_root = resolve_path(scene.scenarios.run_dir) / stamp
    raw_root = resolve_path(scene.scenarios.dataset_root)
    bag_root = resolve_path("artifacts/perception/bags") / stamp
    plans: List[EpisodePlan] = []
    for index, entry in enumerate(scene.scenarios.episodes()):
        seed = entry.seed if entry.seed is not None else base_seed * 1000 + index
        # The session id leads with the run stamp so ingestion, which walks the
        # raw directory in name order, replays episodes chronologically.
        episode_id = f"{stamp}_{index:02d}_{entry.scenario}_{entry.environment}"
        plans.append(
            EpisodePlan(
                index=index,
                scenario=entry.scenario,
                environment=entry.environment,
                seed=seed,
                duration_s=entry.duration_s or scene.scenarios.episode_duration_s,
                episode_id=episode_id,
                config_path=run_root / episode_id / "episode.yaml",
                raw_dir=raw_root / episode_id,
                bag_dir=bag_root / episode_id,
                run_dir=run_root,
            )
        )
    return plans


def adhoc_plan(
    scene: SceneConfig, stamp: str, duration_s: float, seed: int
) -> EpisodePlan:
    """A single collection run with no scenario -- the plain crossing shuttle."""
    run_root = resolve_path(scene.scenarios.run_dir) / stamp
    return EpisodePlan(
        index=0,
        scenario="",
        environment=scene.world.environment,
        seed=seed,
        duration_s=duration_s,
        episode_id=stamp,
        config_path=run_root / stamp / "episode.yaml",
        raw_dir=resolve_path(scene.scenarios.dataset_root) / stamp,
        bag_dir=resolve_path("artifacts/perception/bags") / stamp,
        run_dir=run_root,
    )


def write_episode_config(plan: EpisodePlan, scene_config_path: str | Path) -> Path:
    """Freeze this episode into one YAML every process of it will load.

    The collector rebuilds the blackout schedule and the occluder positions from
    this file rather than from the simulator, so it has to be written before
    anything starts and never touched again.

    An empty ``plan.scenario`` means no scenario: the episode still gets its own
    frozen config and its own session, it just flies the plain shuttle.
    """
    base = read_yaml(scene_config_path)
    overrides = {
        "app": {"headless": True, "duration_s": plan.duration_s},
        "world": {"environment": plan.environment, "environment_seed": plan.seed},
        "scenarios": {
            "enabled": bool(plan.scenario),
            "active": plan.scenario or None,
            "seed": plan.seed,
            "episode_id": plan.episode_id,
            "episode_duration_s": plan.duration_s,
            # Absolute, and stamped with this run: the simulator writes its
            # episode summary here and the orchestrator reads it back.
            "run_dir": str(plan.run_dir),
        },
        "ros2": {"enabled": True},
    }
    merged = deep_merge(base, overrides)
    SceneConfig.from_dict(merged)  # fail here, not three processes later
    plan.config_path.parent.mkdir(parents=True, exist_ok=True)
    plan.config_path.write_text(
        yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return plan.config_path


def run_episode(
    plan: EpisodePlan,
    sample_period_s: float,
    min_visibility: float,
    timeout_s: float,
    record_bag: bool = True,
    extra_launch_args: Sequence[str] = (),
) -> Dict[str, Any]:
    """Start collector, recorder and simulator; wait; shut all three down."""
    python = Path(os.environ.get("SIMLAB_PERCEPTION_PYTHON", sys.executable))
    plan.raw_dir.mkdir(parents=True, exist_ok=True)
    collector = subprocess.Popen(
        [
            str(python), "-m", "simlab.ros.dataset_collector_node", "--ros-args",
            "-p", "use_sim_time:=true", "-p", f"config:={plan.config_path}",
            "-p", f"output:={plan.raw_dir}", "-p", f"sample_period_s:={sample_period_s}",
            "-p", f"min_visibility:={min_visibility}",
        ],
        cwd=PROJECT_ROOT,
    )
    recorder = None
    if record_bag:
        plan.bag_dir.parent.mkdir(parents=True, exist_ok=True)
        recorder = subprocess.Popen(
            [
                "ros2", "bag", "record", "--compression-mode", "file",
                "--compression-format", "zstd", "-o", str(plan.bag_dir), *BAG_TOPICS,
            ],
            cwd=PROJECT_ROOT,
        )

    simulation = None
    started = time.monotonic()
    status = "ok"
    code = None
    try:
        simulation = subprocess.Popen(
            [
                "ros2", "launch", "launch/simlab.launch.py",
                f"config:={plan.config_path}", "headless:=true",
                f"seconds:={plan.duration_s}", "rviz:=false",
                "pipeline:=false", "yolo:=false", *extra_launch_args,
            ],
            cwd=PROJECT_ROOT,
        )
        try:
            code = simulation.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            status = "timeout"
            print(f"[episode] {plan.episode_id}: exceeded {timeout_s:g}s, shutting it down")
        if code:
            status = f"exit_{code}"
    finally:
        stop(simulation)
        stop(collector)
        stop(recorder)

    session = plan.raw_dir / "session.json"
    summary = plan.run_dir / plan.episode_id / "episode_summary.json"
    result = {
        **plan.as_dict(),
        "status": status,
        "exit_code": code,
        "wall_seconds": round(time.monotonic() - started, 1),
        "session": json.loads(session.read_text(encoding="utf-8")) if session.is_file() else None,
        "summary": (
            json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else None
        ),
    }
    return result


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
