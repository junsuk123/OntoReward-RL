"""Dataset, R-GAT, both PPO runs, evaluation and figures against Isaac/PX4.

Every stage writes its artefact before the next one begins, so a failure late
in a long sweep does not throw away the hours before it.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from .bridge import PX4Bridge
from .config import Config
from .evaluation.compare import LABELS, compare_policies, write_table
from .evaluation.plots import make_plots
from .evaluation.sweeps import battery_sweep, gnss_sweep, pad_sweep, wind_sweep
from .ppo.networks import load_agent, save_agent
from .ppo.train import train_ppo
from .rgat.dataset import generate_dataset, load_dataset, merge_datasets, save_dataset
from .rgat.model import load_potential, save_potential
from .rgat.train import train_potential
from .viz.dashboard import Dashboard
from .viz.live import (STORE, DatasetMonitor, EpisodeMonitor, PPOMonitor,
                       RGATMonitor, RewardMonitor)
from .viz.rviz import RvizPublisher

__all__ = ["run_all"]


def _banner(step: int, title: str) -> None:
    print(f"\n=== {step}) {title} ===")


def run_all(cfg: Config) -> dict[str, Any]:
    """Run every stage. Assumes the external stack is already up."""
    np.random.seed(cfg.seed)
    started = time.perf_counter()
    results_dir = Path(cfg.paths.results)
    data_path = Path(cfg.paths.data) / "rgat_dataset_external.npz"
    potential_path = Path(cfg.paths.models) / "rgat_model_external.pt"

    dashboard = Dashboard(cfg).start()
    rviz = RvizPublisher.create(cfg) if cfg.viz.realtime else None
    episode_monitor = EpisodeMonitor(cfg, rviz=rviz) if cfg.viz.realtime else None
    reward_monitor = RewardMonitor(cfg)
    try:
        _banner(1, "External stack connectivity")
        STORE.stage("connectivity")
        probe = PX4Bridge(cfg)
        try:
            state = probe.wait_valid_state()
            print(f"PX4 state valid: z={state['position'][2]:.2f} m, "
                  f"armed={state['armed']}, source={state['source']}")
        finally:
            probe.close()

        _banner(2, "Isaac/PX4 R-GAT dataset generation")
        STORE.stage("dataset", f"{cfg.rgat.data_episodes} expert episodes")
        previous_dataset = None
        episode_offset = 0
        if data_path.is_file() and data_path.with_suffix(".graph.pkl").is_file():
            previous_dataset = load_dataset(data_path)
            old_meta = np.asarray(previous_dataset["meta"])
            episode_offset = int(np.max(old_meta[:, 0])) if old_meta.size else 0
            print(f"Continuing cumulative ontology dataset: "
                  f"{len(previous_dataset['y'])} samples from {episode_offset} episodes.")
        new_dataset = generate_dataset(
            cfg, monitor=DatasetMonitor(cfg), episode_monitor=episode_monitor,
            episode_offset=episode_offset,
            checkpoint=lambda batch: save_dataset(
                merge_datasets(previous_dataset, batch), data_path))
        dataset = merge_datasets(previous_dataset, new_dataset)
        save_dataset(dataset, data_path)
        positive = int(np.count_nonzero(np.asarray(dataset["y"]) > 0.0))
        total_samples = int(len(dataset["y"]))
        print(f"Cumulative dataset committed: {total_samples} samples, "
              f"{100.0 * positive / max(total_samples, 1):.1f}% positive.")
        if positive == 0 or positive == total_samples:
            raise RuntimeError(
                "R-GAT needs both successful and failed landing labels; the cumulative "
                "dataset has only one class. The dataset was saved, but model "
                "checkpoints were not overwritten.")

        _banner(3, "R-GAT potential training")
        print("The simulator remains online, but the UAV stays landed during this "
              "offline neural-network training stage.")
        STORE.stage("rgat", f"{cfg.rgat.epochs} epochs")
        rgat_monitor = RGATMonitor(cfg, graph=dataset["graph"])
        previous_potential = None
        previous_rgat_history = None
        if potential_path.is_file():
            previous_potential, previous_rgat_history = load_potential(
                potential_path, cfg, dataset["graph"])
            print(f"Warm-starting R-GAT from {potential_path.name} "
                  f"({len(previous_rgat_history.get('train_loss', []))} prior epochs).")
        def update_rgat(epoch, history, target, prediction, model):
            rgat_monitor.update(epoch, history, target, prediction, model)
            save_potential(model, cfg, potential_path, dict(history))

        potential, rgat_history = train_potential(
            dataset, cfg, initial_model=previous_potential,
            initial_history=previous_rgat_history,
            on_epoch=update_rgat)
        save_potential(potential, cfg, potential_path, dict(rgat_history))
        # The potential drives the RViz attention overlay and the dashboard's
        # 3D graph view from here on.
        if rviz is not None:
            rviz.potential = potential
        if episode_monitor is not None:
            episode_monitor.potential = potential

        agents: dict[str, Any] = {}
        histories: dict[str, Any] = {}
        for step, (mode, label, checkpoint) in enumerate((
                ("manual", LABELS[0], "ppo_manual_external.pt"),
                ("proposed", LABELS[1], "ppo_rgats_pbrs_external.pt")), start=4):
            _banner(step, f"PPO {mode} on PX4")
            STORE.stage(f"ppo_{mode}", f"{cfg.ppo.train_episodes} episodes")
            monitor = PPOMonitor(cfg, mode)
            checkpoint_path = Path(cfg.paths.models) / checkpoint
            previous_agent = None
            previous_history = None
            if checkpoint_path.is_file():
                previous_agent, previous_history = load_agent(checkpoint_path, cfg)
                print(f"Warm-starting {label} PPO from {checkpoint} "
                      f"({len(previous_history.get('episode', []))} prior episodes).")
            agent, history = train_ppo(
                mode, potential if mode == "proposed" else None, cfg,
                initial_agent=previous_agent, initial_history=previous_history,
                on_episode=monitor.update, on_update=monitor.refresh,
                reward_monitor=reward_monitor,
                checkpoint=lambda model, h, path=checkpoint_path: save_agent(
                    model, cfg, path, dict(h)))
            save_agent(agent, cfg, checkpoint_path, dict(history))
            agents[label] = agent
            histories[label] = history

        _banner(6, "Paired external evaluation (urban: moving lorry, energy, GNSS)")
        STORE.stage("evaluation")
        results = compare_policies(agents[LABELS[0]], agents[LABELS[1]], potential, cfg,
                                   episode_monitor=episode_monitor)
        results["wind_sweep"] = wind_sweep(agents[LABELS[0]], agents[LABELS[1]],
                                           potential, cfg,
                                           episode_monitor=episode_monitor)
        results["pad_sweep"] = pad_sweep(agents[LABELS[0]], agents[LABELS[1]],
                                         potential, cfg,
                                         episode_monitor=episode_monitor)
        results["gnss_sweep"] = gnss_sweep(agents[LABELS[0]], agents[LABELS[1]],
                                           potential, cfg,
                                           episode_monitor=episode_monitor)
        results["battery_bins"] = battery_sweep(agents[LABELS[0]], agents[LABELS[1]],
                                                potential, cfg,
                                                episode_monitor=episode_monitor)
        write_table(results["summary"], results_dir / "summary_metrics_external.csv")
        write_table(results["per_episode"], results_dir / "episode_metrics_external.csv")
        with (results_dir / "comparison_results_external.pkl").open("wb") as handle:
            # The potential is saved separately as a checkpoint; keeping a live
            # module in the pickle would tie the results file to a torch version.
            pickle.dump({k: v for k, v in results.items() if k != "potential"}, handle)

        _banner(7, "Publication figures")
        STORE.stage("figures")
        figures = make_plots(results, histories, cfg)

        elapsed = time.perf_counter() - started
        summary = {
            "elapsed_seconds": elapsed,
            "results": str(results_dir),
            "figures": [str(p) for p in figures],
            "rgat_val_loss": rgat_history["val_loss"][-1] if rgat_history["val_loss"] else None,
            "rgat_device": rgat_history.get("device"),
            "cumulative_dataset_samples": total_samples,
            "cumulative_dataset_positive_fraction": positive / max(total_samples, 1),
            "cumulative_rgat_epochs": len(rgat_history.get("train_loss", [])),
            "cumulative_ppo_episodes": {
                label: len(history.get("episode", []))
                for label, history in histories.items()},
        }
        (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        STORE.stage("done", f"{elapsed / 60:.1f} min")
        print(f"\nExternal pipeline complete in {elapsed / 60:.1f} min: {results_dir}")
        return {"potential": potential, "rgat_history": rgat_history,
                "agents": agents, "histories": histories, "results": results,
                "elapsed_seconds": elapsed, "cfg": cfg}
    finally:
        if rviz is not None:
            rviz.close()
        if dashboard is not None:
            dashboard.stop()
