"""Live training telemetry shared by the browser, RViz, and file exports.

The retired MATLAB monitors each owned a figure window, which is why an
unattended run was invisible from outside MATLAB and a headless one was
invisible entirely. Here the *data* is the shared thing:

* :class:`LiveStore` holds it, thread-safe, bounded.
* :mod:`ontology_rgat.viz.dashboard` serves it over HTTP for a browser.
* the monitors below export a PNG and a CSV to ``results/live`` every few
  episodes or epochs, which is what makes progress inspectable over SSH with
  nothing running locally.

The PNG export uses the Agg backend on purpose: a training run must never
depend on a display being attached, and must never try to open a window on a
machine that has none.
"""
from __future__ import annotations

import csv
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["LiveStore", "DatasetMonitor", "RGATMonitor", "PPOMonitor",
           "RewardMonitor", "BenchmarkMonitor", "EpisodeMonitor", "STORE",
           "MATLAB_COLORS"]


# MATLAB R2025a's default line order.  The browser dashboard uses the same
# sequence, so exported PNG snapshots and the live view remain visually
# comparable when figures are copied into a report.
MATLAB_COLORS = (
    "#0072BD", "#D95319", "#EDB120", "#7E2F8E",
    "#77AC30", "#4DBEEE", "#A2142F",
)


class LiveStore:
    """Bounded, thread-safe series store shared by every live consumer."""

    def __init__(self, history: int = 4000):
        self._lock = threading.Lock()
        self._history = int(history)
        self._series: dict[str, deque] = {}
        self._scalars: dict[str, Any] = {}
        # The ontology snapshot is a whole object rather than a time series:
        # only the latest one is ever drawn, and keeping a history of them
        # would be several megabytes of graph nobody looks at.
        self._graph: dict[str, Any] | None = None
        # Parallel workers must not overwrite one another's graph. Keep one
        # latest snapshot per physical-pair/method while retaining ``graph``
        # above for clients written against the former single-graph API.
        self._graphs: dict[str, dict[str, Any]] = {}
        self._stage = {"name": "idle", "detail": "", "started": time.time()}
        self.revision = 0

    # ------------------------------------------------------------- writing
    def append(self, series: str, point: dict[str, Any]) -> None:
        with self._lock:
            bucket = self._series.get(series)
            if bucket is None:
                bucket = self._series[series] = deque(maxlen=self._history)
            bucket.append(point)
            self.revision += 1

    def replace(self, series: str, points: Iterable[dict[str, Any]]) -> None:
        with self._lock:
            self._series[series] = deque(points, maxlen=self._history)
            self.revision += 1

    def set(self, **scalars: Any) -> None:
        with self._lock:
            self._scalars.update(scalars)
            self.revision += 1

    def graph(self, payload: dict[str, Any] | None, *, key: str | None = None) -> None:
        """Replace one keyed graph snapshot and the legacy latest snapshot."""
        with self._lock:
            self._graph = payload
            if key is not None:
                graph_key = str(key)
                if payload is None:
                    self._graphs.pop(graph_key, None)
                else:
                    self._graphs[graph_key] = payload
            self.revision += 1

    def stage(self, name: str, detail: str = "") -> None:
        with self._lock:
            self._stage = {"name": name, "detail": detail, "started": time.time()}
            self.revision += 1

    # ------------------------------------------------------------- reading
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "revision": self.revision,
                "stage": dict(self._stage),
                "scalars": dict(self._scalars),
                "series": {k: list(v) for k, v in self._series.items()},
                "graph": self._graph,
                "graphs": dict(self._graphs),
                "time": time.time(),
            }

    def series(self, name: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._series.get(name, ()))


STORE = LiveStore()


# ------------------------------------------------------------------ helpers
def _figure(nrows: int, ncols: int, size: tuple[float, float]):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True)
    fig.patch.set_facecolor("white")
    flat = np.atleast_1d(np.asarray(axes)).ravel()
    for axis in flat:
        axis.set_facecolor("white")
        axis.set_prop_cycle(color=MATLAB_COLORS)
        axis.grid(True, color="#D8D8D8", linestyle=":", linewidth=0.7)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#262626")
            spine.set_linewidth(0.8)
        axis.tick_params(colors="#262626", labelsize=8, direction="out")
        axis.title.set_color("#262626")
        axis.xaxis.label.set_color("#262626")
        axis.yaxis.label.set_color("#262626")
    return fig, flat


def _moving_mean(values: Sequence[float], window: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    w = max(1, min(int(window), v.size))
    kernel = np.ones(w) / w
    padded = np.concatenate([np.full(w - 1, v[0]), v])
    return np.convolve(padded, kernel, mode="valid")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class _Monitor:
    """Shared plumbing: throttled PNG/CSV export plus the live store."""

    series_name = "series"
    figure_name = "monitor"

    def __init__(self, cfg, store: LiveStore | None = None):
        self.cfg = cfg
        self.store = store or STORE
        self.live_dir = Path(cfg.paths.live)
        self.every = max(1, int(cfg.viz.live_every))
        self.export = bool(cfg.viz.live_export)

    def _maybe_export(self, index: int, force: bool = False) -> None:
        if not self.export or (index % self.every and not force):
            return
        rows = self.store.series(self.series_name)
        write_csv(self.live_dir / f"{self.figure_name}.csv", rows)
        try:
            self.render(rows, self.live_dir / f"{self.figure_name}.png")
        except Exception as exc:                       # pragma: no cover - plotting
            print(f"WARNING: live snapshot failed for {self.figure_name}: {exc}")

    def render(self, rows: Sequence[dict[str, Any]], path: Path) -> None:
        raise NotImplementedError


class DatasetMonitor(_Monitor):
    """Expert rollout collection.

    The success rate is the panel that matters: a flat zero here means the
    potential's target has no positive examples to learn from, which no amount
    of R-GAT training afterwards can fix.
    """

    series_name = "dataset"
    figure_name = "rgat_dataset"

    def update(self, episode: int, success: float, noise: float, samples: int) -> None:
        self.store.append(self.series_name, {
            "episode": episode, "success": float(success),
            "noise_std": float(noise), "samples": int(samples)})
        self.store.set(dataset_episode=episode)
        self._maybe_export(episode)

    def finish(self) -> None:
        self._maybe_export(0, force=True)

    def render(self, rows, path):
        import matplotlib.pyplot as plt
        if not rows:
            return
        episodes = [r["episode"] for r in rows]
        success = [r["success"] for r in rows]
        fig, ax = _figure(1, 3, (12.0, 3.6))
        ax[0].plot(episodes, _moving_mean(success, max(1, len(rows) // 4)),
                   color=MATLAB_COLORS[4], lw=1.8)
        ax[0].set(xlabel="Episode", ylabel="Success rate", ylim=(0, 1),
                  title="Expert moving success rate")
        ax[1].plot(episodes, np.cumsum(success), color=MATLAB_COLORS[0], lw=1.8,
                   label="successful episodes")
        ax[1].plot(episodes, [r["samples"] for r in rows], color=MATLAB_COLORS[1], lw=1.2,
                   label="samples/episode")
        ax[1].set(xlabel="Episode", ylabel="Count", title="Collection progress")
        ax[1].legend(frameon=False, fontsize=8)
        ax[2].scatter([r["noise_std"] for r in rows], success, s=22, alpha=0.6)
        ax[2].set(xlabel="action noise sigma", ylabel="success", ylim=(-0.1, 1.1),
                  title="Outcome vs perturbation")
        for a in ax:
            a.grid(alpha=0.3)
        fig.savefig(path, dpi=140)
        plt.close(fig)


class RGATMonitor(_Monitor):
    """Potential regression loss and the validation fit it implies."""

    series_name = "rgat"
    figure_name = "rgat_training"

    def __init__(self, cfg, store: LiveStore | None = None, graph=None):
        super().__init__(cfg, store)
        self._scatter: tuple[np.ndarray, np.ndarray] = (np.zeros(0), np.zeros(0))
        # The template captured from the first rollout, so the 3D view can show
        # what the potential is attending to on a graph it actually saw.
        self.graph = graph
        from .graph3d import GraphPublisher   # local: graph3d imports this module
        self.graph_pub = GraphPublisher(cfg, self.store)

    def update(self, epoch: int, history, y_true: np.ndarray, y_pred: np.ndarray,
               model=None) -> None:
        self.store.append(self.series_name, {
            "epoch": epoch,
            "train_mse": float(history["train_loss"][-1]),
            "val_mse": float(history["val_loss"][-1]),
            "epoch_seconds": float(history["epoch_seconds"][-1])})
        self.store.set(rgat_epoch=epoch, rgat_device=history.get("device", "?"),
                       rgat_val_scatter=[
                           [float(a), float(b)] for a, b in
                           zip(y_true[:600].tolist(), y_pred[:600].tolist())])
        self._scatter = (np.asarray(y_true), np.asarray(y_pred))
        last = epoch == int(self.cfg.rgat.epochs)
        if model is not None and self.graph is not None:
            # Every ``live_every`` epochs and always on the last one: the
            # attention early in training is nearly uniform, and watching it
            # concentrate is the point of showing it at all.
            self.graph_pub.potential = model
            self.graph_pub.publish(
                self.graph, source=f"R-GAT epoch {epoch}",
                force=last or not (epoch % self.every))
        self._maybe_export(epoch, force=last)

    def render(self, rows, path):
        import matplotlib.pyplot as plt
        if not rows:
            return
        fig, ax = _figure(1, 2, (9.6, 4.0))
        epochs = [r["epoch"] for r in rows]
        ax[0].plot(epochs, [r["train_mse"] for r in rows], lw=1.6, label="train")
        ax[0].plot(epochs, [r["val_mse"] for r in rows], lw=1.6, label="validation")
        ax[0].set(xlabel="Epoch", ylabel="MSE", title="Potential regression loss")
        ax[0].legend(frameon=False)
        y_true, y_pred = self._scatter
        if y_true.size:
            ax[1].scatter(y_true, y_pred, s=10, alpha=0.35)
        ax[1].plot([-1, 1], [-1, 1], "k--", lw=1)
        ax[1].set(xlabel="target potential", ylabel="predicted potential",
                  xlim=(-1.05, 1.05), title="Validation fit")
        for a in ax:
            a.grid(alpha=0.3)
        fig.savefig(path, dpi=140)
        plt.close(fig)


class PPOMonitor(_Monitor):
    """Return, outcome mix, losses and exploration for one PPO arm.

    The terminal-outcome panel is the one that matters most: a policy that
    learns to hover out the clock shows up as a rising ``timeout`` fraction with
    a flat success rate, which a return curve alone hides.
    """

    STATUSES = ("success", "unsafe_touchdown", "flight_failure", "timeout",
                "battery_depleted", "ground_mislanding")

    def __init__(self, cfg, name: str, store: LiveStore | None = None):
        super().__init__(cfg, store)
        self.name = name
        self.series_name = f"ppo_{name}"
        self.figure_name = f"ppo_{name}"

    def update(self, history) -> None:
        i = len(history["episode"]) - 1
        self.store.append(self.series_name, {
            "episode": history["episode"][i], "iteration": history["iteration"][i],
            "return": history["ret"][i], "success": history["success"][i],
            "steps": history["length"][i], "status": history["status"][i],
            "actor_loss": history["actor_loss"][i],
            "critic_loss": history["critic_loss"][i],
            "policy_std": history["policy_std"][i]})
        self.store.set(**{f"ppo_{self.name}_episode": history["episode"][i]})
        self._maybe_export(int(history["episode"][i]))

    def refresh(self, history) -> None:
        """Rewrite the whole series after an update fills in the losses."""
        self.store.replace(self.series_name, [{
            "episode": history["episode"][i], "iteration": history["iteration"][i],
            "return": history["ret"][i], "success": history["success"][i],
            "steps": history["length"][i], "status": history["status"][i],
            "actor_loss": history["actor_loss"][i],
            "critic_loss": history["critic_loss"][i],
            "policy_std": history["policy_std"][i]}
            for i in range(len(history["episode"]))])
        self._maybe_export(0, force=True)

    def render(self, rows, path):
        import matplotlib.pyplot as plt
        if not rows:
            return
        fig, ax = _figure(2, 3, (13.0, 7.0))
        episodes = [r["episode"] for r in rows]
        window = max(1, min(50, len(rows) // 5))
        returns = [r["return"] for r in rows]
        ax[0].plot(episodes, returns, color="#b8c9e6", lw=0.8)
        ax[0].plot(episodes, _moving_mean(returns, window), color=MATLAB_COLORS[0], lw=1.8)
        ax[0].set(xlabel="Episode", ylabel="Return", title="Episode return")
        ax[1].plot(episodes, _moving_mean([r["success"] for r in rows], window),
                   color=MATLAB_COLORS[4], lw=1.8)
        ax[1].set(xlabel="Episode", ylabel="Success rate", ylim=(0, 1),
                  title="Moving success rate")
        steps = [r["steps"] for r in rows]
        ax[2].plot(episodes, steps, color="#f2d6b8", lw=0.8)
        ax[2].plot(episodes, _moving_mean(steps, window), color=MATLAB_COLORS[1], lw=1.8)
        ax[2].set(xlabel="Episode", ylabel="Steps", title="Episode length")
        ax[3].plot(episodes, [r["actor_loss"] for r in rows], lw=1.4, label="actor")
        twin = ax[3].twinx()
        twin.plot(episodes, [r["critic_loss"] for r in rows], lw=1.4,
                  color=MATLAB_COLORS[1], label="critic")
        ax[3].set(xlabel="Episode", ylabel="Actor loss", title="PPO losses")
        twin.set_ylabel("Critic loss")
        ax[4].plot(episodes, [r["policy_std"] for r in rows],
                   color=MATLAB_COLORS[3], lw=1.8)
        ax[4].set(xlabel="Episode", ylabel="exp(logStd)", title="Exploration std")
        statuses = [r["status"] for r in rows]
        for status in self.STATUSES:
            fraction = _moving_mean([float(s == status) for s in statuses], window)
            ax[5].plot(episodes, fraction, lw=1.4, label=status.replace("_", " "))
        ax[5].set(xlabel="Episode", ylabel="Fraction", ylim=(0, 1),
                  title="Terminal outcome mix")
        ax[5].legend(frameon=False, fontsize=7)
        for a in ax:
            a.grid(alpha=0.3)
        fig.suptitle(f"PPO - {self.name}")
        fig.savefig(path, dpi=140)
        plt.close(fig)


class RewardMonitor:
    """Live decomposition of the R-GAT potential-shaped PPO reward."""

    def __init__(self, cfg, store: LiveStore | None = None):
        self.cfg = cfg
        self.store = store or STORE

    def reset(self, mode: str, episode: int) -> None:
        self.store.replace("reward", [])
        self.store.set(reward_mode=str(mode), reward_episode=int(episode))

    def update(self, *, step: int, t: float, mode: str, reward: float,
               parts: dict[str, Any], status: str) -> None:
        phi = parts.get("phi0")
        phi_next = parts.get("phi1")
        self.store.append("reward", {
            "step": int(step), "t": float(t), "mode": str(mode),
            "status": str(status), "reward": float(reward),
            "base": float(parts.get("base", reward)),
            "shape": float(parts.get("shape", 0.0)),
            "phi": float(phi) if phi is not None else None,
            "phi_next": float(phi_next) if phi_next is not None else None,
            "design_id": parts.get("design_id"),
            "weighted_terms": dict(parts.get("weighted_terms") or {}),
            "shaped": str(mode).lower() == "proposed",
        })


class BenchmarkMonitor:
    """Publish recurrent Shin/OntoReward training and paired evaluation.

    This monitor deliberately accepts already-computed metrics.  It cannot
    reach into the actor observation, so adding the dashboard does not create
    a back door for simulator truth to enter the deployed policy.
    """

    def __init__(self, store: LiveStore | None = None, rviz=None):
        self.store = store or STORE
        self.rviz = rviz
        self._potential = None
        self._potentials: dict[str, Any] = {}
        self.methods: tuple[str, ...] = ()
        self.training_total = 0
        self.evaluation_total = 0
        self.pair_layout: list[dict[str, Any]] = []
        self.pair_status: dict[int, dict[str, Any]] = {}

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, value) -> None:
        self._potential = value
        if self.rviz is not None:
            self.rviz.potential = value

    def set_potential(self, method: str, value) -> None:
        """Bind a reward-side model to exactly one experiment arm."""
        self._potentials[str(method)] = value
        if len(self.methods) <= 1:
            self.potential = value

    def potential_for(self, method: str):
        selected = self._potentials.get(str(method))
        if selected is not None:
            return selected
        return self._potential if len(self.methods) <= 1 else None

    def _resolve_pair_index(self, method: str, pair_index: int | None) -> int:
        if pair_index is not None:
            return int(pair_index)
        matching = [index for index, status in self.pair_status.items()
                    if str(status.get("assigned_method",
                                      status.get("method"))) == str(method)]
        if not matching:
            matching = [index for index, status in self.pair_status.items()
                        if str(status.get("active_method")) == str(method)]
        return int(matching[0] if matching else 0)

    def _update_pair(self, method: str, *, pair_index: int | None = None,
                     **values: Any) -> None:
        """Atomically publish one pair without overwriting another worker."""
        key = self._resolve_pair_index(method, pair_index)
        current = dict(self.pair_status.get(key) or {"index": key})
        # ``method`` in the layout is the experiment arm permanently assigned
        # to this physical pair. Reward-design collection can temporarily run
        # another behavior policy on that pair, so expose it separately rather
        # than making the dashboard look as if the assignment changed.
        assigned = str(current.get(
            "assigned_method", current.get("method", method)))
        current["method"] = assigned
        current["assigned_method"] = assigned
        current["active_method"] = str(method)
        current.update(values)
        current["index"] = key
        self.pair_status[key] = current
        ordered = sorted(self.pair_status.values(), key=lambda item: int(item["index"]))
        self.store.set(parallel_pair_status=ordered)

    @staticmethod
    def _plain(row: dict[str, Any]) -> dict[str, Any]:
        """Convert CSV/numpy values into chart-safe numbers where possible."""
        out: dict[str, Any] = {}
        integer_keys = {"seed", "episode", "steps", "curriculum_level",
                        "training_sample_efficiency", "evaluation_index"}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                out[key] = int(value)
            elif isinstance(value, (np.floating,)):
                out[key] = float(value)
            elif isinstance(value, str):
                try:
                    out[key] = int(value) if key in integer_keys else float(value)
                except ValueError:
                    out[key] = value
            else:
                out[key] = value
        return out

    def configure(self, *, methods: Sequence[str], mode: str, config_hash: str,
                  training_total: int, evaluation_total: int,
                  reward_design_id: str | None = None,
                  reward_design_sha256: str | None = None,
                  pair_layout: Sequence[dict[str, Any]] | None = None) -> None:
        self.methods = tuple(str(method) for method in methods)
        self.training_total = int(training_total)
        self.evaluation_total = int(evaluation_total)
        self.pair_layout = [dict(item) for item in (pair_layout or ())]
        self.pair_status = {
            int(item["index"]): {
                **dict(item),
                "assigned_method": str(item.get("method", "")),
                "active_method": None,
                "phase": "waiting", "episode": 0,
                "step": 0, "status": "waiting",
            }
            for item in self.pair_layout
        }
        self.store.set(
            dashboard_profile="parallel_two_pair", benchmark_mode=str(mode),
            benchmark_methods=list(self.methods), config_hash=str(config_hash),
            training_total=self.training_total,
            evaluation_total=self.evaluation_total,
            reward_design_id=reward_design_id,
            reward_design_sha256=reward_design_sha256,
            parallel_pair_count=max(1, len(self.pair_layout)),
            parallel_pair_layout=self.pair_layout,
            parallel_pair_status=sorted(
                self.pair_status.values(), key=lambda item: int(item["index"])),
            actor_contract={
                "camera": "512x320 mono / 90 deg HFOV / -60 deg pitch",
                "proprioception": "7: body velocity (3) + quaternion (4)",
                "temporal": ("6-keypoint + explicit visibility CNN -> generic "
                             "512 LSTM -> latent y[256]"),
                "actor": "y[6:256] + proprioception -> action[4]",
                "critic": "training only: proprioception[7] + truth[6]",
                "reward_side": ("both: unchanged Shin Table III + active perception; "
                                "proposed only: -lambda_fov * P(future FOV loss)"),
                "state_estimation": ("both: unbounded y[0:6] decoded to physical "
                                     "units; identical scale-normalized auxiliary loss"),
                "forbidden": ("deployed actor: platform/GNSS/truth; onto graph: "
                              "relative estimate, simulator truth, critic truth, "
                              "platform position/velocity"),
            })

    def stage(self, name: str, detail: str = "") -> None:
        self.store.stage(name, detail)

    def reset_episode(self, *, method: str, phase: str, seed: int,
                      scenario: str, curriculum: float,
                      action_scale: float = 1.0,
                      motion_scale: float | None = None,
                      pair_index: int | None = None) -> None:
        if len(self.methods) <= 1:
            self.store.replace("benchmark_step", [])
        self.store.replace(f"benchmark_step_{method}", [])
        resolved_pair_index = self._resolve_pair_index(method, pair_index)
        self.store.replace(f"benchmark_step_pair_{int(resolved_pair_index)}", [])
        # Completed histories only advance after optimizer/checkpoint commit.
        # Publish the in-flight episode separately so a 30 s simulated flight
        # cannot look frozen for one or two minutes on a rendered lockstep run.
        is_training = str(phase) == "training"
        current_episode = len(self.store.series(
            f"benchmark_{'train' if is_training else 'eval'}_{method}")) + 1
        if self.rviz is not None:
            self.rviz.clear_trails(method=method, pair_index=pair_index)
        try:
            from ..pipelines import get_pipeline
            pipeline = get_pipeline(method)
            pipeline_name = pipeline.name
            estimation_status = ("ENABLED" if pipeline.state_estimation_enabled
                                 else "DISABLED")
        except ValueError:
            pipeline_name = str(method)
            estimation_status = "ENABLED" if method == "shin2026" else "LEGACY"
        self.store.set(
            benchmark_phase=str(phase), current_method=str(method),
            current_pipeline=pipeline_name,
            state_estimation_status=estimation_status,
            current_seed=int(seed), current_scenario=str(scenario),
            current_training_episode=(int(current_episode) if is_training else None),
            current_evaluation_episode=(None if is_training else int(current_episode)),
            current_curriculum=float(curriculum),
            current_pad_motion_scale=float(
                curriculum if motion_scale is None else motion_scale),
            current_action_envelope_scale=float(action_scale))
        self._update_pair(
            method, phase=str(phase), seed=int(seed), scenario=str(scenario),
            episode=int(current_episode), episode_kind=("학습" if is_training else "평가"),
            step=0, status="running",
            pipeline=pipeline_name, state_estimation=estimation_status,
            curriculum=float(curriculum), motion_scale=float(
                curriculum if motion_scale is None else motion_scale),
            action_scale=float(action_scale), marker_visible=None,
            success=None, landing_gate=None, pair_index=resolved_pair_index)

    def step(self, *, index: int, dt: float, method: str, reward: float,
             reward_parts: dict[str, Any], estimate, truth, in_fov: bool,
             estimation_loss: float | None,
             state: dict[str, Any] | None = None, pipeline_spec=None,
             semantic_features=None, semantic_graph=None,
             scenario: str = "", status: str = "running",
             pair_index: int | None = None) -> None:
        parts = reward_parts or {}
        point = {
            "step": int(index), "t": float(index * dt), "method": str(method),
            "status": str(status),
            "reward": float(reward), "in_fov": float(bool(in_fov)),
            "pipeline": str(getattr(pipeline_spec, "name", method)),
            "state_estimation_enabled": bool(
                getattr(pipeline_spec, "state_estimation_enabled", estimate is not None)),
        }
        if estimate is not None and estimation_loss is not None:
            estimate = np.asarray(estimate, dtype=float)
            truth = np.asarray(truth, dtype=float)
            point.update({
                "estimation_loss": float(estimation_loss),
                "position_error": float(np.linalg.norm(estimate[:3] - truth[:3])),
                "velocity_error": float(np.linalg.norm(estimate[3:] - truth[3:])),
                "estimated_distance": float(np.linalg.norm(estimate[:3])),
                "estimated_speed": float(np.linalg.norm(estimate[3:])),
            })
        if semantic_features is not None:
            from ..perception import SEMANTIC_FEATURE_NAMES
            values = np.asarray(semantic_features, dtype=float).reshape(-1)
            if values.shape == (len(SEMANTIC_FEATURE_NAMES),):
                point.update({f"semantic_{name}": float(value)
                              for name, value in zip(SEMANTIC_FEATURE_NAMES, values)})
        resolved_pair_index = self._resolve_pair_index(method, pair_index)
        # The browser polls at 1.5 s and the graph is an audit view, not a
        # control input. Re-running the encoder trace every fifth 10 Hz step is
        # sufficient while keeping the telemetry cost away from PPO timing.
        if semantic_graph is not None and (int(index) == 1 or int(index) % 5 == 0):
            from .graph3d import graph_payload
            graph_key = f"pair_{resolved_pair_index}:{method}"
            self.store.graph(graph_payload(
                semantic_graph, potential=self.potential_for(method),
                source=f"pair {resolved_pair_index + 1} · {method} step {index}",
                phi=parts.get("phi"), extra={
                    "graph_id": graph_key,
                    "method": str(method),
                    "pair_index": int(resolved_pair_index),
                }), key=graph_key)
        for key in ("task", "lateral_progress", "vertical_progress",
                    "vertical_speed_penalty", "undershoot_penalty",
                    "yaw_rate_penalty", "active_perception", "shape",
                    "phi", "phi_next", "ontology_fov_reward",
                    "predicted_fov_loss_probability", "fov_margin",
                    "keypoint_confidence", "visible_keypoint_fraction",
                    "visibility_memory", "reacquisition_trend"):
            point[key] = float(parts.get(key, 0.0))
        battery = (state.get("battery") if isinstance(state, dict)
                   and isinstance(state.get("battery"), dict) else {})
        world = (state.get("world") if isinstance(state, dict)
                 and isinstance(state.get("world"), dict) else {})
        pad = (state.get("pad") if isinstance(state, dict)
               and isinstance(state.get("pad"), dict) else {})
        uav_velocity = np.asarray(world.get("velocity", ()), dtype=float)
        pad_velocity = np.asarray(pad.get("velocity", ()), dtype=float)
        enabled = bool(battery.get("enabled", False))
        point.update({
            "uav_speed_m_s": float(np.linalg.norm(uav_velocity))
            if uav_velocity.shape == (3,) else 0.0,
            "ugv_speed_m_s": float(np.linalg.norm(pad_velocity))
            if pad_velocity.shape == (3,) else 0.0,
            "battery_reserve": float(battery.get("reserve", 1.0) if enabled else 1.0),
            "battery_state_of_charge": float(
                battery.get("state_of_charge", 1.0) if enabled else 1.0),
            "battery_remaining_j": float(battery.get("remaining_j", 0.0)),
            "battery_energy_used_j": float(battery.get("energy_used_j", 0.0)),
            "battery_power_kw": float(battery.get("power_w", 0.0)) / 1000.0,
        })
        if len(self.methods) <= 1:
            self.store.append("benchmark_step", point)
        self.store.append(f"benchmark_step_{method}", point)
        self.store.append(
            f"benchmark_step_pair_{int(resolved_pair_index)}", point)
        relative = np.asarray(
            ((state.get("truth") or {}).get("position")
             if isinstance(state, dict) and isinstance(state.get("truth"), dict)
             and (state.get("truth") or {}).get("valid", False)
             else state.get("position", ())) if isinstance(state, dict) else (),
            dtype=float)
        self._update_pair(
            method, step=int(index), status=str(status),
            marker_visible=bool(in_fov),
            uav_speed_m_s=point["uav_speed_m_s"],
            ugv_speed_m_s=point["ugv_speed_m_s"],
            battery_reserve=point["battery_reserve"],
            relative_xyz=(relative.tolist() if relative.shape == (3,) else None),
            reward=float(reward),
            active_perception=point["active_perception"],
            ontology_fov_reward=point["ontology_fov_reward"],
            predicted_fov_loss_probability=point[
                "predicted_fov_loss_probability"],
            fov_margin=point["fov_margin"],
            keypoint_confidence=point["keypoint_confidence"],
            visible_keypoint_fraction=point["visible_keypoint_fraction"],
            pair_index=resolved_pair_index)
        if self.rviz is not None and state is not None:
            self.rviz.publish_benchmark_step(
                state=state, method=method, scenario=scenario, step=index,
                dt=dt, in_fov=in_fov, status=status,
                reward=float(reward), reward_parts=parts,
                semantic_graph=semantic_graph,
                potential=self.potential_for(method),
                pair_index=resolved_pair_index)

    def restore_training(self, method: str, history: Sequence[dict[str, Any]]) -> None:
        rows = [self._plain(dict(row)) for row in history]
        self.store.replace(f"benchmark_train_{method}", rows)
        if rows:
            self.store.set(**{f"benchmark_{method}_episode": rows[-1]["episode"]})

    def training_update(self, method: str, metric: dict[str, Any], *,
                        pair_index: int | None = None) -> None:
        point = self._plain(dict(metric))
        self.store.append(f"benchmark_train_{method}", point)
        self.store.set(**{
            f"benchmark_{method}_episode": point["episode"],
            "benchmark_phase": "training", "current_method": str(method),
            "current_pipeline": str(point.get("pipeline", method)),
            "state_estimation_status": (
                "ENABLED" if point.get("state_estimation_enabled", False)
                else "DISABLED"),
            "last_landing_gate": {
                "status": point.get("status"),
                "safe_landing": point.get("paper_success"),
                "pad_contact": point.get("pad_contact"),
                "position": point.get("landing_gate_position"),
                "vertical_speed": point.get("landing_gate_vertical_speed"),
                "relative_horizontal_speed": point.get(
                    "landing_gate_relative_horizontal_speed"),
                "attitude": point.get("landing_gate_attitude"),
                "angular_rate": point.get("landing_gate_angular_rate"),
            },
        })
        self._update_pair(
            method, phase="training", status=str(point.get("status", "complete")),
            completed_episode=int(point["episode"]),
            success=float(point.get("paper_success", 0.0)),
            landing_gate={
                "contact": point.get("pad_contact"),
                "position": point.get("landing_gate_position"),
                "vertical_speed": point.get("landing_gate_vertical_speed"),
                "relative_horizontal_speed": point.get(
                    "landing_gate_relative_horizontal_speed"),
                "attitude": point.get("landing_gate_attitude"),
                "angular_rate": point.get("landing_gate_angular_rate"),
            }, pair_index=pair_index)

    def restore_evaluation(self, rows: Sequence[dict[str, Any]]) -> None:
        completed = 0
        for method in self.methods:
            selected = []
            for row in rows:
                if row.get("method") == method:
                    point = self._plain(dict(row))
                    point["evaluation_index"] = len(selected) + 1
                    selected.append(point)
            self.store.replace(f"benchmark_eval_{method}", selected)
            completed += len(selected)
            if selected and self.pair_status:
                physical_pair = selected[-1].get("physical_pair_index")
                self._update_pair(
                    method, phase="evaluation", episode=len(selected),
                    episode_kind="평가", completed_episode=len(selected),
                    step=0, status="complete",
                    scenario=str(selected[-1].get("scenario", "")),
                    success=float(selected[-1].get("paper_success", 0.0)),
                    pair_index=(None if physical_pair in (None, "") else
                                int(float(physical_pair))))
        self.store.set(evaluation_completed=completed)

    def evaluation_update(self, method: str, metric: dict[str, Any], *,
                          pair_index: int | None = None) -> None:
        point = self._plain(dict(metric))
        point["evaluation_index"] = len(
            self.store.series(f"benchmark_eval_{method}")) + 1
        completed = sum(len(self.store.series(f"benchmark_eval_{name}"))
                        for name in self.methods) + 1
        self.store.append(f"benchmark_eval_{method}", point)
        self.store.set(
            benchmark_phase="evaluation", current_method=str(method),
            current_scenario=str(point.get("scenario", "")),
            evaluation_completed=completed,
            last_landing_gate={
                "status": point.get("status"),
                "safe_landing": point.get("paper_success"),
                "pad_contact": point.get("pad_contact"),
                "position": point.get("landing_gate_position"),
                "vertical_speed": point.get("landing_gate_vertical_speed"),
                "relative_horizontal_speed": point.get(
                    "landing_gate_relative_horizontal_speed"),
                "attitude": point.get("landing_gate_attitude"),
                "angular_rate": point.get("landing_gate_angular_rate"),
            })
        self._update_pair(
            method, phase="evaluation", status=str(point.get("status", "complete")),
            scenario=str(point.get("scenario", "")),
            success=float(point.get("paper_success", 0.0)),
            landing_gate={
                "contact": point.get("pad_contact"),
                "position": point.get("landing_gate_position"),
                "vertical_speed": point.get("landing_gate_vertical_speed"),
                "relative_horizontal_speed": point.get(
                    "landing_gate_relative_horizontal_speed"),
                "attitude": point.get("landing_gate_attitude"),
                "angular_rate": point.get("landing_gate_angular_rate"),
            }, pair_index=pair_index)


class EpisodeMonitor:
    """Per-step episode telemetry, fanned out to RViz 2 and the dashboard.

    This replaces the retired ``viz.RealtimeMonitor`` figure. The 3D view it
    drew is now RViz 2's job -- it is a better 3D viewer than a MATLAB axes and
    it already has the deck, the markers and the TF tree from the simulator --
    so what stays here is the plumbing that fans one control step out to
    whoever is watching.
    """

    def __init__(self, cfg, rviz=None, store: LiveStore | None = None,
                 label: str = "episode", potential=None):
        self.cfg = cfg
        self.rviz = rviz
        self.store = store or STORE
        self.label = label
        from .graph3d import GraphPublisher   # local: graph3d imports this module
        self.graph_pub = GraphPublisher(cfg, self.store, potential=potential)

    @property
    def potential(self):
        """The model whose attention the 3D graph view draws, if any."""
        return self.graph_pub.potential

    @potential.setter
    def potential(self, model) -> None:
        self.graph_pub.potential = model

    def __call__(self, log, k: int, cur, info: dict[str, Any]) -> None:
        self.store.append("episode", {
            "step": k, "t": float(log.t[-1]), "status": info["status"],
            "z": float(log.x[-1][2]),
            "xy_error": float(np.linalg.norm(log.x[-1][0:2])),
            "reward": float(log.r[-1]), "tilt_deg": float(np.degrees(log.tilt[-1])),
            "reward_base": float(log.reward_base[-1]),
            "reward_shape": float(log.reward_shape[-1]),
            "base": float(log.reward_base[-1]),
            "shape": float(log.reward_shape[-1]),
            "mode": str(info.get("reward_mode", "episode")),
            "shaped": str(info.get("reward_mode", "")).lower() == "proposed",
            "phi_next": (float(log.phi_next[-1])
                         if np.isfinite(log.phi_next[-1]) else None),
            "aero_n": float(log.aero_mag[-1]),
            "wind_speed": float(np.linalg.norm(log.wind[-1])),
            "wind_risk": float(cur.sem.wind_risk),
            "pad_speed": float(log.pad_speed[-1]),
            "closing_speed": float(log.closing_speed[-1]),
            "marker_quality": float(log.marker_quality[-1]),
            "gnss_quality": float(log.gnss_quality[-1]),
            "gnss_sigma_xy": float(log.gnss_sigma_xy[-1]),
            "nav_confidence": float(log.nav_confidence[-1]),
            # What the canyon is costing the pose being flown on, right now.
            "estimate_error_m": float(log.estimate_error_m[-1]),
            "hover_seconds_left": float(log.hover_seconds_left[-1]),
            "battery_power_w": float(log.battery_power_w[-1]),
            "phi": float(log.phi[-1]) if np.isfinite(log.phi[-1]) else None})
        self.store.set(episode_label=self.label, episode_status=info["status"])
        phi = float(log.phi[-1]) if np.isfinite(log.phi[-1]) else None
        self.graph_pub.publish(cur.graph, cur.sem.node_values, phi=phi,
                               source=f"{self.label} t={log.t[-1]:.1f}s")
        if self.rviz is not None:
            self.rviz.publish_step(log, cur, info)

    def reset(self, label: str | None = None) -> None:
        if label:
            self.label = label
        self.store.replace("episode", [])
        self.graph_pub.clear()
        if self.rviz is not None:
            self.rviz.clear_trails()
