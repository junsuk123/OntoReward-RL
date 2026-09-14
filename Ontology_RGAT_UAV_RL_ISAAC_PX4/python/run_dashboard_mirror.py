#!/usr/bin/env python3
"""Serve the current dashboard UI over an older running telemetry process.

This is a read-only migration aid: a long Isaac/PX4 experiment can keep its
in-process dashboard on 8770 while the newly checked-out UI is served on 8771.
The mirror reconstructs only estimator-free semantic graphs from already
published dashboard fields and evaluates a frozen reward artifact for display;
it has no control, ROS, PX4 or optimizer connection.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.request import urlopen

from ontology_rgat.perception.semantic_observation import (
    SEMANTIC_FEATURE_NAMES, SemanticObservation, semantic_graph)
from ontology_rgat.rgat.adaptive_model import (
    FrozenAdaptiveRewardWeights, adaptive_reward_graph)
from ontology_rgat.viz.dashboard import PAGE
from ontology_rgat.viz.graph3d import graph_payload


class StateEnricher:
    """Add keyed graph and frozen MLP-head traces to a legacy state snapshot."""

    def __init__(self, artifact: Path | None, config_hash: str | None):
        self.artifact = None if artifact is None else Path(artifact).resolve()
        self.config_hash = config_hash
        self._model = None
        self._model_mtime_ns = None

    def _adaptive_model(self):
        if self.artifact is None or not self.artifact.is_file():
            return None
        mtime = self.artifact.stat().st_mtime_ns
        if self._model is None or mtime != self._model_mtime_ns:
            self._model = FrozenAdaptiveRewardWeights(
                self.artifact, expected_config_hash=self.config_hash)
            self._model_mtime_ns = mtime
        return self._model

    @staticmethod
    def _observation(point: dict) -> SemanticObservation | None:
        values = {}
        for name in SEMANTIC_FEATURE_NAMES:
            value = point.get(f"semantic_{name}")
            if value is None:
                return None
            values[name] = float(value)
        return SemanticObservation(
            **values, centroid_xy=(0.0, 0.0),
            raw_scale=max(0.0, values["apparent_target_scale"]))

    def enrich(self, state: dict) -> dict:
        state = dict(state)
        # A process started with the new native publisher is authoritative and
        # already contains exact pair-keyed traces.
        if state.get("graphs"):
            return state
        scalars = dict(state.get("scalars") or {})
        series = dict(state.get("series") or {})
        graphs = {}
        preferred = None
        adaptive = self._adaptive_model()
        for pair in scalars.get("parallel_pair_status") or ():
            index = int(pair.get("index", 0))
            points = series.get(f"benchmark_step_pair_{index}") or ()
            if not points:
                continue
            point = points[-1]
            observation = self._observation(point)
            if observation is None:
                continue
            method = str(pair.get("active_method") or pair.get("method") or "unknown")
            is_adaptive = "adaptive_weight" in method
            graph = (adaptive_reward_graph(observation) if is_adaptive
                     else semantic_graph(observation))
            potential = adaptive if is_adaptive else None
            key = f"pair_{index}:{method}"
            payload = graph_payload(
                graph, potential=potential,
                source=f"read-only mirror · pair {index + 1} · {method} step "
                       f"{int(point.get('step', 0))}",
                phi=point.get("phi"), extra={
                    "graph_id": key, "method": method, "pair_index": index,
                    "mirrored": True,
                })
            graphs[key] = payload
            if payload.get("model") is not None:
                preferred = payload
        state["graphs"] = graphs
        if preferred is not None:
            state["graph"] = preferred
        elif graphs:
            state["graph"] = next(iter(graphs.values()))
        return state


def _handler(source_url: str, enricher: StateEnricher):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - stdlib API
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in ("/", "/index.html"):
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path != "/api/state":
                self.send_error(404)
                return
            try:
                with urlopen(source_url, timeout=3.0) as response:
                    state = json.load(response)
                body = json.dumps(
                    enricher.enrich(state), allow_nan=False).encode("utf-8")
                self._send(body, "application/json")
            except Exception as exc:  # dashboard failure must remain isolated
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(body, "application/json", status=502)

        def log_message(self, *args):
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default="http://127.0.0.1:8770/api/state")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    config_hash = None
    if args.manifest and args.manifest.is_file():
        config_hash = json.loads(args.manifest.read_text(encoding="utf-8")).get(
            "config_hash")
    server = ThreadingHTTPServer(
        (args.host, args.port),
        _handler(args.source_url, StateEnricher(args.artifact, config_hash)))
    print(f"Read-only enhanced dashboard: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
