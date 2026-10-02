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
import ast
from dataclasses import dataclass
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

class LivePage:
    """Reload only the dashboard HTML constant when its source file changes.

    The experiment process on port 8770 owns Isaac/PX4/PPO state and is never
    touched.  This read-only mirror parses ``PAGE`` directly from the source on
    each browser refresh, so UI-only edits become visible without restarting
    either the experiment or this mirror.  A syntax error keeps serving the
    last known-good page instead of taking telemetry away from the operator.
    """

    def __init__(self, path: Path | None = None):
        default_path = (Path(__file__).resolve().parent / "ontology_rgat" /
                        "viz" / "dashboard.py")
        self.path = Path(path or default_path).resolve()
        self._page = ("<!doctype html><title>Dashboard unavailable</title>"
                      "<p>Dashboard UI source is unavailable.</p>")
        self._mtime_ns: int | None = None
        self._lock = threading.Lock()
        # Parse the literal instead of importing ``ontology_rgat.viz``. That
        # package also imports plotting and ROS helpers; a read-only HTTP
        # mirror otherwise idles at ~425 MiB before serving its first byte.
        self.page()

    @staticmethod
    def _page_from_source(source: str, filename: str) -> str:
        tree = ast.parse(source, filename=filename)
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "PAGE"
                       for target in targets):
                continue
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise TypeError("dashboard PAGE must be a string literal")
            return value
        raise ValueError("dashboard source does not define PAGE")

    def page(self) -> str:
        try:
            mtime_ns = self.path.stat().st_mtime_ns
        except OSError:
            return self._page
        if mtime_ns == self._mtime_ns:
            return self._page
        with self._lock:
            try:
                mtime_ns = self.path.stat().st_mtime_ns
                if mtime_ns != self._mtime_ns:
                    source = self.path.read_text(encoding="utf-8")
                    self._page = self._page_from_source(source, str(self.path))
                    self._mtime_ns = mtime_ns
            except (OSError, SyntaxError, ValueError, TypeError) as exc:
                print(f"Dashboard UI reload failed; keeping previous page: {exc}",
                      flush=True)
            return self._page


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
            # Importing the model imports PyTorch. Native publishers already
            # provide exact graphs, so a normal mirror must not pay ~350 MiB
            # merely for a legacy fallback it never uses.
            from ontology_rgat.rgat.adaptive_model import (
                FrozenAdaptiveRewardWeights)
            self._model = FrozenAdaptiveRewardWeights(
                self.artifact, expected_config_hash=self.config_hash)
            self._model_mtime_ns = mtime
        return self._model

    @staticmethod
    def _observation(point: dict) -> SemanticObservation | None:
        from ontology_rgat.perception.semantic_observation import (
            SEMANTIC_FEATURE_NAMES, SemanticObservation)
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
            from ontology_rgat.perception.semantic_observation import semantic_graph
            from ontology_rgat.rgat.adaptive_model import adaptive_reward_graph
            from ontology_rgat.viz.graph3d import graph_payload
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


@dataclass(frozen=True)
class RelayResponse:
    """One shared wire representation of the upstream telemetry."""

    body: bytes
    status: int
    stale: bool = False
    error: str | None = None


class StateRelay:
    """Fetch at most one upstream snapshot and share its encoded bytes.

    A dashboard snapshot can contain four bounded step histories and therefore
    be large.  The old mirror decoded, copied and encoded that snapshot in
    every HTTP worker. Slow external connections let several copies coexist;
    in a long run this process reached 4.6 GiB and contributed to the kernel
    OOM-killing the experiment.  This relay serializes fetches, reuses one
    immutable byte string between clients, and serves the last good snapshot
    while the experiment is restarting.
    """

    def __init__(self, source_url: str, enricher: StateEnricher, *,
                 refresh_seconds: float = 1.0,
                 retry_seconds: float = 3.0,
                 cache_path: Path | None = None):
        self.source_url = str(source_url)
        self.enricher = enricher
        self.refresh_seconds = max(0.0, float(refresh_seconds))
        self.retry_seconds = max(self.refresh_seconds, float(retry_seconds))
        self.cache_path = None if cache_path is None else Path(cache_path)
        self._lock = threading.Lock()
        self._body: bytes | None = None
        self._last_attempt = float("-inf")
        self._last_success = float("-inf")
        self._last_persist = float("-inf")
        # Once the upstream publishes exact keyed graphs, enrichment is both
        # unnecessary and scientifically wrong.  Future snapshots can then be
        # relayed byte-for-byte without building a second giant object graph.
        self._native_source = False
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path is None:
            return
        try:
            body = self.cache_path.read_bytes()
            json.loads(body)
        except (OSError, ValueError, TypeError):
            return
        self._body = body

    def _persist(self, body: bytes, now: float) -> None:
        # A restart cache need not follow a 10 Hz revision counter. One atomic
        # snapshot per minute is enough and avoids turning telemetry into I/O.
        if self.cache_path is None or now - self._last_persist < 60.0:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            temporary.write_bytes(body)
            temporary.replace(self.cache_path)
            self._last_persist = now
        except OSError as exc:
            print(f"Dashboard state cache write failed: {exc}", flush=True)

    def get(self) -> RelayResponse:
        now = time.monotonic()
        with self._lock:
            interval = (self.refresh_seconds if self._last_success >= self._last_attempt
                        else self.retry_seconds)
            if now - self._last_attempt < interval:
                if self._body is not None:
                    return RelayResponse(
                        self._body, 200,
                        stale=self._last_success < self._last_attempt)
                return RelayResponse(
                    json.dumps({"error": "upstream telemetry unavailable"}).encode(),
                    502, error="upstream telemetry unavailable")

            self._last_attempt = now
            try:
                request = Request(self.source_url, headers={"Connection": "close"})
                with urlopen(request, timeout=3.0) as response:
                    raw = response.read()
                if self._native_source:
                    body = raw
                else:
                    state = json.loads(raw)
                    if state.get("graphs"):
                        self._native_source = True
                        body = raw
                    else:
                        body = json.dumps(
                            self.enricher.enrich(state), allow_nan=False,
                            separators=(",", ":")).encode("utf-8")
                self._body = body
                self._last_success = now
                self._persist(body, now)
                return RelayResponse(body, 200)
            except Exception as exc:  # dashboard failure must remain isolated
                message = str(exc)
                if self._body is not None:
                    return RelayResponse(
                        self._body, 200, stale=True, error=message)
                return RelayResponse(
                    json.dumps({"error": message}).encode("utf-8"),
                    502, error=message)


def _handler(source_url: str, enricher: StateEnricher,
             live_page: LivePage | None = None,
             relay: StateRelay | None = None):
    live_page = live_page or LivePage()
    relay = relay or StateRelay(source_url, enricher)
    # A browser holding an older fixed-interval page can issue another large
    # request before the prior one crosses a slow tunnel. Bound the wire-side
    # concurrency even in that case; the current page also self-serializes.
    state_slots = threading.BoundedSemaphore(2)
    compression_lock = threading.Lock()
    compressed_source: bytes | None = None
    compressed_body: bytes | None = None

    def wire_body(body: bytes, accept_encoding: str) -> tuple[bytes, str | None]:
        nonlocal compressed_source, compressed_body
        if "gzip" not in accept_encoding.lower():
            return body, None
        with compression_lock:
            if body is not compressed_source:
                compressed_source = body
                # Level 1 is deliberately cheap. Numeric JSON compresses well
                # enough to keep a multi-megabyte live snapshot from spending
                # several polling intervals crossing the public tunnel.
                compressed_body = gzip.compress(body, compresslevel=1)
            return compressed_body or body, "gzip"

    class Handler(BaseHTTPRequestHandler):
        # External reverse proxies may otherwise retain one worker and its
        # response references per idle keep-alive connection.
        protocol_version = "HTTP/1.0"

        def _send(self, body: bytes, content_type: str, status: int = 200,
                  *, relay_response: RelayResponse | None = None,
                  content_encoding: str | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            if content_encoding is not None:
                self.send_header("Content-Encoding", content_encoding)
            if relay_response is not None:
                self.send_header(
                    "X-Dashboard-Upstream",
                    "stale" if relay_response.stale else "live")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):  # noqa: N802 - stdlib API
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in ("/", "/index.html"):
                self._send(live_page.page().encode("utf-8"),
                           "text/html; charset=utf-8")
                return
            if path != "/api/state":
                self.send_error(404)
                return
            if not state_slots.acquire(blocking=False):
                self._send(
                    b'{"error":"telemetry response already in progress"}',
                    "application/json", 429)
                return
            try:
                response = relay.get()
                body, encoding = wire_body(
                    response.body, self.headers.get("Accept-Encoding", ""))
                self._send(body, "application/json", response.status,
                           relay_response=response,
                           content_encoding=encoding)
            finally:
                state_slots.release()

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
    parser.add_argument(
        "--cache-file", type=Path,
        default=Path("/tmp/ontology_rgat_dashboard_state_cache.json"),
        help="last-good telemetry snapshot used across mirror restarts")
    args = parser.parse_args()
    config_hash = None
    if args.manifest and args.manifest.is_file():
        config_hash = json.loads(args.manifest.read_text(encoding="utf-8")).get(
            "config_hash")
    relay = StateRelay(
        args.source_url, StateEnricher(args.artifact, config_hash),
        cache_path=args.cache_file)
    handler = _handler(
        args.source_url, relay.enricher, LivePage(), relay=relay)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler)
    server.daemon_threads = True
    print(f"Read-only hot-reload dashboard: http://{args.host}:{args.port}/",
          flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
