#!/usr/bin/env python3
"""Serve or export the live MATLAB-port training and result dashboard."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.direct_policy.visualization import PAGE, export_report, snapshot  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--planned-updates", type=int)
    parser.add_argument("--episodes-per-update", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--once", action="store_true", help="print current state JSON")
    parser.add_argument("--export", action="store_true",
                        help="write visualization/index.html and state.json, then exit")
    args = parser.parse_args(argv)
    if not args.run_root.is_absolute():
        args.run_root = (ROOT / args.run_root).resolve()

    def current():
        return snapshot(args.run_root, planned_updates=args.planned_updates,
                        episodes_per_update=args.episodes_per_update)

    if args.once:
        print(json.dumps(current(), indent=2, ensure_ascii=False))
        return 0
    if args.export:
        state = export_report(args.run_root, planned_updates=args.planned_updates,
                              episodes_per_update=args.episodes_per_update)
        print(args.run_root.resolve() / "visualization" / "index.html")
        print(json.dumps({"state": state["state"], "arms": len(state["arms"])},
                         ensure_ascii=False))
        return 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib interface
            if self.path.rstrip("/") in ("", "/index.html"):
                body = PAGE.replace("__EMBEDDED_STATE__", "null").encode("utf-8")
                content_type = "text/html; charset=utf-8"
            elif self.path.startswith("/api/state"):
                body = json.dumps(current(), ensure_ascii=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MATLAB-port dashboard: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
