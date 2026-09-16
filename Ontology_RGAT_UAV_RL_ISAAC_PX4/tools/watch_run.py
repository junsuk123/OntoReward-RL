#!/usr/bin/env python3
"""Print a running pipeline's progress from its dashboard API.

The dashboard serves one page, and a browser is not always the convenient way
to answer "is this still going somewhere" -- over SSH, or while the page a
long-lived process is serving predates the panels you want. This reads the
same ``api/state`` the page reads and prints it, so it works against a run
that is already in flight.

Read-only: it never writes to the run and cannot disturb it.

    tools/watch_run.py                 # one snapshot from 127.0.0.1:8770
    tools/watch_run.py --follow        # refresh until interrupted
    tools/watch_run.py --port 8771
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any, Mapping


def fetch(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as reply:
        return json.loads(reply.read().decode("utf-8"))


def _number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if number != number else f"{number:.{digits}f}"


def _bar(done: float, total: float, width: int = 24) -> str:
    total = max(float(total), 1.0)
    filled = int(width * max(0.0, min(1.0, float(done) / total)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def render(state: Mapping[str, Any]) -> str:
    scalars = dict(state.get("scalars") or {})
    series = dict(state.get("series") or {})
    stage = dict(state.get("stage") or {})
    methods = list(scalars.get("benchmark_methods") or [])
    lines = [
        f"stage   {stage.get('name', '?')}"
        + (f"  ({stage.get('detail')})" if stage.get("detail") else ""),
        f"phase   {scalars.get('benchmark_phase', '?')}",
    ]

    trained = sum(len(series.get(f"benchmark_train_{name}", ())) for name in methods)
    total = int(scalars.get("training_total") or 0)
    lines.append(f"PPO     {_bar(trained, total)} {trained}/{total} episodes")
    for name in methods:
        rows = series.get(f"benchmark_train_{name}", ())
        evaluated = series.get(f"benchmark_eval_{name}", ())
        # Score only rows that actually carry the field. Defaulting a missing
        # one to zero would report a failure the run never had.
        success = [float(row["paper_success"]) for row in rows[-50:]
                   if isinstance(row.get("paper_success"), (int, float))]
        rate = (f"{100 * sum(success) / len(success):.1f}%" if success else "--")
        lines.append(f"  {name:<30} train {len(rows):>4}  최근50 성공 {rate:>6}"
                     f"  eval {len(evaluated):>3}")

    for pair in scalars.get("parallel_pair_status") or []:
        lines.append(
            f"  pair {pair.get('index')}  {pair.get('active_method') or pair.get('method')}"
            f"  {pair.get('phase', '?')}  ep={pair.get('episode')}"
            f"  step={pair.get('step')}  [{pair.get('status', '?')}]")

    data = scalars.get("fov_dataset")
    if data:
        lines.append(
            f"FOV데이터 {_bar(data.get('episodes', 0), data.get('maximum', 1))}"
            f" {data.get('episodes')}/최소 {data.get('minimum')}"
            f"·상한 {data.get('maximum')}"
            f"  두regime={'예' if data.get('covered') else '아직'}"
            f"  지도가능 {data.get('supervised_samples')}"
            f"  평균y {_number(data.get('target_mean'))}")
    training = scalars.get("fov_training")
    if training:
        lines.append(
            f"FOV학습  {_bar(training.get('epoch', 0), training.get('total_epochs', 1))}"
            f" epoch {training.get('epoch')}/{training.get('total_epochs')}"
            f"  val {_number(training.get('validation_loss'), 5)}"
            f"  best {_number(training.get('best_validation_loss'), 5)}")
    model = scalars.get("fov_model")
    if model:
        rmse, base = model.get("rmse"), model.get("constant_predictor_rmse")
        try:
            verdict = "상수예측기보다 나음" if float(rmse) < float(base) else "상수예측기 이하"
        except (TypeError, ValueError):
            verdict = "--"
        lines.append(
            f"FOV모델  {str(model.get('design_id') or '--')[:20]}"
            f"  RMSE {_number(rmse)} / 상수 {_number(base)}  {verdict}"
            f"  동결={'예' if model.get('frozen') else '아니오'}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--follow", action="store_true",
                        help="refresh until interrupted")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/api/state"
    while True:
        try:
            print(render(fetch(url, args.timeout)))
        except urllib.error.URLError as exc:
            print(f"대시보드에 연결할 수 없습니다 ({url}): {exc.reason}")
            if not args.follow:
                return 1
        except (ValueError, KeyError) as exc:
            print(f"예상과 다른 응답입니다: {exc}")
            if not args.follow:
                return 1
        if not args.follow:
            return 0
        print("-" * 72, flush=True)
        try:
            time.sleep(max(1.0, args.interval))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
