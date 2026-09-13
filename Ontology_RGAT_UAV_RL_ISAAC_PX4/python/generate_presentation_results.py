#!/usr/bin/env python3
"""실험 산출물에서 세미나 슬라이드용 지표와 그림을 만든다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ontology_rgat.evaluation.presentation import write_presentation_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="manifest.json이 있는 실험 결과 디렉터리")
    parser.add_argument("--output-dir", type=Path,
                        help="기본값: <results-dir>/presentation")
    args = parser.parse_args()
    summary = write_presentation_results(args.results_dir, args.output_dir)
    print(json.dumps({
        "performance_status": summary["performance_status"],
        "reward_artifact_status": summary["reward_artifact_status"],
        "figures": summary["figures"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
