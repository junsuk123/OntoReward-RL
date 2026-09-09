#!/usr/bin/env python3
"""Measure the R-GAT CPU/GPU crossover on this machine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ontology_rgat.config import default_config
from ontology_rgat.rgat.benchmark import benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+",
                        default=[32, 256, 1024, 4096, 16384])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--mode", default="quick", choices=("quick", "full"))
    args = parser.parse_args()
    benchmark(args.batches, args.repetitions, default_config(args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
