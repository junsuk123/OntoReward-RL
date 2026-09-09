from __future__ import annotations

import argparse

from .batch import TemporalBatchRunner
from .config import DEFAULT_TEMPORAL_CONFIG,load_temporal_config


def main()->int:
    parser=argparse.ArgumentParser(description="Accelerated temporal environment change experiment")
    parser.add_argument("--config",default=str(DEFAULT_TEMPORAL_CONFIG));args=parser.parse_args()
    print(TemporalBatchRunner(load_temporal_config(args.config)).run());return 0


if __name__=="__main__":raise SystemExit(main())
