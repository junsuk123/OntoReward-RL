from __future__ import annotations

import argparse

from .config import DEFAULT_EXPERIMENT_CONFIG, load_experiment_config
from .pipeline import ExperimentRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="GNSS-denied ontology-guided terrain visual matching experiment")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    args = parser.parse_args()
    output = ExperimentRunner(load_experiment_config(args.config)).run()
    print(output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
