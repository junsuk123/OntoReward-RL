#!/usr/bin/env python3
"""Primary scientific runner: Shin baseline versus baseline + FOV-risk R-GAT."""
from __future__ import annotations

from run_three_pipeline import main


if __name__ == "__main__":
    try:
        exit_code = main(primary_only=True)
    except KeyboardInterrupt:
        print("Pipeline interrupted; checkpoints and completed rows were preserved.")
        exit_code = 130
    raise SystemExit(exit_code)

