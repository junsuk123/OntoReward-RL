#!/usr/bin/env python3
"""Render the parallel RViz layout for N UAV/UGV pairs.

``rviz/ontology_rgat_parallel.rviz`` is hand-authored for exactly two pairs, so
a run sized by the machine rather than by that file would show two of however
many are flying. Rather than hand-maintain a layout per pair count, this clones
the checked-in pair group, which stays the single place the per-pair displays
are defined.

    make_rviz_layout.py --pairs 4 --output /tmp/layout.rviz
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "rviz/ontology_rgat_parallel.rviz"
# Group titles name the two arms of the primary comparison. Beyond them a pair
# is a replica of one of those arms, and the learner is what decides which, so
# the title says only the index rather than guessing at a method name.
ARM_TITLES = ("Baseline Shin SE fixed", "Proposed + Ontology-R-GAT FOV")


def _reindex(node, source: int, target: int):
    """Copy ``node`` with every pair-``source`` reference pointed at ``target``."""
    text = json.dumps(node)
    for pattern in ("/landing_rl/pair_%d/", "/landing_pair_%d/",
                    "landing_pad_%d", "uav_body_%d"):
        text = text.replace(pattern % source, pattern % target)
    return json.loads(text)


def build(pairs: int, template_path: Path = TEMPLATE,
          arm_titles: Sequence[str] | None = None) -> dict:
    if pairs < 1:
        raise ValueError("--pairs must be at least one")
    layout = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    manager = layout["Visualization Manager"]
    displays = manager["Displays"]
    groups = [d for d in displays if d.get("Class") == "rviz_common/Group"]
    if not groups:
        raise ValueError(f"{template_path} defines no pair group to clone")
    template_group = groups[0]
    others = [d for d in displays if d.get("Class") != "rviz_common/Group"]

    # Pairs are handed to the arms in contiguous blocks, matching
    # run_three_pipeline._training_pair_replicas: with four pairs and two arms
    # the split is {arm0: [0, 1], arm1: [2, 3]}, not an alternation. Labelling
    # them alternately would put the wrong method's name over a camera.
    # Titles name the arms the *training* pairs belong to. A run may compare
    # more arms than it trains -- the 2026-09-22 comparison flies a non-learned
    # visual servo that takes no training pair -- so the caller passes the arms
    # that actually hold a pair rather than this file assuming two.
    titles = [str(title) for title in (arm_titles or ARM_TITLES)] or list(ARM_TITLES)
    per_arm = max(1, pairs // len(titles))
    rebuilt = []
    for index in range(pairs):
        group = _reindex(copy.deepcopy(template_group), 0, index)
        arm_index = min(index // per_arm, len(titles) - 1)
        arm = titles[arm_index]
        if per_arm > 1:
            arm = f"{arm} · replica {index % per_arm + 1}/{per_arm}"
        group["Name"] = f"Pair {index + 1} · {arm}"
        for child in group.get("Displays", []):
            name = str(child.get("Name", ""))
            if "camera" in name.lower():
                child["Name"] = f"Pair {index + 1} camera · {arm}"
        rebuilt.append(group)

    # The TF display lists the frames it draws; give it every pair's.
    for display in others:
        frames = display.get("Frames")
        if isinstance(frames, dict):
            kept = {key: value for key, value in frames.items()
                    if not key.startswith(("landing_pad_", "uav_body_"))}
            for index in range(pairs):
                kept[f"landing_pad_{index}"] = {"Value": True}
                kept[f"uav_body_{index}"] = {"Value": True}
            display["Frames"] = kept
    manager["Displays"] = others + rebuilt
    return layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--template", default=str(TEMPLATE))
    parser.add_argument(
        "--arm-title", action="append", dest="arm_titles", default=None,
        help="name of an arm that holds training pairs, in pair order; "
             "repeat once per arm (default: the two-arm comparison)")
    args = parser.parse_args()
    layout = build(args.pairs, Path(args.template), args.arm_titles)
    Path(args.output).write_text(
        yaml.safe_dump(layout, sort_keys=False, allow_unicode=True),
        encoding="utf-8")


if __name__ == "__main__":
    main()
