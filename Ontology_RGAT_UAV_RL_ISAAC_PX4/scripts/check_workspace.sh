#!/usr/bin/env bash
# Everything that can be checked without Isaac, PX4 or ROS.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

python3 -m compileall -q \
  "$workspace_root/ros2_ws/src/ontology_rgat_px4/ontology_rgat_px4" \
  "$workspace_root/python" \
  "$workspace_root/isaac_sim" \
  "$workspace_root/tools"
bash -n "$workspace_root"/scripts/*.sh

# Appended, never replaced: overwriting PYTHONPATH drops ROS 2 off the path,
# and then the ROS-dependent checks silently skip instead of running.
PYTHONPATH="$workspace_root/ros2_ws/src/ontology_rgat_px4:$workspace_root/python:${PYTHONPATH:-}" \
  python3 -m pytest -q "$workspace_root/tests"

# The learner must never grow its own physics. Isaac owns the rigid body and
# PX4 owns the estimator; a rotor model or an integrator appearing here would
# mean the experiment had quietly stopped being flight-stack-in-the-loop.
# grep, not rg: a missing ripgrep used to make this guard pass silently.
if grep -rEn 'rk4|def eom\(|rotor_forces|distributed_aero|integrate_rigid_body' \
  "$workspace_root/python/ontology_rgat"; then
  printf 'The learner has grown an in-process simulator.\n' >&2
  exit 1
fi

# legacy_matlab/ is reference only. Nothing that runs may depend on it.
# This file is excluded because it names the directory in order to look for it.
if grep -rEn --binary-files=without-match --exclude=check_workspace.sh 'legacy_matlab' \
  "$workspace_root/python" "$workspace_root/scripts" "$workspace_root/isaac_sim" \
  "$workspace_root/ros2_ws/src"; then
  printf 'A live component references legacy_matlab/.\n' >&2
  exit 1
fi

printf 'Workspace checks passed.\n'
