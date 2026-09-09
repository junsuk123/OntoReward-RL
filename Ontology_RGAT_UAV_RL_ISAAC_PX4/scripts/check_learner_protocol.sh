#!/usr/bin/env bash
# The Python learner against tools/fake_gateway.py: the protocol, the
# pad-relative state, the ontology, the observation and one full episode,
# without ROS, PX4 or Isaac.
#
# It binds the gateway UDP port, so it refuses to run alongside a live gateway
# rather than quietly validating the learner against the real flight stack.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
port=${ONTOLOGY_RGAT_FAKE_PORT:-14650}
log=/tmp/ontology_rgat_fake_gateway.log

if ss -lnuH 2>/dev/null | grep -q ":${port} "; then
  printf 'UDP %s is already bound. Stop the live gateway first, or set\n' "$port" >&2
  printf 'ONTOLOGY_RGAT_FAKE_PORT to an unused port.\n' >&2
  exit 1
fi

setsid python3 "$workspace_root/tools/fake_gateway.py" --port "$port" >"$log" 2>&1 &
gateway_pid=$!
trap 'kill -TERM -"$gateway_pid" 2>/dev/null || true' EXIT
sleep 1
if ! kill -0 "$gateway_pid" 2>/dev/null; then
  printf 'Fake gateway failed to start.\n' >&2
  cat "$log" >&2
  exit 1
fi

PYTHONPATH="$workspace_root/python:${PYTHONPATH:-}" \
  ONTOLOGY_RGAT_FAKE_PORT="$port" python3 - <<'PYEOF'
import math
import os

from ontology_rgat.config import default_config
from ontology_rgat.env import LandingEnv, run_episode
from ontology_rgat.expert import PolicySpec

port = int(os.environ["ONTOLOGY_RGAT_FAKE_PORT"])
cfg = default_config("quick")
cfg.external.gateway_port = port
cfg.external.local_port = port + 1
cfg.viz.rviz.enabled = False
cfg.viz.dashboard.enabled = False

env = LandingEnv.reset(42, cfg)
try:
    cur = env.current()
    assert cur.obs.size == cfg.rl.obs_dim == 23, cur.obs.size
    assert cur.graph.X.shape[1] == 14 and cur.graph.goal_node == 13
    assert cur.meas["pad_speed"] > 0, "the fake deck is not moving"
    assert cur.meas["battery"]["enabled"], "the fake pack reports no energy"
    # The fake reports a canyon fix, so a run that reads open sky here means
    # the gnss block is not reaching the learner at all.
    assert 0.0 < cur.sem.gnss_integrity < 1.0, cur.sem.gnss_integrity
    assert env.last_diag["gnss"]["enabled"], "the fake link reports no receiver"
    assert "error_enu_m" not in env.last_diag["gnss"], "the true error leaked"
    # And the fake offsets its truth block, so scoring must not equal sensing.
    assert env.last_diag["ground_truth"]["valid"]
    nxt, reward, done, info = env.step([0.0, -0.1, 0.2, 0.0], cur, "sparse", None)
    assert nxt.obs.size == 23 and not done and info["status"] == "running"
    assert math.isfinite(reward) and math.isfinite(info["energy_used_j"])
    assert info["pad_speed"] > 0 and info["battery_reserve"] >= 0
    assert info["position_error_m"] > 0.5, info["position_error_m"]
finally:
    env.close()

log = run_episode(PolicySpec("expert"), "manual", None, 7, cfg)
assert log.metrics["steps"] == cfg.sim.max_steps, log.metrics["steps"]
assert log.metrics["duration_s"] > 0.9 * cfg.sim.max_time, log.metrics["duration_s"]
print("LEARNER_PROTOCOL_INTEGRATION=PASS")
PYEOF
