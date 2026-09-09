#!/usr/bin/env bash
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
log=/tmp/ontology_rgat_fake_gateway.log
setsid python3 "$workspace_root/tools/fake_gateway.py" >"$log" 2>&1 &
gateway_pid=$!
trap 'kill -TERM -"$gateway_pid" 2>/dev/null || true' EXIT
sleep 1
# Without this the fake loses port 14650 to a live gateway and MATLAB silently
# validates itself against the real flight stack instead.
if ! kill -0 "$gateway_pid" 2>/dev/null; then
  printf 'Fake gateway failed to start; stop any live gateway on the UDP port.\n' >&2
  cat "$log" >&2
  exit 1
fi
matlab -batch "cd('$workspace_root/matlab'); cfg=defaultExternalConfig('quick','sitl'); env=sim.resetState(42,cfg); cur=sim.getCurrent(env,cfg); [env2,next,r,done,info]=sim.step(env,[0;-0.1;0.2;0],cur,'sparse',[],cfg); assert(numel(next.obs)==15 && isfinite(r) && ~done && strcmp(info.status,'running')); env2.bridge.disarm(); fprintf('MATLAB_PROTOCOL_INTEGRATION=PASS\\n');"
