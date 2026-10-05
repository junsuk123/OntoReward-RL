#!/usr/bin/env python3
"""Open an owned, disarmed UAV/pad/UGV side view; never run a policy.

The flight lock/profile guards are shared with actual training. This viewer
does not adopt a live experiment, reset/arm a vehicle or count setup support
as a learned landing. SIGTERM/Ctrl-C stops only its owned stack.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from ontology_rgat.spatial.core import SpatialConfig, schema_for
from ontology_rgat.spatial.environment import IsaacBackend
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.two_axis.artifacts import json_text
from run_spatial_pipeline import live_stack


def disarmed_view_snapshot(bridge):
    state=bridge.transact('state',{},('state',))
    if state.get('armed') is not False:
        raise RuntimeError('Idle viewer requires confirmed disarmed telemetry')
    return dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
                purpose='UAV/pad/UGV side-view scene; no policy running',
                read_only_queries=True,reset_or_arm_sent=False,
                task_landing_evidence=False,
                note='Disarmed setup support is not a learned landing.',state=state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contract-version',choices=['5','6','7','8','9','10', 'reference'], default='reference')
    parser.add_argument('--minutes',type=int,choices=range(1,241),default=60)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    schema=schema_for(args.contract_version)
    cfg=replace(SpatialConfig(),schema=schema,
                isaac_profile_sha256=deployment_profile(schema)['sha256'])
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    with live_stack(args.output,schema=schema,headless=False):
        backend=IsaacBackend(cfg)
        try:
            report=disarmed_view_snapshot(backend.bridge)
            (args.output/'initial-disarmed-state.json').write_text(json_text(report,indent=2)+'\n')
            print('DISARMED_SIDE_VIEW_READY; no policy/reset/arm commands',flush=True)
            deadline=time.monotonic()+60*args.minutes
            while not stopped and time.monotonic()<deadline:
                report=disarmed_view_snapshot(backend.bridge)
                temporary=args.output/'latest-state.json.partial'
                temporary.write_text(json_text(report,indent=2)+'\n')
                temporary.replace(args.output/'latest-state.json')
                time.sleep(2.)
        finally:
            # No reset occurred, so this closes only the read-only bridge.
            backend.close()


if __name__=='__main__':
    main()
