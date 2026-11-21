# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Owned Isaac Sim + PX4 SITL stack for one flight session (moved verbatim from the archived
``python/run_spatial_pipeline.py`` on 2026-11-20, when every non-ORL route was archived).

``live_stack`` takes the single flight lock, refuses to take over an Isaac that is already
running unless ``adopt``, rebuilds a stale gateway only when nothing is running, starts the
owned stack and always stops it on exit.  The current-semantics CLI (ORL and its comparison
arms) is its only caller.
"""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import subprocess

from ontology_rgat.config import default_config
from ontology_rgat.spatial.lifecycle import used_reset_recoveries
from ontology_rgat.spatial.runtime_contract import deployment_manifest, deployment_profile
from ontology_rgat.stack import ExternalStack, current as current_stack

ROOT = Path(__file__).resolve().parents[2]   # the package root holding scripts/ and config/


@contextmanager
def live_stack(output, *, adopt=False, headless=False, schema="spatial-causal-rgat/5",
               reset_recoveries=0, isolate_episodes=False, profile_path=None):
    if isolate_episodes and adopt:
        raise ValueError('fresh-stack episodes require ownership, never adoption')
    if not 0 <= reset_recoveries <= 5:
        raise ValueError('reset recovery budget must be in [0,5]')
    manifest = (deployment_manifest(profile_path) if profile_path is not None
                else deployment_profile(schema))
    profile = Path(manifest["path"])
    with open("/tmp/ontology_rgat_flight_pipeline.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another flight pipeline owns the stack")
        # Inspect program tokens, never shell/search text containing names.
        active = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                tokens = (entry / "cmdline").read_bytes().decode().split("\0")
            except (OSError, UnicodeError):
                continue
            if any(Path(t).name == "landing_world.py" for t in tokens[:3]):
                active.append(tokens)
        if active and not adopt:
            raise RuntimeError(
                "Isaac already active; refusing takeover without --adopt-stack"
            )
        if active and not all(str(profile) in tokens for tokens in active):
            raise RuntimeError("active Isaac uses a different profile")
        check = subprocess.run(
            [str(ROOT / "scripts/sync_gateway.sh"), "--check"],
            capture_output=True,
            text=True,
        )
        if check.returncode:
            if active:
                raise RuntimeError(
                    "active gateway source is stale; stop owned stack before rebuild"
                )
            subprocess.run([str(ROOT / "scripts/sync_gateway.sh")], check=True)
        stack = ExternalStack(
            default_config("quick"),
            config_path=profile,
            headless=headless,
            viewport_pair_index=0,
            startup_attempts=2,
            log_dir=output / "stack",
        )
        stack.spatial_recovery_budget = int(reset_recoveries)
        stack.spatial_recoveries_used = used_reset_recoveries(output)
        stack.spatial_isolate_episodes=bool(isolate_episodes)
        stack.spatial_episode_count=0
        stack.spatial_last_stop_confirmed=True
        previous_stack = current_stack()
        try:
            stack.start()
            current_stack(stack)
            yield stack
        finally:
            try:
                stack.stop()
            finally:
                current_stack(previous_stack)
