"""Resolved deployment manifest, independent of spectator camera placement."""
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import time
from ..benchmarks.experiment import load_experiment, configuration_hash


def wait_runtime_identity(bridge, *, profile_sha256, source_sha256, timeout=30.):
    """Read-only bounded DDS discovery wait, never tolerate a known mismatch.

    A listening UDP gateway can precede its first Isaac clock subscription.
    Missing/stale clock data may become ready; a reported different identity
    cannot. This is pre-reset readiness, not policy-time fault recovery.
    """
    if not math.isfinite(timeout) or not 0 < timeout <= 30.:
        raise ValueError('identity readiness timeout must be in (0,30] seconds')
    started = time.monotonic()
    deadline = started + timeout
    attempts = 0
    clock = {}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f'running Isaac identity unavailable after {timeout}s; '
                             f'no reset/arm sent; last clock={clock!r}')
        state = bridge.transact('state', {}, ('state',), timeout=min(8., remaining))
        attempts += 1
        clock = (state.get('extra') or {}).get('spatial_clock') or {}
        if not isinstance(clock, dict):
            raise ValueError('running Isaac identity is malformed; no reset/arm sent')
        for key, expected in (('profile_sha256', profile_sha256),
                              ('source_sha256', source_sha256)):
            observed = clock.get(key)
            if observed is not None and observed != expected:
                raise ValueError(f'running Isaac {key} is stale: '
                                 f'{observed!r} != {expected!r}; restart owned stack')
        if (time.monotonic() < deadline and clock.get('valid') is True
                and clock.get('profile_sha256') == profile_sha256
                and clock.get('source_sha256') == source_sha256):
            return dict(attempts=attempts, wall_seconds=time.monotonic()-started,
                        clock=dict(clock), read_only=True, reset_or_arm_sent=False)
        time.sleep(min(.05, max(0., deadline-time.monotonic())))


def scientific_configuration(resolved):
    scientific = deepcopy(resolved)
    scientific.get("isaac", {}).pop("viewport_follow", None)
    scientific.get("parallel", {}).pop("operator_view", None)
    return scientific


def runtime_profile_hash(resolved):
    return configuration_hash(scientific_configuration(resolved))


def runtime_source_hash(root=None):
    """Detect an adopted world still running old code after a source edit.

    Kept separate from the scientific YAML/checkpoint signature: bugfixes may
    be checkpoint-compatible, but the actual running world must be current.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    paths = sorted((root/"isaac_sim").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def deployment_profile(schema="spatial-causal-rgat/5"):
    from .core import REFERENCE_SCHEMAS

    filename = ("spatial-isaac-system-v6.yaml" if schema in ("spatial-causal-rgat/6", "spatial-causal-rgat/7", "spatial-causal-rgat/8", "spatial-causal-rgat/9")
                else "spatial-isaac-system.yaml")
    if schema in ('spatial-causal-rgat/9', 'spatial-causal-rgat/10',
                  *REFERENCE_SCHEMAS):
        # The unified contract flies the v9/v10 world: same camera mount,
        # direct-acceleration action and seeded disturbances. Only the packet
        # layout and the per-axis graph changed, and neither is deployed state.
        filename = 'spatial-isaac-system-v9.yaml'
    if schema in ("spatial-reference/4", "spatial-reference/5", "spatial-reference/6",
                  "spatial-reference/7"):
        # The Isaac-calibrated vehicle: landing gear and the 17-tag board.
        filename = 'spatial-isaac-system-v11.yaml'
    if schema == "spatial-reference/8":
        # v11 without the legs: the stock Iris gear and the same board.
        filename = 'spatial-isaac-system-v12.yaml'
    path = Path(__file__).resolve().parents[3] / "config" / filename
    scientific = scientific_configuration(load_experiment(path))
    return {
        "path": str(path),
        "sha256": configuration_hash(scientific),
        "resolved_scientific_configuration": scientific,
        "world_source_sha256": runtime_source_hash(),
    }
