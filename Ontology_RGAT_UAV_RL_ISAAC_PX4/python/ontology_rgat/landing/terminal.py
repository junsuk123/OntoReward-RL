"""The one terminal-reward table, discount horizon and curriculum ramp.

Source of truth: upstream MATLAB
``src/+landing2d/+config/defaultPlanarVisibilityConfig.m`` (reward struct) and
``src/+landing2d/+config/primaryConfig.m`` (unsafePenaltyCurriculumStart), at
``ugv_landing_2d_workspace@83f10d6``. Both the 2D port and the 3D extension
read these; a dimension study is only a dimension study if the objective is
identical across dimensions.

Why this module exists
----------------------
The table previously existed in four places that drifted apart:
``two_axis/config.py`` defaults, ``two_axis_reference_v28.yaml``,
``two_axis_context_rgat_comparison.yaml`` and ``spatial/environment.py``
class attributes. The spatial copy was moved to TASK_TIMEOUT -30 / SAFE_ABORT
-40 / unsafe -50 with ``discount_tau`` 350 on 2026-10-05, on a derivation that
landing only beat waiting above a 53.9 % success rate. The derivation compared
terminal values alone and omitted the dense readiness/potential terms, which
pay an approaching arm a large positive running return. Measured over 60 seeds
with one controller and only the table swapped:

====================  ===========================  ==========================
arm                   TO -30 / AB -40 / UN -50     TO -12 / AB -15 / UN -40
                      tau 350                      tau 70 (this table)
====================  ===========================  ==========================
attempt (86.7 % land)  +20.29                       +19.14
hold to the deadline   -24.45                       -4.26
zero action            -38.68                       -12.48
====================  ===========================  ==========================

Attempting wins under both, so the divergence bought nothing and cost 2D/3D
parity. Guarded by ``test_attempting_outranks_waiting_on_both_routes``, which
measures returns instead of deriving a break-even.
"""
from __future__ import annotations

# Order is load-bearing: it is serialized into ExperimentConfig's sha256.
REFERENCE_TERMINAL_BONUS: tuple[tuple[str, float], ...] = (
    ("SUCCESS", 25.0), ("SAFE_ABORT", -15.0),
    ("TASK_TIMEOUT", -12.0), ("UNSAFE_CONTACT", -40.0),
    ("UNAUTHORIZED_CONTACT", -40.0),
    ("MISSED_PAD_CONTACT", -40.0),
    ("SAFETY_ENVELOPE_VIOLATION", -40.0),
)
REFERENCE_TERMINAL_REWARDS: dict[str, float] = dict(REFERENCE_TERMINAL_BONUS)

#: ``reward.discountTimeConstant`` and ``reward.referenceTime`` upstream.
REFERENCE_DISCOUNT_TAU_S = 70.0
REFERENCE_MISSION_HORIZON_S = 70.0

#: ``cfg.rl.unsafePenaltyCurriculumStart`` in primaryConfig.m.
REFERENCE_CURRICULUM_START_UNSAFE = -20.0

#: Price of sustained body rate, paid in the RUNNING term of both routes.
#:
#: Angular rate already appears in ``readiness``, but readiness is paid as a
#: difference, so its sum telescopes to the endpoint and a vehicle that spins
#: for the whole approach pays nothing for it. That is not academic: the 3D
#: terminal-descent corridor arms only while ``|w_xy| <= touchdown_rate``, and
#: on the run1 checkpoints the rate condition -- not tilt -- is what fails.
#: The quantity is ``(|w_xy| / touchdown_rate)^2``, the supervisor's own
#: settled test, so a vehicle sitting exactly on its rate limit pays 1.0 in
#: either dimension. Measured over four nominal episodes per arm with the
#: deterministic action that evaluation uses:
#:
#: ====================  ==========  ===============  =============
#: controller            mean spin   corridor armed   lands
#: ====================  ==========  ===============  =============
#: ppo_vector_canonical       21.74            0.0 %        0 %
#: ppo_ontology_rgat           8.22           29.3 %        0 %
#: diagnostic PD               3.74           77.7 %    25-35 %
#: ppo_semantic_flat           0.38           75.4 %        0 %
#: ====================  ==========  ===============  =============
#:
#: `semantic_flat` is already smoother than the controller that lands, so this
#: price is not a complete explanation of the training failure and must not be
#: presented as one. Over a 70 s episode the term costs ``weight * mean_spin``.
#: Both routes read this constant; a dimension study needs one objective.
#:
#: DEFAULT 0.0, because weight 1.0 was trained and measured and lost. Over
#: 2 seeds x 3 arms x 200 iterations against the identical run at weight 0:
#:
#: * it did what it targeted -- ``ppo_vector_canonical`` went from 0.0 % to
#:   92-98 % of in-band steps inside the body-rate limit and 0.0 % to 84-85 %
#:   corridor arming, beating the controller that lands, and training SUCCESS
#:   rose 1/14400 to 23/14400 (22 of them that arm)
#: * held-out landing stayed 0.0 % in all six runs
#: * it bought the rate with TILT: ``ppo_ontology_rgat`` seed 828 went from
#:   79.9 % to 17.5 % inside the tilt limit, median tilt 3.25 -> 14.23 deg
#:   against a 5 deg limit, because holding a large steady attitude is the
#:   cheapest way to stop rotating
#: * SAFE_ABORT rose 26.70 % to 43.57 % while TASK_TIMEOUT fell 69.18 % to
#:   52.25 %, i.e. DOWN the terminal ordering
#:
#: Priced alone it leaks into tilt. Raise it only together with a tilt term,
#: and only with a trained comparison; ``roll_pitch`` has the same telescoping
#: problem and is the obvious companion. At 0.0 both routes are byte-identical
#: to the upstream port, which ``tools/contract_snapshot.py`` requires.
REFERENCE_SPIN_WEIGHT = 0.0

#: Identified by status, never by magnitude. A table revision once gave
#: SAFE_ABORT the unsafe outcomes' value, and every call site that matched the
#: literal -40.0 silently reclassified aborts as crashes.
UNSAFE_REASONS: tuple[str, ...] = (
    "UNSAFE_CONTACT", "UNAUTHORIZED_CONTACT",
    "MISSED_PAD_CONTACT", "SAFETY_ENVELOPE_VIOLATION",
)

ORDERED_REASONS: tuple[str, ...] = ("SUCCESS", "TASK_TIMEOUT", "SAFE_ABORT")


def validate_terminal_ordering(table) -> None:
    """SUCCESS > TASK_TIMEOUT > SAFE_ABORT > unsafe outcomes.

    A controller that keeps the pad in view until the deadline must outrank one
    that induces its own visual loss, otherwise aborting is a reward shortcut.
    """
    table = dict(table)
    missing = [name for name in ORDERED_REASONS + UNSAFE_REASONS
               if name not in table]
    if missing:
        raise ValueError(f"terminal table is missing {missing}")
    unsafe = max(table[name] for name in UNSAFE_REASONS)
    ordering = [(name, table[name]) for name in ORDERED_REASONS]
    ordering.append(("unsafe_outcomes", unsafe))
    for (left, lvalue), (right, rvalue) in zip(ordering, ordering[1:]):
        if not lvalue > rvalue:
            raise ValueError(
                f"terminal reward ordering violated: {left} ({lvalue}) must "
                f"rank above {right} ({rvalue})")


def validate_curriculum_ramp(start, table) -> None:
    """The outcome ranking must hold at every rung, not only at nominal.

    The curriculum softens the unsafe penalty so a first landing attempt is
    affordable. Softening it past SAFE_ABORT inverts the ranking on the easy
    rungs -- crashing becomes cheaper than aborting -- and a policy trained
    there learns to dive. two_axis measured 92-98 % unsafe contact doing
    exactly that. Checking only the nominal table missed it.
    """
    table = dict(table)
    start = float(start)
    if not start < table["SAFE_ABORT"]:
        raise ValueError(
            f"curriculum start_unsafe_contact_penalty ({start}) must stay below "
            f"SAFE_ABORT ({table['SAFE_ABORT']}); otherwise an unsafe "
            "contact outranks a safe abort on the easy rungs")
    if start > max(table[name] for name in UNSAFE_REASONS):
        return  # the ramp only ever tightens from here, which is the intent
    raise ValueError("the unsafe ramp must start softer than its nominal value")


def validate_discount_horizon(discount_tau_s, mission_horizon_s) -> None:
    """Credit assignment must not make WHEN an outcome lands dominate WHAT it was.

    This is a sanity bound, not the feasibility criterion: a derivation from
    discounted terminal values alone is what produced the unnecessary 2D/3D
    divergence this module documents. The falsifiable check is the measured
    attempt-versus-wait comparison.
    """
    if not discount_tau_s > 0 or not mission_horizon_s > 0:
        raise ValueError("discount horizon and mission horizon must be positive")
    if discount_tau_s < 0.5 * mission_horizon_s:
        raise ValueError(
            f"discount_tau ({discount_tau_s} s) discounts the deadline below "
            f"exp(-2) of an immediate outcome over a {mission_horizon_s} s "
            "mission; outcome timing would outweigh outcome identity")
