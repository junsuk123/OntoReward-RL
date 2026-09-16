"""Scientific information boundaries for each deployed RL pipeline.

These flags are not presentation metadata.  Model construction, objectives,
reward dispatch and dataset preparation all consume the same immutable spec so
an estimator-dependent reward cannot be attached to an estimator-free actor by
accident.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


# The one admissible definition of landing-pad field-of-view retention: the pad
# centre projects inside the rendered camera's frustum with positive depth.
# See ``isaac_sim/keypoint_geometry.geometric_pad_center_in_fov``.
GEOMETRIC_FOV_CRITERION = "geometric_pad_center_in_fov"

# Perception settings that belong to the retained legacy ArUco profile and may
# never appear in the primary two-pipeline experiment.
FORBIDDEN_PRIMARY_VISION_KEYS = ("dictionary", "board")

# How the additive FOV reward reads its scalar out of the ontology graph.
# ``direct_graph_scalar`` is the current method: the second relational layer
# has one unit and the FutureFOVUnavailability node of that layer IS the
# output. ``binary_classifier_linear_head`` is the retired readout that put an
# nn.Linear on the goal embedding and regressed a binary loss indicator; it is
# kept as a distinct ID so no old run silently reads as the current method.
FOV_REWARD_READOUTS = ("direct_graph_scalar", "binary_classifier_linear_head")
RETIRED_FOV_REWARD_READOUTS = ("binary_classifier_linear_head",)


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    state_estimation_enabled: bool
    auxiliary_estimation_loss_enabled: bool
    active_perception_enabled: bool
    reward_mode: str
    ontology_enabled: bool
    ontology_input_mode: str | None
    use_direct_rgat_potential: bool
    fov_risk_reward_enabled: bool = False
    reserved_latent_dimensions: int = 6
    use_adaptive_reward_weights: bool = False
    adaptive_reward_architecture: str | None = None
    fov_reward_readout: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pipeline name must not be empty")
        if self.reserved_latent_dimensions != 6:
            raise ValueError("the controlled actor reserves exactly six latent dimensions")
        if self.auxiliary_estimation_loss_enabled and not self.state_estimation_enabled:
            raise ValueError("auxiliary estimation loss requires state estimation")
        if self.state_estimation_enabled != self.auxiliary_estimation_loss_enabled:
            raise ValueError(
                "explicit state estimation and its auxiliary supervision must agree")
        if self.active_perception_enabled and not (
                self.state_estimation_enabled
                and self.auxiliary_estimation_loss_enabled):
            raise ValueError("active perception requires supervised state estimation")
        if self.reward_mode not in {
                "shin_table_active", "shin_table_no_active", "semantic_pbrs",
                "adaptive_weight"}:
            raise ValueError(f"unknown pipeline reward mode: {self.reward_mode}")
        if (self.reward_mode in {"shin_table_active", "shin_table_no_active"}
                and ((self.reward_mode == "shin_table_active")
                     != self.active_perception_enabled)):
            raise ValueError(
                "shin_table_active reward and active perception must agree")
        if not self.fov_risk_reward_enabled and self.fov_reward_readout is not None:
            raise ValueError("only the FOV-risk reward declares a readout")
        if self.fov_risk_reward_enabled:
            if not (self.ontology_enabled and self.state_estimation_enabled
                    and self.auxiliary_estimation_loss_enabled
                    and self.active_perception_enabled):
                raise ValueError("FOV-risk reward is an additive extension of full Shin only")
            if self.reward_mode != "shin_table_active":
                raise ValueError("FOV-risk reward must preserve the Shin reward mode")
            if self.ontology_input_mode != "fov_semantic_observation":
                raise ValueError("FOV-risk reward requires the strict visual ontology input")
            if self.use_direct_rgat_potential or self.use_adaptive_reward_weights:
                raise ValueError("FOV-risk reward cannot use PBRS or adaptive weights")
            if self.fov_reward_readout not in FOV_REWARD_READOUTS:
                raise ValueError(
                    "FOV-risk reward must declare its readout: "
                    f"one of {FOV_REWARD_READOUTS}")
        elif self.ontology_enabled:
            if self.ontology_input_mode != "semantic_observation":
                raise ValueError("legacy ontology pipelines require semantic_observation input")
            if self.reward_mode == "semantic_pbrs":
                if not self.use_direct_rgat_potential or self.use_adaptive_reward_weights:
                    raise ValueError("semantic PBRS must use only the scalar R-GAT potential")
                if self.state_estimation_enabled:
                    raise ValueError("semantic PBRS pipeline cannot enable state estimation")
            elif self.reward_mode == "adaptive_weight":
                if self.use_direct_rgat_potential or not self.use_adaptive_reward_weights:
                    raise ValueError("adaptive reward mode must use the R-GAT weight head")
                if self.adaptive_reward_architecture not in {"rgat", "gat", "mlp"}:
                    raise ValueError("adaptive reward architecture must be rgat, gat or mlp")
            else:
                raise ValueError("ontology pipeline has an incompatible reward mode")
        elif (self.ontology_input_mode is not None or self.use_direct_rgat_potential
              or self.fov_risk_reward_enabled
              or self.use_adaptive_reward_weights
              or self.adaptive_reward_architecture is not None):
            raise ValueError("non-ontology pipeline cannot configure an ontology input/potential")
        if self.reward_mode == "semantic_pbrs" and not self.ontology_enabled:
            raise ValueError("semantic PBRS requires the ontology pipeline")

    @property
    def actor_latent_slice(self) -> slice:
        return slice(self.reserved_latent_dimensions, None)

    def to_manifest(self) -> dict:
        return asdict(self)


PIPELINES = {
    "shin_se_fixed": PipelineSpec(
        name="shin_se_fixed",
        state_estimation_enabled=True,
        auxiliary_estimation_loss_enabled=True,
        active_perception_enabled=True,
        reward_mode="shin_table_active",
        ontology_enabled=False,
        ontology_input_mode=None,
        use_direct_rgat_potential=False,
    ),
    "shin_se_onto_rgat_recovery": PipelineSpec(
        name="shin_se_onto_rgat_recovery",
        state_estimation_enabled=True,
        auxiliary_estimation_loss_enabled=True,
        active_perception_enabled=True,
        reward_mode="shin_table_active",
        ontology_enabled=True,
        ontology_input_mode="fov_semantic_observation",
        use_direct_rgat_potential=False,
        fov_risk_reward_enabled=True,
        fov_reward_readout="direct_graph_scalar",
    ),
}

# Historical and ablation-only definitions are intentionally absent from the
# primary runner. They retain their exact IDs so no old method silently aliases
# to either final scientific pipeline.
LEGACY_PIPELINES = {
    # Retired readout: an nn.Linear head on the goal embedding predicting a
    # binary "FOV lost within H" indicator. Kept under its original ID so its
    # checkpoints and results stay attributable to the method that produced
    # them; it is not runnable in the primary experiment.
    "shin_se_onto_rgat_fov": PipelineSpec(
        name="shin_se_onto_rgat_fov",
        state_estimation_enabled=True,
        auxiliary_estimation_loss_enabled=True,
        active_perception_enabled=True,
        reward_mode="shin_table_active",
        ontology_enabled=True,
        ontology_input_mode="fov_semantic_observation",
        use_direct_rgat_potential=False,
        fov_risk_reward_enabled=True,
        fov_reward_readout="binary_classifier_linear_head",
    ),
    "shin_se": PipelineSpec(
        name="shin_se",
        state_estimation_enabled=True,
        auxiliary_estimation_loss_enabled=True,
        active_perception_enabled=True,
        reward_mode="shin_table_active",
        ontology_enabled=False,
        ontology_input_mode=None,
        use_direct_rgat_potential=False,
    ),
    "no_se": PipelineSpec(
        name="no_se",
        state_estimation_enabled=False,
        auxiliary_estimation_loss_enabled=False,
        active_perception_enabled=False,
        reward_mode="shin_table_no_active",
        ontology_enabled=False,
        ontology_input_mode=None,
        use_direct_rgat_potential=False,
    ),
    "onto_no_se": PipelineSpec(
        name="onto_no_se",
        state_estimation_enabled=False,
        auxiliary_estimation_loss_enabled=False,
        active_perception_enabled=False,
        reward_mode="semantic_pbrs",
        ontology_enabled=True,
        ontology_input_mode="semantic_observation",
        use_direct_rgat_potential=True,
    ),
}

ADAPTIVE_PIPELINES = {
    "shin_se_rgat_weight": PipelineSpec(
        name="shin_se_rgat_weight", state_estimation_enabled=True,
        auxiliary_estimation_loss_enabled=True, active_perception_enabled=True,
        reward_mode="adaptive_weight", ontology_enabled=True,
        ontology_input_mode="semantic_observation", use_direct_rgat_potential=False,
        use_adaptive_reward_weights=True, adaptive_reward_architecture="rgat"),
    "no_se_fixed": PipelineSpec(
        name="no_se_fixed", state_estimation_enabled=False,
        auxiliary_estimation_loss_enabled=False, active_perception_enabled=False,
        reward_mode="shin_table_no_active", ontology_enabled=False,
        ontology_input_mode=None, use_direct_rgat_potential=False),
    "onto_rgat_adaptive_weight_no_se": PipelineSpec(
        name="onto_rgat_adaptive_weight_no_se", state_estimation_enabled=False,
        auxiliary_estimation_loss_enabled=False, active_perception_enabled=False,
        reward_mode="adaptive_weight", ontology_enabled=True,
        ontology_input_mode="semantic_observation", use_direct_rgat_potential=False,
        use_adaptive_reward_weights=True, adaptive_reward_architecture="rgat"),
    "onto_rgat_potential_pbrs_no_se": PipelineSpec(
        name="onto_rgat_potential_pbrs_no_se", state_estimation_enabled=False,
        auxiliary_estimation_loss_enabled=False, active_perception_enabled=False,
        reward_mode="semantic_pbrs", ontology_enabled=True,
        ontology_input_mode="semantic_observation", use_direct_rgat_potential=True),
    "mlp_adaptive_weight_no_se": PipelineSpec(
        name="mlp_adaptive_weight_no_se", state_estimation_enabled=False,
        auxiliary_estimation_loss_enabled=False, active_perception_enabled=False,
        reward_mode="adaptive_weight", ontology_enabled=True,
        ontology_input_mode="semantic_observation", use_direct_rgat_potential=False,
        use_adaptive_reward_weights=True, adaptive_reward_architecture="mlp"),
    "gat_adaptive_weight_no_se": PipelineSpec(
        name="gat_adaptive_weight_no_se", state_estimation_enabled=False,
        auxiliary_estimation_loss_enabled=False, active_perception_enabled=False,
        reward_mode="adaptive_weight", ontology_enabled=True,
        ontology_input_mode="semantic_observation", use_direct_rgat_potential=False,
        use_adaptive_reward_weights=True, adaptive_reward_architecture="gat"),
    "rgat_adaptive_weight_no_se": PipelineSpec(
        name="rgat_adaptive_weight_no_se", state_estimation_enabled=False,
        auxiliary_estimation_loss_enabled=False, active_perception_enabled=False,
        reward_mode="adaptive_weight", ontology_enabled=True,
        ontology_input_mode="semantic_observation", use_direct_rgat_potential=False,
        use_adaptive_reward_weights=True, adaptive_reward_architecture="rgat"),
}
ALL_PIPELINES = {**PIPELINES, **LEGACY_PIPELINES, **ADAPTIVE_PIPELINES}

# Old commands remain accepted, but the three new names are the only primary
# comparison IDs.  In particular, legacy ``ontoreward`` remains legacy rather
# than silently pretending its estimate-based distilled reward is onto_no_se.
LEGACY_PIPELINE_ALIASES = {
    "shin2026": "shin_se",
    "manual_no_active": "no_se",
}


def get_pipeline(name: str) -> PipelineSpec:
    canonical = LEGACY_PIPELINE_ALIASES.get(str(name), str(name))
    try:
        return ALL_PIPELINES[canonical]
    except KeyError as exc:
        raise ValueError(f"unknown pipeline: {name}") from exc


def primary_pipeline_ids() -> tuple[str, ...]:
    return tuple(PIPELINES)


def available_pipeline_ids() -> tuple[str, ...]:
    return tuple(ALL_PIPELINES)


_BASELINE_SPEC_FIELDS = (
    "state_estimation_enabled", "auxiliary_estimation_loss_enabled",
    "active_perception_enabled", "reward_mode", "reserved_latent_dimensions",
)


def assert_primary_baseline_equivalence() -> None:
    """Fail fast if the proposed pipeline changes any baseline contract flag."""
    baseline = PIPELINES["shin_se_fixed"]
    proposed = PIPELINES["shin_se_onto_rgat_recovery"]
    mismatches = [name for name in _BASELINE_SPEC_FIELDS
                  if getattr(baseline, name) != getattr(proposed, name)]
    if mismatches:
        raise RuntimeError(
            f"proposed pipeline changed Shin baseline fields: {mismatches}")
    if baseline.ontology_enabled or baseline.fov_risk_reward_enabled:
        raise RuntimeError("Shin baseline must not enable the ontology branch")
    if not (proposed.ontology_enabled and proposed.fov_risk_reward_enabled):
        raise RuntimeError("proposed pipeline must add the FOV-risk ontology branch")
    if proposed.use_direct_rgat_potential or proposed.use_adaptive_reward_weights:
        raise RuntimeError("primary proposed pipeline cannot use PBRS/adaptive weights")
    if proposed.fov_reward_readout != "direct_graph_scalar":
        raise RuntimeError(
            "the proposed reward must read its scalar from the graph itself")


def validate_pipeline_configuration(config: dict) -> None:
    """Refuse YAML declarations that disagree with executable presets."""
    configured = tuple(config.get("pipelines") or ())
    if not configured:
        raise ValueError("experiment configuration must list pipelines")
    unknown = set(configured) - set(ALL_PIPELINES)
    if unknown:
        raise ValueError(f"unknown configured pipelines: {sorted(unknown)}")
    retired = sorted(name for name in configured
                     if ALL_PIPELINES[name].fov_reward_readout
                     in RETIRED_FOV_REWARD_READOUTS)
    if retired:
        raise ValueError(
            f"pipelines {retired} declare a retired FOV reward readout; their "
            "IDs are kept for attribution only and cannot be run")
    contracts = config.get("pipeline_contract") or {}
    for name in configured:
        spec = ALL_PIPELINES[name]
        declared = contracts.get(name) or {}
        expected = {
            "state_estimation": spec.state_estimation_enabled,
            "auxiliary_estimation_loss": spec.auxiliary_estimation_loss_enabled,
            "active_perception": spec.active_perception_enabled,
            "reward_mode": spec.reward_mode,
            "ontology_enabled": spec.ontology_enabled,
            "ontology_input_mode": spec.ontology_input_mode,
            "use_direct_rgat_potential": spec.use_direct_rgat_potential,
        }
        for key, value in expected.items():
            if declared.get(key) != value:
                raise ValueError(
                    f"pipeline {name} YAML {key} disagrees with executable spec")
        optional_expected = {
            "fov_risk_reward": spec.fov_risk_reward_enabled,
            "use_adaptive_reward_weights": spec.use_adaptive_reward_weights,
            "adaptive_reward_architecture": spec.adaptive_reward_architecture,
            "fov_reward_readout": spec.fov_reward_readout,
        }
        for key, value in optional_expected.items():
            if key in declared and declared[key] != value:
                raise ValueError(
                    f"pipeline {name} YAML {key} disagrees with executable spec")
    ppo_gamma = float((config.get("ppo") or {}).get("gamma", .99))
    design_gamma = float((config.get("rgat_design") or {}).get(
        "outcome_discount", ppo_gamma))
    if abs(ppo_gamma - design_gamma) > 1e-12:
        raise ValueError("R-GAT design/PBRS gamma must equal PPO gamma")
    if any(name in PIPELINES for name in configured):
        assert_primary_baseline_equivalence()
    if any(ALL_PIPELINES[name].fov_risk_reward_enabled for name in configured):
        risk = config.get("fov_risk") or {}
        coefficient = float(risk.get("lambda_fov", 0.1))
        if coefficient < 0.0:
            raise ValueError("fov_risk.lambda_fov must be non-negative")
        criterion = str(risk.get(
            "visibility_criterion", GEOMETRIC_FOV_CRITERION))
        if criterion != GEOMETRIC_FOV_CRITERION:
            raise ValueError(
                "the future-FOV-loss label must be geometric pad-centre "
                f"frustum visibility, not {criterion!r}")
        if float(risk.get("prediction_horizon_seconds", 1.0)) <= 0.0:
            raise ValueError("fov_risk.prediction_horizon_seconds must be positive")
    adaptive_specs = [ALL_PIPELINES[name] for name in configured
                      if ALL_PIPELINES[name].use_adaptive_reward_weights]
    if adaptive_specs:
        adaptive = config.get("adaptive_reward") or {}
        if not bool(adaptive.get("enabled", False)):
            raise ValueError("adaptive reward pipelines require adaptive_reward.enabled")
        if int(adaptive.get("num_components", 0)) != 5:
            raise ValueError("adaptive reward must use exactly five components")
        weights = tuple(float(v) for v in adaptive.get("baseline_weights", ()))
        total = float(adaptive.get("total_weight", 0.0))
        if len(weights) != 5 or any(value <= 0.0 for value in weights):
            raise ValueError("adaptive baseline weights must be five positive values")
        if abs(sum(weights) - total) > 1e-9 or abs(total - 5.5) > 1e-9:
            raise ValueError("adaptive reward weights must preserve total 5.5")
        if not bool(adaptive.get("freeze_during_ppo", False)):
            raise ValueError("adaptive R-GAT must remain frozen during PPO")
        if not 0.0 <= float(adaptive.get(
                "baseline_mixture_epsilon", -1.0)) <= 1.0:
            raise ValueError("adaptive baseline mixture epsilon must be in [0,1]")
        if float(adaptive.get("logit_scale_kappa", -1.0)) < 0.0:
            raise ValueError("adaptive logit scale must be non-negative")
        design = config.get("adaptive_reward_design") or {}
        if abs(float(design.get("trajectory_gamma", ppo_gamma)) - ppo_gamma) > 1e-12:
            raise ValueError("adaptive trajectory gamma must equal PPO gamma")


def assert_no_aruco_in_primary_system(system: Mapping[str, Any]) -> None:
    """Refuse a primary run whose simulator profile still carries ArUco.

    The deployed policy perceives the pad with a learned six-keypoint encoder.
    A marker board, a dictionary or a detector-driven pose source anywhere in
    the same profile would make the visual target and the claimed perception
    method disagree, so this fails the run rather than reporting a comparison
    built on two different perception systems.
    """
    vision = dict((system or {}).get("vision") or {})
    mode = str(vision.get("mode", ""))
    if mode != "keypoint_fiducial":
        raise ValueError(
            "the primary two-pipeline experiment requires vision.mode "
            f"'keypoint_fiducial', not {mode!r}")
    present = [key for key in FORBIDDEN_PRIMARY_VISION_KEYS if vision.get(key)]
    if present:
        raise ValueError(
            f"ArUco settings {present} are forbidden in the primary "
            "experiment; set them to null")
    if bool(vision.get("pose_source_for_policy", False)):
        raise ValueError(
            "a detector-solved pose must never drive the primary policy")
    landing_pad = dict(vision.get("landing_pad") or {})
    if str(landing_pad.get("layout", "hexagonal")) != "hexagonal":
        raise ValueError("the six-keypoint landing target must be hexagonal")
