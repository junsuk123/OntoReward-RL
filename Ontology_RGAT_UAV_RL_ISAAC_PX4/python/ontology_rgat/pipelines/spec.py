"""Scientific information boundaries for each deployed RL pipeline.

These flags are not presentation metadata.  Model construction, objectives,
reward dispatch and dataset preparation all consume the same immutable spec so
an estimator-dependent reward cannot be attached to an estimator-free actor by
accident.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


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
    reserved_latent_dimensions: int = 6

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
                "shin_table_active", "shin_table_no_active", "semantic_pbrs"}:
            raise ValueError(f"unknown pipeline reward mode: {self.reward_mode}")
        if ((self.reward_mode == "shin_table_active")
                != self.active_perception_enabled):
            raise ValueError(
                "shin_table_active reward and active perception must agree")
        if self.ontology_enabled:
            if self.ontology_input_mode != "semantic_observation":
                raise ValueError("ontology pipelines require semantic_observation input")
            if not self.use_direct_rgat_potential:
                raise ValueError("primary ontology pipeline must use direct R-GAT potential")
            if self.state_estimation_enabled:
                raise ValueError("semantic ontology pipeline cannot enable state estimation")
            if self.reward_mode != "semantic_pbrs":
                raise ValueError("ontology pipeline must use semantic PBRS")
        elif self.ontology_input_mode is not None or self.use_direct_rgat_potential:
            raise ValueError("non-ontology pipeline cannot configure an ontology input/potential")
        if self.reward_mode == "semantic_pbrs" and not self.ontology_enabled:
            raise ValueError("semantic PBRS requires the ontology pipeline")

    @property
    def actor_latent_slice(self) -> slice:
        return slice(self.reserved_latent_dimensions, None)

    def to_manifest(self) -> dict:
        return asdict(self)


PIPELINES = {
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
        return PIPELINES[canonical]
    except KeyError as exc:
        raise ValueError(f"unknown pipeline: {name}") from exc


def primary_pipeline_ids() -> tuple[str, ...]:
    return tuple(PIPELINES)


def validate_pipeline_configuration(config: dict) -> None:
    """Refuse YAML declarations that disagree with executable presets."""
    configured = tuple(config.get("pipelines") or ())
    if not configured:
        raise ValueError("three-pipeline configuration must list pipelines")
    unknown = set(configured) - set(PIPELINES)
    if unknown:
        raise ValueError(f"unknown configured pipelines: {sorted(unknown)}")
    contracts = config.get("pipeline_contract") or {}
    for name in configured:
        spec = PIPELINES[name]
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
    ppo_gamma = float((config.get("ppo") or {}).get("gamma", .99))
    design_gamma = float((config.get("rgat_design") or {}).get(
        "outcome_discount", ppo_gamma))
    if abs(ppo_gamma - design_gamma) > 1e-12:
        raise ValueError("R-GAT design/PBRS gamma must equal PPO gamma")
