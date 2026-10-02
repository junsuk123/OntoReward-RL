# Repository execution contract

The primary research configuration is
`Ontology_RGAT_UAV_RL_ISAAC_PX4/config/experiments/two_axis_context_rgat_comparison.yaml`.

- Keep the learned action at two channels: longitudinal and vertical net
  acceleration. Pitch is a derived physical state/setpoint, never a third
  policy action.
- Keep `physical_state`, measurements, causal estimator memory, decision
  context, safety status, and episode status separate.
- All three primary arms must use the same causal packet registry hash,
  environment, reward, supervisor, timing, scenarios, and sensor events.
- Simulator truth is allowed only in the isolated training reward and
  evaluator. It must not enter an actor, critic, graph, estimator, or safety
  supervisor input.
- Keep legacy three-channel, PBRS, FOV-reward, adaptive-reward, behavior
  cloning, and frozen-selective experiments under their existing IDs. Never
  alias their checkpoints to `two-axis-context-rgat-v1`.
- Do not run long PPO training, real-vehicle commands, or push changes without
  explicit user approval. Bounded local tests and smoke rollouts are allowed.
- Preserve the external three-field PX4 gateway only through
  `two_axis.adapter.px4_gateway_command`; its third field is derived from the
  two-axis request and is not learned.
