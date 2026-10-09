# Sensor-realistic migration audit matrix

Current pins (2026-10-09 KST): source `608225805fa447c2f8e2756e33378c32fd975797`,
destination baseline `3fc2a9fa659f0f933706b5da6918aafec6239a20`. The source was checked out
read-only from its remote HEAD. The destination worktree was clean before this refactor.

| Component | Source behavior | Current destination behavior | Evidence | Decision / gate |
|---|---|---|---|---|
| Entry point | MATLAB direct PPO experiment | Bare `run.sh` owns the minimal-observation system; MATLAB port is explicit | `run.sh`, `test_run_sh_system_route.py` | Retain existing primary route; PASSED |
| Backend dispatch | One MATLAB simulator path | Replay, local diagnostic, and actual Isaac/PX4 are selected by the execution contract | `direct_policy/backends.py`, `test_live_isaac_adapter.py` | Removed implicit local fallback; PASSED |
| Planar observation | Fixed causal 12D vector | Fixed 12D vector and named graph mapping | `contracts.py`, `observation.py`, `tests/fixtures/matlab_port_golden_6082258.json` | Retain; numerical vector/graph parity PASSED |
| Spatial observation | Optional source path is not the final migration contract | Fixed 21D provisional estimate vector | `SPATIAL_FIELDS`, `spatial_vector` | Replace/extend; FAILED final P3 gate |
| Graph | Seven nodes, four relations, separate actor/critic encoders | Same planar graph; 3D adds feature channels but no target-attitude/uncertainty nodes | `graph.py`, `models.py` | Retain planar; extend spatial; FAILED final P3 gate |
| PPO | Latent Gaussian, tanh/scale, clipped actor/value updates | End-to-end actor/critic encoders, variable-time GAE discount, checkpoint optimizer resume | `ppo.py`, `runner.py` | Extend numerical parity and KL diagnostics; partial implementation |
| Camera and pad estimate | Marker PnP + causal CV estimator | Existing rendered ArUco/PnP/EKF stream is consumed by live adapter | `marker_vision.py`, `isaac_adapter.py` | Retain integration; pad attitude/ambiguity covariance still missing |
| Own navigation | Simulator-specific planar navigation | Existing PX4 EKF own pose/velocity stream | `spatial/environment.py`, `CausalIsaacObservation` | Retain; covariance provenance not exposed to policy |
| Action | Planar `[a_x,a_z]` | Planar embedding or spatial `[a_x,a_y,a_z]`, acceleration-only PX4 fields | `isaac_adapter.py`, `DirectIsaacBackend` | Retain; coupled limit is applied by shared controller |
| Safety attribution | Direct source execution | Direct profile plus separately versioned shield profile | contract hashes and authority trace | Retain separation; policy/applied/measured signals recorded |
| Isaac lifecycle | Not applicable | Owned stack, reset, command ownership, cleanup through existing spatial backend | `run_spatial_pipeline.live_stack`, `DirectIsaacBackend` | Current pin NOT_RUN; historical run is not re-labelled |
| Training | Direct PPO | Local and opt-in actual-Isaac update paths; resume restores model/optimizer/update/RNG | `run_train`, `--resume-checkpoint-root` | Current actual-Isaac training NOT_RUN |

## Fidelity matrix

| Effect | Status | Evidence / limitation |
|---|---|---|
| Intrinsics, extrinsics, board geometry | Implemented | deployment profile + rendered ArUco detector |
| Frustum, image bounds, occlusion | Implemented in scored Isaac path | rendered camera; local backend is only a diagnostic |
| Corner/PnP noise | Approximate | profile/model assumptions; not hardware calibrated |
| Exposure and motion blur | Approximate | simulator rendering profile, no measured hardware fit |
| Optical latency and capture alignment | Implemented | capture/receive/decision provenance and capture-time own pose |
| Dropout and stale-frame handling | Implemented | sample id, age, reset-scoped observer |
| Pad orientation/angular motion estimate | Missing | current direct-policy measurement exposes position only |
| Own-navigation covariance in policy input | Missing | validity/age exist; uncertainty does not |
| Rotor/contact physics | Implemented only on actual Isaac backend | local reference backend is not acceptance evidence |
| Turning/tilting pad final scenario | Missing from direct-policy acceptance campaign | required before final spatial claim |

The 21D spatial schema is therefore explicitly provisional. It is not described as completion
of the attached migration brief, and historical one-seed results are not evidence for the new pin.
