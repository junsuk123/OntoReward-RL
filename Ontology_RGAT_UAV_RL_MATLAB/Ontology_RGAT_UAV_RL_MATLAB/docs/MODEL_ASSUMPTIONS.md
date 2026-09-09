# Model assumptions, calibration items and limits

## Source-aligned items
- The research framing follows the uploaded paper: ontology-based meaning for drone state, wind state and landing-pad observation; derived wind risk, visual stability and alignment; and future extension to RL reward design.
- The default rigid-body mass/inertia are aligned to a commonly circulated SJTU-drone model (`m=1.477 kg`, `J=diag(0.1152,0.1152,0.218) kg m^2`).
- The maximum total force default is `30 N`, matching the current public `sjtu_drone` plugin documentation.

## Explicit engineering assumptions that must be calibrated for publication
- body dimensions and arm geometry,
- propeller thrust/torque coefficients `kT`, `kQ`, motor time constant,
- panel normal drag coefficient,
- surface roughness / skin-friction model,
- ground-effect approximation,
- sensor noise and marker detection model,
- turbulence spectrum/mode amplitudes and localized gust parameters,
- sigmoid weights used to encode semantic wind risk.

These are kept in `config/defaultConfig.m` so they can be replaced by CAD, bench-test, wind-tunnel, flight-log or URDF/SDF-derived values without rewriting algorithms.

## Important fidelity boundary
This project gives a high-detail **distributed panel aerodynamic rigid-body simulation**. It is not Navier-Stokes CFD and does not resolve propeller wake/body interaction, blade-element aerodynamics, dynamic stall, vortex shedding, rotor inflow coupling, structural flex, ESC electrical dynamics or full ground-contact mechanics. If those phenomena are required, couple MATLAB to a validated CFD/Simscape/FlightGear/Gazebo plant and keep the ontology/R-GAT/RL layers unchanged.

## Scientific-validation recommendation
Before claiming physical fidelity, identify parameters from real logs and validate at three levels:
1. component: thrust stand + motor step response,
2. subsystem: hover/attitude disturbance response and aerodynamic drag tests,
3. system: held-out flight trajectories under measured 3-D wind.


## Reward-model data assumption
The R-GAT potential is pretrained from safe/unsafe terminal outcomes of a perturbed expert controller. This is intentionally independent of the manual baseline reward, but it introduces behavior-policy bias. For a paper, test sensitivity to the behavior policy and add data from random, baseline-policy, and real-flight rollouts.
