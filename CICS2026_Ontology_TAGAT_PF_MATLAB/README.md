# Schema·Ontology-Guided TA-GAT Adaptive Particle Sampling for Reliable UGV Localization

MATLAB research prototype for comparing six UGV localization baselines on the open NCLT dataset.

## Core research pipeline

Raw NCLT sensors
→ Schema standardization
→ Validated ontology relation graph
→ GAT / R-GAT / temporal-adaptive R-GAT
→ reliability & localization-difficulty estimation
→ bounded soft prior
→ particle-filter proposal mixture / covariance / particle budget
→ pose + uncertainty

The ontology does **not** directly output the final pose or particle count. The learned graph model estimates time-varying relation importance and reliability. The PF remains the final probabilistic state estimator.

## Dataset

Default dataset: **University of Michigan NCLT (North Campus Long-Term Dataset)**, session `2013-01-10`.

The default session was chosen because it is one of the smallest NCLT sessions while still providing:
- consumer GPS / RTK GPS
- Microstrain IMU
- wheel/6-DOF odometry
- Hokuyo UTM-30LX 2-D LiDAR
- SLAM-based ground truth

Official NCLT page:
https://robots.engin.umich.edu/nclt/

Paper:
N. Carlevaris-Bianco, A. K. Ushani, R. M. Eustice,
"University of Michigan North Campus Long-Term Vision and Lidar Dataset,"
International Journal of Robotics Research, 2016.

NCLT is distributed under the Open Database License / Database Contents License.
Do not redistribute NCLT-derived data without following its license requirements.

### Data limitation that matters for this project

NCLT consumer GPS provides fix mode, satellite count, latitude, longitude, altitude, track, and speed.
It does **not** provide the C/N0 and HDOP fields used in the separate GNSS-fault-detection study.
Therefore this project deliberately uses only measurements actually supported by NCLT:
`fixMode`, `numSV`, `gpsInnovation`, `gpsSpeedResidual`, LiDAR ICP quality, odometry covariance, and motion dynamics.

## Baselines

1. `PF-Fixed`
   - fixed particle count
   - fixed process / measurement covariance
   - bootstrap motion proposal

2. `PF-Adaptive`
   - same fixed sensor model
   - ESS + KLD based adaptive particle count

3. `Ontology-Rule-PF`
   - schema features + interpretable smooth rules
   - bounded reliability-driven proposal mixture / covariance / particle budget

4. `GAT-PF`
   - fully connected untyped graph
   - single-time-step graph attention
   - learned reliability / difficulty

5. `Ontology-RGAT-PF`
   - ontology-constrained graph topology
   - typed relations
   - relation-specific transformations and attention
   - single-time-step input

6. `Ontology-TAGAT-PF` (proposed)
   - same ontology relation graph
   - sliding-window temporal recurrence
   - temporal-change term in edge attention
   - time-varying relation weights
   - learned reliability / difficulty
   - bounded PF prior

## Important TA-GAT note

The implementation here is a **research adaptation for this UGV idea**, not an exact reproduction of the photovoltaic TA-GAT paper.
The cited paper uses eDMD / Lifted-Koopman operators and window-wise graph adaptation for PV forecasting.
This project intentionally keeps only the transferable concept:
**window-wise time-varying graph attention / relation reweighting**.
It does not include the paper's eDMD or Lifted-Koopman blocks.

Reference:
Z. Wang, D. Ma, Q. Wang, X. Zuo, K. Qi,
"TA-GAT: Temporal-adaptive graph attention network for multi-station photovoltaic power forecasting,"
Solar Energy, 314, 114703, 2026.
DOI: 10.1016/j.solener.2026.114703

## MATLAB requirements

Recommended:
- MATLAB R2025b
- Deep Learning Toolbox (required for GAT / R-GAT / TA-GAT training)
- No Mapping Toolbox is required for GPS conversion.
- No Navigation Toolbox is required for the included custom 2-D ICP / PF implementation.

The NCLT parser and PF are written with base MATLAB functions.
The learned graph models use `dlarray`, `dlgradient`, and `dlfeval`.

## Quick start

### A. Real NCLT experiment

```matlab
run_nclt_experiment
```

On first run, the code downloads the selected NCLT session from the official University of Michigan server:
- sensor archive
- Hokuyo archive
- ground-truth CSV

Default raw download is roughly tens of MB, not the multi-GB Velodyne stream.

To disable automatic downloading:

```matlab
cfg = defaultConfig(pwd);
cfg.data.autoDownload = false;
```

### B. Fast synthetic smoke test

```matlab
run_synthetic_demo
```

The synthetic mode is only for validating the entire software pipeline.
Research results should use NCLT.

## Main output directory

Each experiment creates:

```text
results/run_YYYYMMDD_HHMMSS/
    config.json
    console.log
    metrics.csv
    dataset_summary.csv
    training_GAT.csv
    training_RGAT.csv
    training_TAGAT.csv
    model_GAT.mat
    model_RGAT.mat
    model_TAGAT.mat
    predictions.mat
    PF-Fixed/
        trajectory.csv
        step_log.csv
        result.mat
    ...
    Ontology-TAGAT-PF/
        trajectory.csv
        step_log.csv
        result.mat
    attention_GAT.csv
    attention_RGAT.csv
    attention_TAGAT.csv
    final_trajectory.png
    final_position_error.png
    final_accuracy_summary.png
    final_particles_runtime.png
    final_reliability.png
    final_TAGAT_attention.png
    all_results.mat
```

## Reproducibility and split

Default:
- first 55%: training
- next 15%: validation
- last 30%: localization test

For a paper, use **cross-session evaluation**:
train graph model on one or more NCLT sessions and test on a different session.
The code is modular so the data adapter can be extended to multiple sessions.

## PF mixture proposal

The ontology / learned variants use a mixture of:

```text
motion proposal
GNSS-centered proposal
LiDAR-odometry-centered proposal
broad recovery proposal
```

The proposal weights are bounded by a non-zero exploration floor.
Sensor covariance is also clipped between lower / upper limits.
This prevents an incorrect ontology or network output from collapsing PF exploration.

## Recommended ablation

Use the generated `metrics.csv` to report:
- Position RMSE / MAE / P95
- Yaw RMSE
- failure rate (> configured threshold)
- mean NEES
- mean particle count
- mean step runtime
- real-time factor

The model sequence isolates:
- adaptive PF effect
- ontology/rule effect
- graph-attention effect
- relation-type effect
- temporal-adaptation effect

## Source tree

```text
config/       experiment configuration
data/         NCLT download, parsing, synchronization, synthetic fallback
schema/       standardized feature construction
ontology/     ontology graph and rule baseline
lidar/        Hokuyo parsing and custom scan-to-scan ICP
learning/     GAT / R-GAT / TA-GAT training and inference
pf/           particle-filter implementation
evaluation/   metrics and result saving
viz/          live and final visualization
utils/        SE(2), interpolation, math helpers
tests/        lightweight smoke tests
```
