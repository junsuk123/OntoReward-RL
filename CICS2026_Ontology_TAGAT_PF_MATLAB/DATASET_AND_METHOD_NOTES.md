# Dataset and method notes

## Why NCLT

NCLT is especially suitable for this UGV localization concept because the robot platform is a Segway UGV and the dataset contains synchronized navigation and perception sensors across repeated campus traversals.

The official paper reports:
- 27 sessions over 15 months
- Velodyne HDL-32E
- Hokuyo UTM-30LX / URG-04LX
- Microstrain IMU
- wheel odometry
- standard GPS and RTK GPS
- global SLAM ground truth

The ground truth is built from a large cross-session SLAM graph using LiDAR scan-matching and high-accuracy RTK GPS; odometry is used for interpolation.

The default `2013-01-10` session is relatively compact on the official download table:
- sensor archive about 21 MB
- Hokuyo archive about 26 MB
- ground-truth CSV about 20 MB

The code intentionally avoids the multi-GB Velodyne archive for the default experiment.

## NCLT coordinate conversion used in code

The official NCLT paper defines the local GPS linearization around:
- latitude origin: 42.293227 deg
- longitude origin: -83.709657 deg
- altitude origin: 270 m
- equatorial radius: 6,378,135 m
- polar radius: 6,356,750 m

`gpsToLocalNCLT.m` implements the paper's local x/y/z equations directly.

## Hokuyo parser

The official NCLT devkit's `read_hokuyo_30m.py` uses:
- uint64 little-endian timestamp
- 1081 uint16 ranges
- angle -135 to +135 deg at 0.25 deg steps
- scale 0.005 m and offset -100 m

The MATLAB parser follows that format.

## What is and is not "ontology reasoning" here

The proposed learned pipeline uses the ontology primarily to define a validated relation graph.
A full OWL reasoner is not embedded in the MATLAB runtime.

Recommended workflow:
1. design OWL ontology in Protégé
2. validate class / relation consistency offline with HermiT
3. export the validated topology / relation types
4. use that graph in MATLAB for R-GAT / TA-GAT
5. keep final pose estimation probabilistic in the PF

`validateOntologyGraph.m` only performs structural checks; it is not a replacement for a Description Logic reasoner.

## Learned reliability supervision

The graph models are trained with GT-derived reliability targets.
Ground truth is **never included in runtime input features**.

Targets:
- GNSS reliability: function of GNSS-to-GT error and availability
- LiDAR reliability: function of integrated LiDAR-odometry error plus ICP quality
- odometry reliability: function of odometry-to-GT error plus odometry covariance / dynamics
- localization difficulty: aggregate of the three sensor reliabilities plus dynamics

This enables a controlled baseline study before attempting fully end-to-end differentiable PF learning.

## Recommended publication-grade extension

For a paper:
- train GAT/R-GAT/TA-GAT on multiple NCLT sessions
- test on unseen sessions / seasons
- keep the ontology graph fixed
- report edge attention changes around GPS-degraded / LiDAR-feature-poor transitions
- compare with fixed PF, KLD adaptive PF, rule ontology PF
- repeat at multiple random seeds
- report confidence intervals
