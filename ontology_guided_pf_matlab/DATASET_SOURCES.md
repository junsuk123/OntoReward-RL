# Dataset sources / realism basis

The default demo uses a synthetic UrbanNav-like CSV so that the MATLAB experiment runs immediately without multi-GB downloads or RINEX preprocessing.

## 1) UrbanNav Dataset (primary realism reference)
- Official GitHub: https://github.com/IPNL-POLYU/UrbanNavDataset
- Tokyo setup: u-blox F9P GNSS 5 Hz, Trimble NetR9 10 Hz, Tamagawa TAG264 IMU 50 Hz, Velodyne VLP-32C LiDAR 10 Hz, Applanix POS LV620 ground truth 10 Hz.
- Hong Kong setup: low-cost GNSS, multiple LiDARs, camera, Xsens IMU and SPAN-CPT ground truth in dense urban canyons.
- Known challenge: NLOS/multipath in high-rise urban canyons.

## 2) Urban vehicle GNSS/IMU dataset (alternative smaller real dataset)
- GitHub: https://github.com/LuckydogZY/urban_vehicle_database
- Repository describes triple-frequency multi-GNSS at 5 Hz, high-rate IMU at 100 Hz and reference ground truth; download is linked externally from the repository.

## Why synthetic data is bundled
The official UrbanNav Tokyo package is multi-GB and its GNSS measurements are provided mainly as RINEX, while LiDAR is in rosbag. A fair end-to-end MATLAB use therefore requires additional RTKLIB/RINEX and ROS preprocessing. To make the proposed PF experiment reproducible with one command, the bundled CSV directly contains:

- local ground truth pose,
- odometry/IMU-like controls,
- GNSS position + numSV/HDOP/CN0/hAcc quality schema,
- LiDAR pseudo map-matching pose + matching-score / feature-count schema,
- semantic environment context.

The synthetic generator correlates NLOS jumps with degraded GNSS quality indicators and LiDAR registration outliers with degraded matching scores. This is important because the proposed ontology layer must infer reliability from observable evidence rather than from hidden ground-truth error.
