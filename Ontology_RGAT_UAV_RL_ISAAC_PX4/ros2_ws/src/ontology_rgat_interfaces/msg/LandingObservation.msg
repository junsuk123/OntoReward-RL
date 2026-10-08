# Minimal observation o_t: pad perception + own localization, nothing inferred.
#
# header.stamp     decision time t
# header.frame_id  world ENU frame ("map")
#
# Physical units. Normalization and every judgement (pad velocity, visibility,
# pad-loss class, readiness, inhibit) belong to the ontology node; the safety
# supervisor reads this same message and nothing richer.
std_msgs/Header header
string schema_id                    # "minimal-landing-obs/2"

# 1. own localization: IMU + GNSS fused by the EKF (PX4 EKF2), NOT a raw GNSS fix
geometry_msgs/Point own_position    # ENU [m]
geometry_msgs/Vector3 own_velocity  # ENU [m/s], EKF velocity state (not differenced)
bool own_valid                      # EKF reports a valid horizontal and vertical solution
float32 own_age_s                   # t - stamp of the EKF solution used

# 2. pad perception: camera -> ArUco -> PnP, rotated into ENU axes at capture time
geometry_msgs/Vector3 pad_relative_position  # pad minus vehicle body, ENU axes [m];
                                             # last detected value is HELD when not detected
bool pad_detected                   # detected in the most recent processed frame
float32 pad_age_s                   # t - capture time of the last detection (capped)
builtin_interfaces/Time pad_capture_stamp

# Fixed-order copy, the single vector every arm reads:
# [own_x, own_y, own_z, own_vx, own_vy, own_vz,
#  rel_x, rel_y, rel_z, own_valid, own_age_s, pad_detected, pad_age_s]
float32[13] vector
