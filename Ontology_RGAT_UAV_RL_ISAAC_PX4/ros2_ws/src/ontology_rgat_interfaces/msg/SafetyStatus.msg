# Safety supervisor state for one decision step. The supervisor reads only
# LandingObservation (and its own short history of it).
std_msgs/Header header              # stamp = decision time of the observation it acted on

uint8 MODE_NOMINAL = 0              # command passes (after saturation / handover limits)
uint8 MODE_DESCENT_HOLD = 1         # descent not authorized: vertical rate held, lateral passes
uint8 MODE_TERMINAL_COMMIT = 2      # pad lost in the blind zone: committed braked descent
uint8 MODE_ABORT_HOLD = 3           # own-EKF position hold, all axes overridden
uint8 mode

uint8 PAD_VISIBLE = 0
uint8 PAD_TRANSIENT_DROPOUT = 1     # brief miss, any altitude
uint8 PAD_TERMINAL_OCCLUSION = 2    # lost because the vehicle is on top of it
uint8 PAD_LOST = 3                  # left the frame / stale / drifted
uint8 pad_loss_class
uint8 pad_lost_reason               # bitmask, see docs/MINIMAL_OBSERVATION_ROS_PIPELINE_KO.md

geometry_msgs/Vector3 requested_acceleration   # policy output, ENU
geometry_msgs/Vector3 applied_acceleration     # what was sent on, ENU
bool[3] intervened                  # per axis x, y, z (per-axis PPO mask, not one bit)
