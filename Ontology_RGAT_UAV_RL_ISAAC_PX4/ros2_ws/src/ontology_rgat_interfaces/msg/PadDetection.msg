# Output of the ArUco pad detector: one PnP solve of the registered board.
#
# header.stamp     image CAPTURE time, not publish time (transport delay stays visible)
# header.frame_id  vehicle body FLU frame ("base_link"); the camera mount
#                  offset and optical->body rotation are already applied
#
# Only position enters the observation vector (after rotation into ENU with the
# EKF attitude at capture time). marker_count and reprojection_error_px are
# diagnostics for logging, never features.
std_msgs/Header header

bool detected                       # a board pose was solved from at least one registered marker
geometry_msgs/Point position        # board origin in body FLU [m]; zero when not detected

uint8 marker_count                  # markers that braced the solve
float32 reprojection_error_px       # mean reprojection error of the accepted solve
