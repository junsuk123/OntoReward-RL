#!/usr/bin/env bash
# Source this -- do not run it -- to get a shell that can see the stack.
#
#   source scripts/ros_env.sh
#   ros2 topic list
#
# The XRCE-DDS agent speaks Fast DDS, so every consumer of /fmu/* and of the
# simulator's topics has to as well. A shell whose profile selects a different
# RMW does not merely read empty messages: two DDS implementations never
# discover each other at all, so `ros2 topic list` comes back with nothing but
# /parameter_events and /rosout and the stack looks dead when it is fine.
#
# This is exactly what ~/.bashrc does on this machine -- it exports
# rmw_cyclonedds_cpp -- which is why an interactive shell needs this and the
# run scripts, which set it themselves, do not.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  printf 'Source this file, do not execute it:\n\n  source %s\n\n' "${BASH_SOURCE[0]}" >&2
  exit 1
fi

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset CYCLONEDDS_URI

_ros_env_ws=${ASCII_ROS2_WS:-$HOME/.local/share/ontology_rgat_uav_rl/ros2_ws}
set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
if [ -f "$_ros_env_ws/install/local_setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$_ros_env_ws/install/local_setup.bash"
fi
set -u 2>/dev/null || true
unset _ros_env_ws

printf 'RMW_IMPLEMENTATION=%s, CYCLONEDDS_URI cleared. This shell can see the stack.\n' \
  "$RMW_IMPLEMENTATION"
