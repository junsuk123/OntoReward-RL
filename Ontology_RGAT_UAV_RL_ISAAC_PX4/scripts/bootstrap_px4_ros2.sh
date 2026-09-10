#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ascii_ros2_ws=${ASCII_ROS2_WS:-/home/${USER}/.local/share/ontology_rgat_uav_rl/ros2_ws}
px4_version=${PX4_VERSION:-v1.14.3}
px4_msgs_branch=${PX4_MSGS_BRANCH:-release/1.14}
xrce_version=${XRCE_AGENT_VERSION:-v2.4.2}

if [[ ${1:-} == "--help" ]]; then
  printf '%s\n' \
    "Usage: $0" \
    "Environment overrides:" \
    "  PX4_VERSION=$px4_version" \
    "  PX4_MSGS_BRANCH=$px4_msgs_branch (must match PX4)" \
    "  XRCE_AGENT_VERSION=$xrce_version" \
    "  ROS_DISTRO=humble" \
    "  ASCII_ROS2_WS=$ascii_ros2_ws (ROS 2 Humble non-ASCII path workaround)"
  exit 0
fi

set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
set -u
mkdir -p "$workspace_root/external" "$workspace_root/ros2_ws/src"
python3 -m pip install --user -r "$workspace_root/requirements-dev.txt"

if [[ ! -d "$workspace_root/external/PX4-Autopilot/.git" ]]; then
  git clone --branch "$px4_version" --depth 1 \
    https://github.com/PX4/PX4-Autopilot.git \
    "$workspace_root/external/PX4-Autopilot"
fi

# PX4 v1.14 has a legacy pip spec (>=3.0.*) rejected by current pip.
if grep -q 'matplotlib>=3.0\.\*' \
  "$workspace_root/external/PX4-Autopilot/Tools/setup/requirements.txt"; then
  git -C "$workspace_root/external/PX4-Autopilot" apply \
    "$workspace_root/patches/px4-v1.14-python-requirements.patch"
fi

python3 -m pip install --user \
  --constraint "$workspace_root/requirements-px4-constraints.txt" \
  --requirement "$workspace_root/external/PX4-Autopilot/Tools/setup/requirements.txt"

# PX4 v1.14 publishes neither the estimator diagnostics used to verify
# GNSS/DR switching nor vehicle_land_detected, vehicle_command_ack,
# vehicle_thrust_setpoint and battery_status over uXRCE-DDS. Without the land
# detector the gateway cannot tell a hovering vehicle from a landed one, so
# touchdown is never detected; without the thrust setpoint it cannot price the
# energy a landing costs.
#
# The guard names the newest topic and restores the file before patching, so a
# tree carrying an earlier version of this patch is brought up to date instead
# of being skipped because an older topic is already present.
dds_topics=src/modules/uxrce_dds_client/dds_topics.yaml
if ! grep -q 'estimator_status_flags' \
  "$workspace_root/external/PX4-Autopilot/$dds_topics"; then
  git -C "$workspace_root/external/PX4-Autopilot" checkout -- "$dds_topics"
  git -C "$workspace_root/external/PX4-Autopilot" apply \
    "$workspace_root/patches/px4-v1.14-publish-land-detected.patch"
fi

# The client's topic table is generated from that yaml at build time, so a
# binary older than the yaml silently publishes the old topic set. Checking the
# timestamps rather than only the patch branch also catches a yaml edited or
# patched by hand, which is otherwise indistinguishable from an up-to-date tree.
px4_binary="$workspace_root/external/PX4-Autopilot/build/px4_sitl_default/bin/px4"
if [[ -x "$px4_binary" \
  && "$workspace_root/external/PX4-Autopilot/$dds_topics" -nt "$px4_binary" ]]; then
  printf '%s is newer than the built px4; forcing a rebuild.\n' "$dds_topics"
  rm -rf "$workspace_root/external/PX4-Autopilot/build/px4_sitl_default"
fi

if [[ ! -d "$workspace_root/ros2_ws/src/px4_msgs/.git" ]]; then
  git clone --branch "$px4_msgs_branch" \
    https://github.com/PX4/px4_msgs.git \
    "$workspace_root/ros2_ws/src/px4_msgs"
fi

if [[ ! -d "$workspace_root/external/Micro-XRCE-DDS-Agent/.git" ]]; then
  git clone --branch "$xrce_version" --recursive \
    https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
    "$workspace_root/external/Micro-XRCE-DDS-Agent"
fi

# Agent v2.4.2 references the now-deleted moving Fast-DDS branch 2.12.x.
# Pin the final upstream 2.12 release so a clean bootstrap remains reproducible.
if grep -q 'set(_fastdds_tag 2.12.x)' \
  "$workspace_root/external/Micro-XRCE-DDS-Agent/CMakeLists.txt"; then
  git -C "$workspace_root/external/Micro-XRCE-DDS-Agent" apply \
    "$workspace_root/patches/micro-xrce-dds-agent-fastdds-v2.12.2.patch"
fi

cmake -S "$workspace_root/external/Micro-XRCE-DDS-Agent" \
  -B "$workspace_root/external/Micro-XRCE-DDS-Agent/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$workspace_root/external/install"
cmake --build "$workspace_root/external/Micro-XRCE-DDS-Agent/build" --parallel
cmake --install "$workspace_root/external/Micro-XRCE-DDS-Agent/build"

# Always ask Ninja for an incremental build. The DDS bridge is generated from
# dds_topics.yaml; checking only for an existing px4 binary leaves that generated
# table stale after this workspace adds or changes a published topic.
make -C "$workspace_root/external/PX4-Autopilot" px4_sitl_default

# Humble's rosidl dependency parser corrupts non-ASCII source/build paths.
# Mirror only the ROS packages into a stable ASCII-only runtime workspace.
mkdir -p "$ascii_ros2_ws/src"
cmake -E copy_directory \
  "$workspace_root/ros2_ws/src/px4_msgs" "$ascii_ros2_ws/src/px4_msgs"
cmake -E copy_directory \
  "$workspace_root/ros2_ws/src/ontology_rgat_px4" \
  "$ascii_ros2_ws/src/ontology_rgat_px4"

cd "$ascii_ros2_ws"
set +u
colcon build --symlink-install
set -u
printf 'Bootstrap complete. Source %s\n' "$ascii_ros2_ws/install/local_setup.bash"
