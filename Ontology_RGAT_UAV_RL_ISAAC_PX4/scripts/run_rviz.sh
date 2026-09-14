#!/usr/bin/env bash
# The live landing view. Start this beside the learner; the learner publishes
# /landing_rl/* and the simulator publishes the deck pose and the marker solve.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ascii_ws=${ASCII_ROS2_WS:-$HOME/.local/share/ontology_rgat_uav_rl/ros2_ws}
layout="$workspace_root/rviz/ontology_rgat.rviz"
rviz_arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel-pairs)
      [[ $# -ge 2 ]] || { echo "ERROR: --parallel-pairs needs a value" >&2; exit 2; }
      [[ "$2" =~ ^[12]$ ]] || {
        echo "ERROR: --parallel-pairs must be 1 or 2 for the primary comparison" >&2; exit 2;
      }
      if (( $2 > 1 )); then
        layout="$workspace_root/rviz/ontology_rgat_parallel.rviz"
      fi
      shift 2
      ;;
    *) rviz_arguments+=("$1"); shift ;;
  esac
done

# Everything that touches /fmu/* must speak Fast DDS, because that is what the
# XRCE-DDS agent speaks. A shell with a different default sees the topics but
# reads nothing from them.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset CYCLONEDDS_URI

# Launched from a snap-packaged terminal -- VS Code's integrated one, which is
# where this pipeline usually gets started -- the shell carries the snap's own
# GTK and locale paths. RViz is a Qt/GTK application, so it follows them into
# /snap/core20 and dies on `undefined symbol: __libc_pthread_init` before it
# ever opens a window. None of these belong to a system binary, so drop them.
unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE GTK_MODULES LOCPATH \
      GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR GIO_MODULE_DIR \
      GSETTINGS_SCHEMA_DIR GTK_DATA_PREFIX
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
    | grep -v '^/snap/' | paste -sd: -)
  export LD_LIBRARY_PATH
fi

# ROS's setup.bash reads variables it has not set, which is fatal under
# `set -u`: the script dies here and never reaches the exec below, while
# whatever launched it sees a process that started and exited cleanly.
# shellcheck disable=SC1091
set +u
source /opt/ros/humble/setup.bash
set -u
if [ -f "$ascii_ws/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  set +u
  source "$ascii_ws/install/setup.bash"
  set -u
fi

exec rviz2 -d "$layout" "${rviz_arguments[@]}"
