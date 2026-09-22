#!/usr/bin/env bash
# The live landing view. Start this beside the learner; the learner publishes
# /landing_rl/* and the simulator publishes the deck pose and the marker solve.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ascii_ws=${ASCII_ROS2_WS:-$HOME/.local/share/ontology_rgat_uav_rl/ros2_ws}
layout="$workspace_root/rviz/ontology_rgat.rviz"
rviz_arguments=()
# Arm names for the generated pair titles, in pair order. Only the arms that
# hold training pairs: a run may compare more arms than it trains.
arm_titles=()
pairs=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm-title)
      [[ $# -ge 2 ]] || { echo "ERROR: --arm-title needs a value" >&2; exit 2; }
      arm_titles+=(--arm-title "$2")
      shift 2
      ;;
    --parallel-pairs)
      [[ $# -ge 2 ]] || { echo "ERROR: --parallel-pairs needs a value" >&2; exit 2; }
      [[ "$2" =~ ^[1-8]$ ]] || {
        echo "ERROR: --parallel-pairs must be between 1 and 8" >&2; exit 2;
      }
      pairs="$2"
      shift 2
      ;;
    *) rviz_arguments+=("$1"); shift ;;
  esac
done

# The layout is chosen after every argument has been read, not while parsing:
# --arm-title may follow --parallel-pairs on the command line, and generating
# the layout mid-parse silently dropped the titles that came after it.
if (( pairs == 2 )); then
  layout="$workspace_root/rviz/ontology_rgat_parallel.rviz"
elif (( pairs > 2 )); then
  # The checked-in layout defines two pairs. Beyond that, clone its pair group
  # so every flying pair gets a camera and a trail instead of only the first
  # two being shown.
  layout="$(mktemp -t ontology_rgat_rviz_XXXXXX.rviz)"
  python3 "$workspace_root/scripts/make_rviz_layout.py" \
    --pairs "$pairs" --output "$layout" "${arm_titles[@]+"${arm_titles[@]}"}"
fi

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
