#!/usr/bin/env bash
# Keep the gateway that actually runs in step with the one in this repository.
#
# ROS 2 Humble's rosidl dependency parser corrupts non-ASCII source and build
# paths, so scripts/bootstrap_px4_ros2.sh mirrors the ROS packages into an
# ASCII-only runtime workspace and builds there. `ros2 run` therefore executes a
# *copy*: editing ros2_ws/src changes nothing until the copy is refreshed, and
# colcon copies the sources again into its build tree, so there are two
# staging steps between an edit and the running process.
#
# None of that is visible from the learner, which imports the repository
# directly. A gateway left behind shows up as a protocol error minutes into a
# run -- "Gateway state is missing field gnss" -- after Isaac has booted and
# been restarted twice, which is a long way from the actual cause.
#
#   sync_gateway.sh            refresh the mirror and rebuild the package
#   sync_gateway.sh --check    report drift and exit non-zero, changing nothing
#
# The launchers run --check before starting a gateway. Full bootstrap still
# does the same mirror-and-build for every package; this is the incremental
# path, because nobody rebuilds PX4 to pick up a one-line change to a Python
# gateway, which is exactly how the copy falls behind.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd "$here/.." && pwd)
ros2_runtime=${ASCII_ROS2_WS:-/home/${USER}/.local/share/ontology_rgat_uav_rl/ros2_ws}
package=ontology_rgat_px4
source_dir="$workspace_root/ros2_ws/src/$package"
source_package="$source_dir/$package"

mode=sync
case "${1:-}" in
  --check) mode=check ;;
  "") ;;
  -h|--help)
    sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit 0 ;;
  *)
    printf 'usage: %s [--check]\n' "$(basename "${BASH_SOURCE[0]}")" >&2
    exit 2 ;;
esac

source_ros() {
  unset CYCLONEDDS_URI
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  set +u
  source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
  if [[ -f "$ros2_runtime/install/local_setup.bash" ]]; then
    source "$ros2_runtime/install/local_setup.bash"
  fi
  set -u
}

# The directory `ros2 run` would import, which is the only one worth comparing:
# the mirror can be current while the build tree it feeds is not, and asking
# Python is the one answer that does not assume a colcon layout.
resolve_live_package() {
  set +e
  python3 -c 'import pathlib, ontology_rgat_px4 as p
print(pathlib.Path(p.__file__).resolve().parent)' 2>/dev/null
  set -e
}

in_step() {
  # --exclude=__pycache__: the bytecode is regenerated per interpreter and
  # differing there means nothing.
  diff -rq --exclude=__pycache__ "$source_package" "$1" >/dev/null 2>&1
}

live_gateway_pid() {
  # No match is the normal case, and under pipefail it would otherwise take
  # the script down with it.
  pgrep -f "$package/[r]os2_gateway" 2>/dev/null | head -1 || true
}

# ------------------------------------------------------------------- check
if [[ $mode == check ]]; then
  source_ros
  live=$(resolve_live_package)
  if [[ -z $live ]]; then
    printf '%s is not built. Run scripts/bootstrap_px4_ros2.sh.\n' "$package" >&2
    exit 1
  fi
  # A workspace on an ASCII path can be built in place, in which case there is
  # no copy to fall behind.
  [[ $live == "$source_package" ]] && exit 0
  in_step "$live" && exit 0
  {
    printf 'The gateway that would run is not the one in this repository.\n\n'
    diff -rq --exclude=__pycache__ "$source_package" "$live" || true
    printf '\n  repository: %s\n  would run:  %s\n\n' "$source_package" "$live"
    printf 'A gateway from an older mirror starts cleanly and then fails the\n'
    printf 'protocol check mid-run. Refresh it first:\n\n  %s\n\n' \
      "$here/sync_gateway.sh"
  } >&2
  exit 1
fi

# -------------------------------------------------------------------- sync
if [[ ! -d $source_package ]]; then
  printf 'No gateway source at %s.\n' "$source_package" >&2
  exit 1
fi
pid=$(live_gateway_pid)
if [[ -n $pid ]]; then
  printf 'A gateway is running (pid %s). Rebuilding under it would leave the\n' "$pid" >&2
  printf 'old code in the live process, which is the confusion this script\n' >&2
  printf 'exists to remove. Stop the stack first.\n' >&2
  exit 1
fi

# Replace rather than overlay. cmake -E copy_directory never removes anything,
# so a module deleted from the repository would keep running from the mirror
# and the build tree long after it stopped existing here.
for stage in src build install; do
  target="$ros2_runtime/$stage/$package"
  # Belt and braces before an rm -rf built out of the environment.
  if [[ -d $target && $target == */$stage/$package ]]; then
    rm -rf "$target"
  fi
done
mkdir -p "$ros2_runtime/src"
cmake -E copy_directory "$source_dir" "$ros2_runtime/src/$package"

source_ros
cd "$ros2_runtime"
set +u
colcon build --symlink-install --packages-select "$package"
set -u

"$here/sync_gateway.sh" --check
printf 'Gateway in step with %s\n' "$source_package"
