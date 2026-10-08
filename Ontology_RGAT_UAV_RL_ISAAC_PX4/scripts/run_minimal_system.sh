#!/usr/bin/env bash
# The whole minimal-observation system in one command (bare ./run.sh).
#
#   preflight -> ROS packages built -> teacher -> BC -> PPO -> 48 held-out
#   -> stress scenarios -> Isaac/PX4 flights (in-process and ROS-chain control)
#
# Design and every number: docs/MINIMAL_OBSERVATION_ROS_PIPELINE_KO.md.
#
# Options (anything else is passed to tools/minimal_full_pipeline.py):
#   --no-isaac            stop after the local stages; never touch a stack
#   --nominal-training    train on the nominal contract instead of the
#                         per-episode randomized stress ("dr", the default:
#                         doc section 15.3)
#   --run-root PATH       default results/minimal_system_<ontology schema>,
#                         so a rerun RESUMES and a contract change starts fresh
#   --dry-run             print the plan; build nothing, start nothing
#
# Isaac/PX4 runs only when ISAACSIM_PATH/python.sh, ROS 2 Humble and the
# ASCII runtime workspace with px4_msgs are all present, and never when an
# Isaac is already running (it is skipped with the reason, not taken over).
set -Eeuo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$here/.." && pwd)
runtime_root=${MINIMAL_RUNTIME_ROOT:-$HOME/.local/share/ontology_rgat_uav_rl}
ascii_ws=${ASCII_ROS2_WS:-$runtime_root/ros2_ws}
minimal_ws=$runtime_root/minimal_ws
export PYTHONUNBUFFERED=1

isaac=1; dry_run=0; training=(--train-scenario dr); run_root=""; passthrough=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-isaac) isaac=0 ;;
    --nominal-training) training=() ;;
    --dry-run) dry_run=1 ;;
    --run-root) run_root=$2; shift ;;
    --run-root=*) run_root=${1#--run-root=} ;;
    *) passthrough+=("$1") ;;
  esac
  shift
done

say() { printf '[run.sh %(%H:%M:%S)T] %s\n' -1 "$*"; }

schema=$(cd "$project_root/python" && python3 -c \
  'from ontology_rgat.minimal import ONTOLOGY_SCHEMA_ID as s; print(s.replace("minimal-landing-", "").replace("/", "-"))')
run_root=${run_root:-$project_root/results/minimal_system_$schema}

# --- preflight: python ------------------------------------------------------
missing=$(python3 - <<'EOF'
import importlib
print(" ".join(m for m in ("numpy", "scipy", "torch", "cv2", "yaml")
               if importlib.util.find_spec(m) is None))
EOF
)
if [ -n "$missing" ]; then
  echo "Missing Python modules: $missing (pip install -r $project_root/requirements.txt)" >&2
  exit 1
fi

# --- preflight: what the flight stage needs ---------------------------------
isaac_reason=""
if [ "$isaac" = 1 ]; then
  if [ ! -f /opt/ros/humble/setup.bash ]; then isaac_reason="ROS 2 Humble not found"
  elif [ ! -f "$ascii_ws/install/local_setup.bash" ]; then
    isaac_reason="ASCII runtime workspace $ascii_ws not built (scripts/bootstrap_px4_ros2.sh)"
  elif [ -z "${ISAACSIM_PATH:-}" ] || [ ! -x "$ISAACSIM_PATH/python.sh" ]; then
    isaac_reason="ISAACSIM_PATH/python.sh not found"
  else
    for entry in /proc/[0-9]*; do
      tokens=$( { tr '\0' ' ' < "$entry/cmdline"; } 2>/dev/null || true)
      case "$tokens" in *landing_world.py*)
        case "$tokens" in *bash\ -c*) ;; *) isaac_reason="an Isaac stack is already running (not taken over)";; esac ;;
      esac
    done
  fi
  [ -n "$isaac_reason" ] && isaac=0
fi

say "run root : $run_root"
say "training : ${training[*]:-nominal}"
if [ "$isaac" = 1 ]; then say "Isaac/PX4: ON (will start an owned stack after the local stages)"
else say "Isaac/PX4: off${isaac_reason:+ -- $isaac_reason}"; fi

# Five training seeds unless told otherwise (doc sections 14-15 are five-seed).
seeds=(--seeds 828 829 830 831 832)
for arg in "${passthrough[@]}"; do [ "$arg" = "--seeds" ] && seeds=(); done
pipeline=(python3 "$project_root/tools/minimal_full_pipeline.py" --run-root "$run_root"
          "${seeds[@]}" "${training[@]}" "${passthrough[@]}")
[ "$isaac" = 1 ] && pipeline+=(--isaac)

if [ "$dry_run" = 1 ]; then
  say "dry run: nothing is built, trained or flown"
  exec "${pipeline[@]}" --dry-run
fi

# --- ROS packages: mirrored into an ASCII workspace (rosidl corrupts
# --- non-ASCII paths; same convention as scripts/sync_gateway.sh) ----------
if [ -f /opt/ros/humble/setup.bash ]; then
  say "building ontology_rgat_interfaces + ontology_rgat_landing in $minimal_ws"
  mkdir -p "$minimal_ws/src"
  for package in ontology_rgat_interfaces ontology_rgat_landing; do
    rsync -a --delete --exclude __pycache__ "$project_root/ros2_ws/src/$package/" "$minimal_ws/src/$package/"
  done
  (
    set +u
    source /opt/ros/humble/setup.bash
    [ -f "$ascii_ws/install/local_setup.bash" ] && source "$ascii_ws/install/local_setup.bash"
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; unset CYCLONEDDS_URI
    cd "$minimal_ws"
    colcon build --packages-select ontology_rgat_interfaces ontology_rgat_landing \
      > "$minimal_ws/build.log" 2>&1 || { tail -20 "$minimal_ws/build.log" >&2; exit 1; }
  )
  export MINIMAL_ROS_INSTALL=$minimal_ws/install
  say "ROS packages built ($MINIMAL_ROS_INSTALL)"
else
  say "ROS 2 Humble not found: ROS nodes not built (local stages only)"
fi

export ONTOLOGY_RGAT_ROOT=$project_root
say "starting pipeline: ${pipeline[*]}"
exec "${pipeline[@]}"
