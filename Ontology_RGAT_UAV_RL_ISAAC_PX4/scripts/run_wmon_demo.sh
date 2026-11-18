#!/usr/bin/env bash
# One command to fly a trained current-semantics policy and watch it:
#   RViz2 (rviz/wmon_demo.rviz) + two rqt_plot windows + tools/wmon_demo.py.
#
#   ./run.sh wmon-demo                                   # /4 WMON nav, rgat seed 10, local plant
#   ./run.sh wmon-demo --profile kalman --model 12/ontology_rgat_ppo --cell dropout_ood
#   ./run.sh wmon-demo --profile wmon --model 10/bond --cell combined_id --episodes 5
#   ./run.sh wmon-demo --backend isaac --allow-isaac      # owned Isaac Sim + PX4 SITL + DDS + gateway
#   ./run.sh wmon-demo --no-viewers ...                   # flight only (no RViz2 / rqt)
#
# All other options go to tools/wmon_demo.py (--help).  The viewers stay open after the
# flight; Ctrl+C closes everything this script started.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
viewers=1
args=()
for a in "$@"; do
  if [[ "$a" == "--no-viewers" ]]; then viewers=0; else args+=("$a"); fi
done
# shellcheck disable=SC1091
set +u; source "$project_root/scripts/ros_env.sh"; set -u   # Fast DDS, as the PX4 agent
# A terminal inside the VS Code snap exports snap GTK/GIO/locale paths; rviz2 then loads the
# snap's libpthread and dies (undefined symbol __libc_pthread_init).  Restore a host GUI env.
if [[ -n "${SNAP:-}" ]]; then
  unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR \
        GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR LOCPATH
  [[ -n "${XDG_DATA_DIRS_VSCODE_SNAP_ORIG:-}" ]] && export XDG_DATA_DIRS="$XDG_DATA_DIRS_VSCODE_SNAP_ORIG"
  [[ -n "${XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG:-}" ]] && export XDG_CONFIG_DIRS="$XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG"
fi
logs="$project_root/results/wmon_demo/logs"
mkdir -p "$logs"
pids=()
# each viewer runs in its own process group (setsid); Qt viewers ignore SIGTERM, so close
# them like Ctrl+C (SIGINT) and force whatever is left
cleanup() {
  for p in "${pids[@]:-}"; do kill -INT -- "-$p" 2>/dev/null || true; done
  sleep 2
  for p in "${pids[@]:-}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
}
rqt_plot_bin="$(ros2 pkg prefix rqt_plot)/lib/rqt_plot/rqt_plot"
trap cleanup EXIT INT TERM
if [[ $viewers -eq 1 ]]; then
  setsid rviz2 -d "$project_root/rviz/wmon_demo.rviz" > "$logs/rviz2.log" 2>&1 & pids+=($!)
  setsid "$rqt_plot_bin" /wmon_demo/rel_pos_error/data /wmon_demo/rel_vel_error/data \
           /wmon_demo/vision_age/data /wmon_demo/nav_age/data /wmon_demo/nav_valid/data \
           > "$logs/rqt_errors.log" 2>&1 & pids+=($!)
  setsid "$rqt_plot_bin" /wmon_demo/action/vector/x /wmon_demo/action/vector/y \
           /wmon_demo/action/vector/z /wmon_demo/wmon_confidence/vector/x \
           /wmon_demo/wmon_confidence/vector/y \
           > "$logs/rqt_action.log" 2>&1 & pids+=($!)
  sleep 3   # let the viewers subscribe before the first decision
fi
cd "$project_root"
python3 "$project_root/tools/wmon_demo.py" "${args[@]}"
if [[ $viewers -eq 1 ]]; then
  echo "wmon-demo: flight finished; RViz2 and rqt stay open -- Ctrl+C to close them."
  wait
fi
