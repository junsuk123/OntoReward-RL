#!/usr/bin/env bash
# Shared ROS 2 environment for every simlab process. Source, don't execute.
#
# Isaac Sim runs on Python 3.11 while ROS 2 Humble ships Python 3.10 modules.
# Sourcing setup.bash puts /opt/ros/humble/*/python3.10/... on PYTHONPATH, and
# Isaac's interpreter would then try to import a cp310 rclpy and crash. The
# C++ ROS 2 bridge only needs LD_LIBRARY_PATH / AMENT_PREFIX_PATH, so we keep
# those and drop the Python entries.

: "${ROS_DISTRO_SETUP:=/opt/ros/humble/setup.bash}"

if [[ -z "${ROS_DISTRO:-}" ]]; then
    # shellcheck disable=SC1090
    source "$ROS_DISTRO_SETUP"
fi

# Both sides must agree on the middleware, or topics silently never connect.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# Match the user's CycloneDDS setup (loopback only) on both sides; a mismatch
# here means discovery silently fails.
if [[ -z "${CYCLONEDDS_URI:-}" && -f "$HOME/cyclonedds.xml" ]]; then
    export CYCLONEDDS_URI="file://$HOME/cyclonedds.xml"
fi

# Drop every Python 3.10 entry -- /opt/ros and any sourced overlay workspace
# alike; all of them carry cp310 extension modules Isaac's 3.11 cannot load.
strip_ros_python_paths() {
    local cleaned="" part
    IFS=':' read -ra parts <<< "${PYTHONPATH:-}"
    for part in "${parts[@]}"; do
        [[ -z "$part" ]] && continue
        [[ "$part" == *"python3.10"* ]] && continue
        cleaned="${cleaned:+$cleaned:}$part"
    done
    printf '%s' "$cleaned"
}

# When this shell was started from a snap-confined app (e.g. the VS Code snap),
# GTK_PATH points at the snap's GTK modules. RViz2 then loads them, drags in
# /snap/core20's glibc and dies with:
#   symbol lookup error: .../libpthread.so.0: undefined symbol: __libc_pthread_init
# Dropping the snap entries is enough; the system GTK modules are found anyway.
strip_snap_gtk_path() {
    if [[ "${GTK_PATH:-}" == *"/snap/"* ]]; then
        local cleaned="" part
        IFS=':' read -ra parts <<< "$GTK_PATH"
        for part in "${parts[@]}"; do
            [[ -z "$part" || "$part" == /snap/* ]] && continue
            cleaned="${cleaned:+$cleaned:}$part"
        done
        if [[ -n "$cleaned" ]]; then
            export GTK_PATH="$cleaned"
        else
            unset GTK_PATH
        fi
    fi
}
