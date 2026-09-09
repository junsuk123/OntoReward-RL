#!/usr/bin/env bash
# Isaac Sim opens a file-change watch per extension directory. The default
# 65536 watch budget is already consumed here by the Synology cloud-drive
# daemon (~50k) and VS Code (~15k), so Kit logs ~2000 "errno=28 / No space
# left on device" errors at startup. Raising the budget removes them.
set -euo pipefail
echo "fs.inotify.max_user_watches = 524288" | sudo tee /etc/sysctl.d/99-isaacsim-inotify.conf
echo "fs.inotify.max_user_instances = 1024" | sudo tee -a /etc/sysctl.d/99-isaacsim-inotify.conf
sudo sysctl --system | grep -i max_user_watches || true
sysctl fs.inotify.max_user_watches
sysctl fs.inotify.max_user_instances
