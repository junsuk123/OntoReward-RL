"""Screen capture of ORL demo flights in the real operating environment (docs/figures).

``capture`` flies deterministic episodes of one trained checkpoint on an owned Isaac Sim +
PX4 SITL stack with the GUI on, next to two ROS 2 tools: ``image_view`` on the onboard
camera's annotated ArUco topic and RViz2 following the vehicle (TF from
``demo_tf_relay.py``).  It records the whole X screen with ffmpeg and writes
``timeline.json`` with the wall time of every reset, handover, decision and end.

``media`` cuts that recording into the README media: a still, a GIF of the first flight
and an MP4 of every flight, both at 4x with the flight phase written on each frame.

Demo only.  The seeds are demo seeds, nothing here is scored or evidence for anything,
and every Isaac flight still needs the user's approval and ``--allow-isaac``.

    python3 tools/isaac_demo_capture.py capture OUT CHECKPOINT --allow-isaac [--seeds 999902,999903,999904]
    python3 tools/isaac_demo_capture.py media OUT ../docs/figures [--still OUT/still_999902_048.png]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "python"))
DISPLAY = os.environ.get("DISPLAY", ":0")
SCREEN = (1920, 1080)
ISAAC_W = 1616   # Isaac Sim window; its stage/property panels sit under the right column
RIGHT_X = 1280   # camera (image_view, top) and RViz2 (bottom) share the right 640 px
CROP = "1866:1053:54:27"   # the three windows on a GNOME desktop (54 px dock, 27 px top bar)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SPEED = 4        # GUI rendering ran Isaac ~3.8x slower than real time, so 4x is ~sim time
# The VS Code snap leaks GTK/locale paths that make rviz2 load the snap's libpthread and die.
CLEAN = ("unset GTK_PATH GTK_EXE_PREFIX GIO_MODULE_DIR GTK_IM_MODULE_FILE "
         "GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR LOCPATH GSETTINGS_SCHEMA_DIR; "
         'export XDG_DATA_DIRS="${XDG_DATA_DIRS_VSCODE_SNAP_ORIG:-$XDG_DATA_DIRS}" '
         'XDG_CONFIG_DIRS="${XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG:-$XDG_CONFIG_DIRS}"; ')
ROS = (CLEAN + "source /opt/ros/humble/setup.bash >/dev/null 2>&1; "
       f"source {PKG}/scripts/ros_env.sh >/dev/null 2>&1; ")
CAMERA_SUFFIX = "/perception/landing_camera/annotated"


# ---------------------------------------------------------------- windows and screen
def windows() -> list[tuple[str, str]]:
    out = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            rows.append((parts[0], parts[3]))
    return rows


def find(test) -> str | None:
    return next((wid for wid, title in windows() if test(title)), None)


def place(wid: str, x: int, y: int, w: int, h: int) -> None:
    subprocess.run(["wmctrl", "-i", "-r", wid, "-b", "remove,maximized_vert,maximized_horz"],
                   capture_output=True)
    subprocess.run(["wmctrl", "-i", "-r", wid, "-e", f"0,{x},{y},{w},{h}"], capture_output=True)
    subprocess.run(["wmctrl", "-i", "-a", wid], capture_output=True)


def grab(path: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "0",
                    "-video_size", f"{SCREEN[0]}x{SCREEN[1]}", "-i", f"{DISPLAY}.0+0,0",
                    "-frames:v", "1", str(path)], capture_output=True)


def ros_tool(command: str, log: Path) -> subprocess.Popen:
    """A ROS 2 tool in its own process group, so ``ros2 run`` children stop with it."""
    return subprocess.Popen(["bash", "-lc", ROS + "exec " + command], start_new_session=True,
                            stdout=log.open("w"), stderr=subprocess.STDOUT)


def stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def rviz_config(path: Path) -> None:
    import yaml
    path.write_text(yaml.safe_dump({
        "Panels": [],
        "Visualization Manager": {
            "Class": "", "Enabled": True, "Name": "root",
            "Global Options": {"Background Color": "38; 41; 46", "Fixed Frame": "map",
                               "Frame Rate": 30},
            "Displays": [
                {"Class": "rviz_default_plugins/Grid", "Name": "Grid", "Enabled": True,
                 "Cell Size": 1, "Plane Cell Count": 120, "Reference Frame": "pad",
                 "Color": "110; 110; 110"},
                {"Class": "rviz_default_plugins/Path", "Name": "UAV path", "Enabled": True,
                 "Topic": {"Value": "/demo/uav_path"}, "Color": "60; 170; 255",
                 "Line Style": "Billboards", "Line Width": 0.06},
                {"Class": "rviz_default_plugins/Path", "Name": "Pad path", "Enabled": True,
                 "Topic": {"Value": "/demo/pad_path"}, "Color": "255; 140; 40",
                 "Line Style": "Billboards", "Line Width": 0.06},
                {"Class": "rviz_default_plugins/Odometry", "Name": "UAV", "Enabled": True,
                 "Topic": {"Value": "/landing_uav0/state/odom_truth", "Depth": 5},
                 "Keep": 1, "Shape": {"Value": "Axes", "Axes Length": 0.6, "Axes Radius": 0.06}},
                {"Class": "rviz_default_plugins/Odometry", "Name": "Pad", "Enabled": True,
                 "Topic": {"Value": "/landing_pad/state/odom_truth", "Depth": 5},
                 "Keep": 1, "Shape": {"Value": "Axes", "Axes Length": 0.8, "Axes Radius": 0.04}},
                {"Class": "rviz_default_plugins/TF", "Name": "TF", "Enabled": True,
                 "Show Names": True, "Show Axes": False, "Show Arrows": False}],
            "Views": {"Current": {"Class": "rviz_default_plugins/ThirdPersonFollower",
                                  "Target Frame": "uav", "Distance": 9.0, "Pitch": 0.45,
                                  "Yaw": 3.6, "Focal Point": {"X": 0, "Y": 0, "Z": 0}}}},
        "Window Geometry": {"Width": SCREEN[0] - RIGHT_X, "Height": 540}}, sort_keys=False))


# ---------------------------------------------------------------- capture
def capture(out: Path, checkpoint: Path, seeds: list[int], allow_isaac: bool) -> int:
    if not allow_isaac:
        raise SystemExit("an Isaac flight needs the user's approval and --allow-isaac")
    import torch
    from ontology_rgat.current_semantics.backend import make_isaac_backend
    from ontology_rgat.current_semantics.contracts import load_config
    from ontology_rgat.current_semantics.models import build_model
    from ontology_rgat.current_semantics.pilot import fly_policy
    from ontology_rgat.live_stack import live_stack
    from ontology_rgat.spatial.runtime_contract import deployment_profile

    out.mkdir(parents=True, exist_ok=True)
    config = load_config(PKG / "config/current_semantics/spatial_view_contact_v1.yaml")
    folder = checkpoint.parent.name
    arm = "orl_ppo:" + folder[len("orl_ppo_"):] if folder.startswith("orl_ppo_") else folder
    model = build_model(arm, config, seed=0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True)["model_state"], strict=True)
    model.eval()
    log = (out / "capture.log").open("a")

    def say(message: str) -> None:
        print(message, flush=True)
        log.write(f"{time.time():.3f} {message}\n")
        log.flush()

    timeline: dict = {"arm": arm, "checkpoint": str(checkpoint), "seeds": seeds,
                      "speed": SPEED, "crop": CROP, "episodes": []}
    current: dict = {}
    act = model.act

    def timed_act(*args, **kwargs):
        current.setdefault("decisions_wall", []).append(time.time())
        return act(*args, **kwargs)
    model.act = timed_act
    profile = Path(deployment_profile("spatial-reference/12")["path"])
    helpers: list[subprocess.Popen] = []
    recorder = None
    with live_stack(out / "isaac_stack", adopt=False, headless=False,
                    schema="spatial-reference/12", reset_recoveries=2, profile_path=profile):
        backend = make_isaac_backend(config, allow_isaac=True)
        reset = backend.reset

        def timed_reset(seed):
            current["reset_wall"] = time.time()
            return reset(seed)
        backend.reset = timed_reset
        try:
            topics = subprocess.run(["bash", "-lc", ROS + "ros2 topic list"], capture_output=True,
                                    text=True, timeout=60).stdout
            camera = next((t for t in topics.split() if t.endswith(CAMERA_SUFFIX)), CAMERA_SUFFIX)
            rviz_config(out / "orl_landing.rviz")
            helpers.append(ros_tool(f"python3 {PKG / 'tools/demo_tf_relay.py'}",
                                    out / "relay.log"))
            helpers.append(ros_tool(f"rviz2 -d {out / 'orl_landing.rviz'}", out / "rviz.log"))
            helpers.append(ros_tool(f"ros2 run image_view image_view --ros-args -r image:={camera}",
                                    out / "image_view.log"))
            ids: dict[str, str | None] = {}
            deadline = time.time() + 120
            while time.time() < deadline:
                ids = {"isaac": find(lambda t: t.startswith("Isaac Sim")),
                       "camera": find(lambda t: t == "/image"),
                       "rviz": find(lambda t: t.endswith(" - RViz") or t.endswith(" - RViz*"))}
                if all(ids.values()):
                    break
                time.sleep(1.0)
            say(f"windows {ids}")
            if not all(ids.values()):
                raise RuntimeError(f"a window did not appear: {ids}")
            place(ids["isaac"], 0, 0, ISAAC_W, SCREEN[1])
            place(ids["camera"], RIGHT_X, 0, SCREEN[0] - RIGHT_X, 480)
            place(ids["rviz"], RIGHT_X, 540, SCREEN[0] - RIGHT_X, 540)
            time.sleep(5.0)
            grab(out / "layout.png")
            recorder = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "0",
                 "-framerate", "15", "-video_size", f"{SCREEN[0]}x{SCREEN[1]}",
                 "-i", f"{DISPLAY}.0+0,0", "-c:v", "libx264", "-preset", "ultrafast",
                 "-crf", "20", "-pix_fmt", "yuv420p", str(out / "screen.mp4")],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=(out / "ffmpeg.log").open("w"))
            timeline["record_start_wall"] = time.time()
            for seed in seeds:
                current.clear()
                result: dict = {}

                def fly():
                    try:
                        result["episode"] = fly_policy(
                            backend, model, config, seed=seed, deterministic=True,
                            generator=torch.Generator().manual_seed(seed), record=True)
                    except Exception as exc:  # noqa: BLE001 - reported in the timeline
                        result["error"] = f"{type(exc).__name__}: {exc}"
                say(f"flying {arm} {checkpoint.name} seed {seed}")
                thread = threading.Thread(target=fly, daemon=True)
                thread.start()
                k = 0
                while thread.is_alive():
                    time.sleep(1.0)
                    if current.get("decisions_wall") and k % 4 == 0:
                        grab(out / f"still_{seed}_{k:03d}.png")
                    k += 1
                thread.join()
                end = time.time()
                grab(out / f"still_{seed}_end.png")
                episode = result.get("episode") or {}
                walls = current.get("decisions_wall", [])
                row = {"seed": seed, "status": episode.get("status"),
                       "decisions": episode.get("decisions"),
                       "sim_s": float(sum(episode["dt_s"])) if "dt_s" in episode else None,
                       "reset_wall": current.get("reset_wall"),
                       "handover_wall": walls[0] if walls else None,
                       "last_decision_wall": walls[-1] if walls else None, "end_wall": end,
                       "error": result.get("error")}
                timeline["episodes"].append(row)
                (out / "timeline.json").write_text(json.dumps(timeline, indent=2))
                say(json.dumps(row))
                time.sleep(4.0)
        finally:
            if recorder is not None:
                try:
                    recorder.communicate(b"q", timeout=30)
                except Exception:  # noqa: BLE001
                    recorder.kill()
                timeline["record_stop_wall"] = time.time()
            for proc in helpers:
                stop(proc)
            backend.close()
            (out / "timeline.json").write_text(json.dumps(timeline, indent=2))
    say("done")
    return 0


# ---------------------------------------------------------------- media
def _labels(timeline: dict, chosen: list[int], offset: float, size: int) -> str:
    """drawtext filters (output time) naming each flight's phase, plus the speed."""
    r0, eps = timeline["record_start_wall"], timeline["episodes"]
    pad, box, top = round(size * 0.5), round(size * 0.33), round(size * 1.7)

    def label(text: str, a: float, b: float, color: str = "white") -> str:
        a, b = (a - offset) / SPEED, (b - offset) / SPEED
        return (f"drawtext=fontfile={FONT}:text='{text}':x={pad}:y=h-{round(size * 1.95)}"
                f":fontsize={size}:fontcolor={color}:box=1:boxcolor=black@0.6:boxborderw={box}"
                f":enable='between(t,{a:.2f},{b:.2f})'")
    parts = []
    for i in chosen:
        e = eps[i]
        start = 0.0 if i == 0 else eps[i - 1]["end_wall"] - r0 + 1.0
        handover, last = e["handover_wall"] - r0, e["last_decision_wall"] - r0
        end = e["end_wall"] - r0 + 1.0
        tag = f"Demo flight {i + 1}/{len(eps)} (seed {e['seed']})"
        parts.append(label(f"{tag}   PX4 reset and hover, waiting for handover", start, handover))
        parts.append(label(f"{tag}   ORL policy flying", handover, last))
        status = str(e["status"]).replace(":", " ")
        verdict = (f"{status} (landed on the moving pad)" if status == "SUCCESS" else status)
        parts.append(label(f"{tag}   {verdict}", last, end,
                           "0x7CFC00" if status == "SUCCESS" else "0xFF6060"))
    parts.append(f"drawtext=fontfile={FONT}:text='{SPEED}x speed':x=w-tw-{pad}:y={top}"
                 f":fontsize={size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw={box}")
    return ",".join(parts)


def media(run: Path, figures: Path, still: Path | None) -> int:
    timeline = json.loads((run / "timeline.json").read_text())
    r0, eps = timeline["record_start_wall"], timeline["episodes"]
    crop = timeline.get("crop", CROP)
    screen = str(run / "screen.mp4")
    figures.mkdir(parents=True, exist_ok=True)
    ff = ["ffmpeg", "-y", "-loglevel", "error"]
    every = _labels(timeline, list(range(len(eps))), 0.0, 24)
    subprocess.run(ff + ["-i", screen, "-an", "-vf",
                         f"crop={crop},setpts=PTS/{SPEED},fps=30,scale=1280:-2:flags=lanczos,{every}",
                         "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", str(figures / "orl_isaac_px4_demo.mp4")],
                   check=True)
    first = eps[0]
    begin = max(0.0, first["handover_wall"] - r0 - 6.0)
    length = first["end_wall"] - r0 + 1.0 - begin
    one = _labels(timeline, [0], begin, 15)
    subprocess.run(ff + ["-ss", f"{begin:.2f}", "-t", f"{length:.2f}", "-i", screen, "-an", "-vf",
                         f"crop={crop},setpts=PTS/{SPEED},fps=8,scale=800:-2:flags=lanczos,{one},"
                         "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                         "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                         "-loop", "0", str(figures / "orl_isaac_px4_demo.gif")], check=True)
    if still is None:   # three quarters of the way through the first flight's policy phase
        at = first["handover_wall"] - r0 + 0.75 * (first["last_decision_wall"]
                                                   - first["handover_wall"])
        subprocess.run(ff + ["-ss", f"{at:.2f}", "-i", screen, "-frames:v", "1", "-vf",
                             f"crop={crop}", str(figures / "isaac_rviz_operation.png")], check=True)
    else:
        subprocess.run(ff + ["-i", str(still), "-vf", f"crop={crop}",
                             str(figures / "isaac_rviz_operation.png")], check=True)
    for name in ("isaac_rviz_operation.png", "orl_isaac_px4_demo.gif", "orl_isaac_px4_demo.mp4"):
        print(f"{figures / name}  {(figures / name).stat().st_size / 1e6:.1f} MB")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture", help="fly and record (needs --allow-isaac)")
    c.add_argument("out", type=Path)
    c.add_argument("checkpoint", type=Path)
    c.add_argument("--seeds", default="999902,999903,999904")
    c.add_argument("--allow-isaac", action="store_true")
    m = sub.add_parser("media", help="cut a capture into the still, GIF and MP4")
    m.add_argument("run", type=Path)
    m.add_argument("figures", type=Path)
    m.add_argument("--still", type=Path)
    args = parser.parse_args(argv)
    if args.command == "capture":
        return capture(args.out.resolve(), args.checkpoint.resolve(),
                       [int(s) for s in args.seeds.split(",")], args.allow_isaac)
    return media(args.run.resolve(), args.figures.resolve(), args.still)


if __name__ == "__main__":
    raise SystemExit(main())
