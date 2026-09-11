"""Own the DDS agent, Isaac Sim/PX4 and the gateway processes.

Port of the retired ``matlab/src/+stack/ExternalStack.m``. The pipeline needs
three long-lived processes that are normally started by hand in three
terminals. This starts them in the order ``docs/OPERATIONS.md`` requires, waits
for each readiness signal, and guarantees that whatever it started is stopped
again, including on error.

Processes that are already running are adopted, not restarted, and are left
alone on teardown: a session the user started by hand is theirs.

One thing got simpler in the port. MATLAB prepends its own runtime to the
loader path, which breaks ROS 2 and Isaac, so the MATLAB version had to scrub
``LD_LIBRARY_PATH`` and friends out of every child's environment. A plain
Python process does not do that, so the children inherit the environment they
would get from a terminal.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .config import Config

__all__ = ["ExternalStack", "current"]

AGENT_PORT = 8888          # Micro XRCE-DDS Agent, UDP
GATEWAY_PORT = 14650       # learner <-> gateway, UDP

_registered: "ExternalStack | None" = None
_KEEP = object()   # "no argument", distinct from an explicit None


def current(new_stack: Any = _KEEP) -> "ExternalStack | None":
    """The stack the running pipeline owns, or ``None``.

    ``env.LandingEnv.reset`` consults this to recover from a simulator that has
    stopped accepting arm commands, which PX4 SITL does after long sessions.
    Only the pipeline registers a stack; a hand-started stack is never
    restarted from under the user.
    """
    global _registered
    if new_stack is not _KEEP:
        _registered = new_stack
    return _registered


def _udp_port_bound(port: int) -> bool:
    result = subprocess.run(["ss", "-lnuH"], capture_output=True, text=True, check=False)
    return f":{port} " in result.stdout


def _process_running(pattern: str) -> bool:
    result = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True,
                            check=False)
    return any(pattern in line for line in result.stdout.splitlines())


def _log_contains(path: Path, needle: str) -> bool:
    if not path.is_file():
        return False
    try:
        return needle in path.read_text(errors="replace")
    except OSError:
        return False


def _tail(path: Path, lines: int = 20) -> str:
    if not path or not Path(path).is_file():
        return "(no log)"
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(unreadable log)"


class StackError(RuntimeError):
    pass


class ExternalStack:
    """Start, adopt and stop the three processes the experiment runs on."""

    def __init__(self, cfg: Config, *, isaac_sim_path: str | None = None,
                 headless: bool | str = "auto",
                 gateway_args: str = "--target sitl --allow-arm",
                 config_path: str | Path | None = None,
                 log_dir: str | Path | None = None,
                 agent_timeout: float = 30.0, isaac_timeout: float = 600.0,
                 gateway_timeout: float = 60.0):
        self.root = Path(cfg.paths.root)
        if not (self.root / "scripts").is_dir():
            raise StackError(f"No scripts/ directory under {self.root}.")
        self.isaac_sim_path = isaac_sim_path or os.environ.get("ISAACSIM_PATH", "")
        self.headless = self._resolve_headless(headless)
        self.gateway_args = gateway_args
        self.config_path = (Path(config_path).expanduser().resolve()
                            if config_path is not None
                            else (self.root / "config" / "system.yaml").resolve())
        if not self.config_path.is_file():
            raise StackError(f"Stack configuration does not exist: {self.config_path}")
        self.log_dir = Path(log_dir) if log_dir else Path("/tmp/ontology_rgat_stack")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timeouts = {"agent": agent_timeout, "isaac": isaac_timeout,
                         "gateway": gateway_timeout}
        self.managed: list[dict[str, Any]] = []

    @staticmethod
    def _resolve_headless(value: bool | str) -> bool:
        """``'auto'`` shows a window wherever there is a display to show it on."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() == "auto":
            return not os.environ.get("DISPLAY")
        raise StackError("headless must be True, False or 'auto'")

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.start_agent()
        self.start_isaac()
        self.start_gateway()
        print(f"External stack ready (logs in {self.log_dir}).")

    def start_agent(self) -> None:
        if _udp_port_bound(AGENT_PORT):
            print(f"DDS agent already listening on UDP {AGENT_PORT}; adopting it.")
            return
        log = self._launch("agent", [str(self.root / "scripts" / "run_dds_agent.sh")])
        self._wait_for(lambda: _udp_port_bound(AGENT_PORT), self.timeouts["agent"],
                       "DDS agent", log)
        print(f"DDS agent listening on UDP {AGENT_PORT}.")

    def start_isaac(self) -> None:
        if _process_running("landing_world.py"):
            if not self.headless and _process_running("landing_world.py --headless"):
                raise StackError(
                    "Isaac Sim is already running in headless mode. Stop that process "
                    "before starting the default GUI pipeline, or rerun with "
                    "--headless.")
            print("Isaac/PX4 already running; adopting it.")
            return
        if not self.isaac_sim_path or not (Path(self.isaac_sim_path) / "python.sh").is_file():
            raise StackError(
                "Isaac Sim not found. Pass --isaac-sim-path or set ISAACSIM_PATH to "
                f"the directory containing python.sh (got {self.isaac_sim_path!r}).")
        env = {"ISAACSIM_PATH": self.isaac_sim_path,
               "HEADLESS": "1" if self.headless else "0"}
        if not self.headless:
            display = os.environ.get("DISPLAY")
            if not display:
                raise StackError("Isaac Sim was asked for a window but DISPLAY is not "
                                 "set. Run from a graphical session, or pass --headless.")
            print(f"Isaac Sim will open a window on DISPLAY {display}.")
        log = self._launch(
            "isaac",
            [str(self.root / "scripts" / "run_isaac.sh"), str(self.config_path)],
            env)
        # PX4 is launched by Pegasus once the world is loaded, so its banner is
        # the only signal that the whole simulator is up.
        print("Waiting for Isaac Sim and PX4 (first boot can take minutes)...")
        self._wait_for(lambda: _log_contains(log, "Ready for takeoff"),
                       self.timeouts["isaac"], "Isaac Sim + PX4 SITL", log)
        print("Isaac Sim up and PX4 reports Ready for takeoff.")

    def start_gateway(self) -> None:
        if _udp_port_bound(GATEWAY_PORT):
            print(f"Gateway already listening on UDP {GATEWAY_PORT}; adopting it.")
            return
        command = [str(self.root / "scripts" / "run_gateway.sh"),
                   "--config", str(self.config_path)] + shlex.split(self.gateway_args)
        log = self._launch("gateway", command)
        self._wait_for(lambda: _udp_port_bound(GATEWAY_PORT),
                       self.timeouts["gateway"], "PX4 gateway", log)
        # Binding UDP only proves that the node process is alive.  Its ROS 2
        # publishers can still be matching Isaac's subscriptions; a reset sent
        # in that short window is valid UDP but is dropped by DDS before any
        # subscriber exists.  One bounded discovery window avoids turning the
        # first episode of a long run into a full stack restart.
        time.sleep(1.0)
        print(f"Gateway listening on UDP {GATEWAY_PORT}.")

    def restart(self) -> None:
        """Cycle the simulator.

        PX4 SITL degrades over long sessions -- ``battery_status`` goes stale
        under lockstep and arming is then refused -- so a long sweep may need a
        clean simulator rather than a lost run.
        """
        self.stop()
        time.sleep(3.0)
        self.start()

    def stop(self) -> None:
        """Terminate only the processes this object started."""
        while self.managed:
            entry = self.managed.pop()
            print(f"Stopping {entry['name']} (pid {entry['pid']}).")
            self._kill_group(entry["pid"])

    def is_ready(self) -> bool:
        return (_udp_port_bound(AGENT_PORT) and _udp_port_bound(GATEWAY_PORT)
                and _process_running("landing_world.py"))

    def __enter__(self) -> "ExternalStack":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ---------------------------------------------------------------- detail
    def _launch(self, name: str, command: list[str],
                env: dict[str, str] | None = None) -> Path:
        """Start one process in its own session and record its pid.

        ``start_new_session`` puts the whole tree in one process group, so
        Pegasus' PX4 child dies with Isaac instead of holding TCP 4560.
        """
        log = self.log_dir / f"{name}.log"
        environment = dict(os.environ)
        environment.update(env or {})
        handle = log.open("wb")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=environment,
                                   start_new_session=True, cwd=str(self.root))
        self.managed.append({"name": name, "pid": process.pid, "log": log,
                             "process": process, "handle": handle})
        return log

    def _wait_for(self, predicate: Callable[[], bool], timeout: float,
                  description: str, log: Path) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            dead = [e for e in self.managed if e["process"].poll() is not None]
            if dead:
                raise StackError(f"{description} exited during startup. Log:\n{_tail(log)}")
            time.sleep(0.5)
        raise StackError(f"{description} was not ready within {timeout:.0f} s. "
                         f"Log:\n{_tail(log)}")

    @staticmethod
    def _kill_group(pid: int) -> None:
        # A negative pid targets the process group created by start_new_session,
        # so Pegasus' PX4 child goes down with the simulator.
        # Let rclpy/Isaac close their contexts and publishers before escalating.
        # SIGTERM makes Isaac's bridge invalidate its context mid-frame, which
        # produces alarming RCLError tracebacks on every otherwise clean run.
        for sig, patience in ((signal.SIGINT, 40), (signal.SIGTERM, 40),
                              (signal.SIGKILL, 0)):
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            for _ in range(patience):
                try:
                    os.killpg(pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.25)
