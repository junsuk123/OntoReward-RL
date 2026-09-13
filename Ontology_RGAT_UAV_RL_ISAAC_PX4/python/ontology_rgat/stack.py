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
import threading
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
                 gateway_timeout: float = 60.0,
                 parallel_pairs: int = 1):
        self.root = Path(cfg.paths.root)
        if not (self.root / "scripts").is_dir():
            raise StackError(f"No scripts/ directory under {self.root}.")
        self.isaac_sim_path = isaac_sim_path or os.environ.get("ISAACSIM_PATH", "")
        self.headless = self._resolve_headless(headless)
        self.gateway_args = gateway_args
        self.parallel_pairs = int(parallel_pairs)
        if self.parallel_pairs < 1:
            raise StackError("parallel_pairs must be positive")
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
        self.generation = 0
        self._restart_lock = threading.RLock()
        self._shutdown_requested = False

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
        parallel_marker = f"--parallel-pairs {self.parallel_pairs}"
        if _process_running("landing_world.py"):
            if self.parallel_pairs > 1 and not _process_running(parallel_marker):
                raise StackError(
                    "A single-pair Isaac world is already active. Let that run finish "
                    "before starting the requested multi-pair world.")
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
        isaac_command = [str(self.root / "scripts" / "run_isaac.sh"),
                         str(self.config_path)]
        if self.parallel_pairs > 1:
            isaac_command += ["--parallel-pairs", str(self.parallel_pairs)]
        log = self._launch(
            "isaac",
            isaac_command,
            env)
        # PX4 is launched by Pegasus once the world is loaded, so its banner is
        # the only signal that the whole simulator is up.
        print("Waiting for Isaac Sim and PX4 (first boot can take minutes)...")
        ready_count = lambda: sum(
            "Ready for takeoff" in line
            for line in log.read_text(errors="replace").splitlines()
        ) if log.is_file() else 0
        self._wait_for(lambda: ready_count() >= self.parallel_pairs,
                       self.timeouts["isaac"], "Isaac Sim + PX4 SITL", log)
        print(f"Isaac Sim up and {self.parallel_pairs} PX4 instance(s) report Ready for takeoff.")

    def start_gateway(self) -> None:
        for index in range(self.parallel_pairs):
            port = GATEWAY_PORT + 2 * index
            if _udp_port_bound(port):
                print(f"Gateway {index} already listening on UDP {port}; adopting it.")
                continue
            command = [str(self.root / "scripts" / "run_gateway.sh"),
                       "--config", str(self.config_path)] + shlex.split(
                           self.gateway_args)
            if self.parallel_pairs > 1:
                command += ["--pair-index", str(index),
                            "--parallel-pairs", str(self.parallel_pairs),
                            "--gateway-port", str(port)]
            log = self._launch(f"gateway_{index}", command)
            self._wait_for(lambda port=port: _udp_port_bound(port),
                           self.timeouts["gateway"], f"PX4 gateway {index}", log)
        # Binding UDP only proves that the node process is alive.  Its ROS 2
        # publishers can still be matching Isaac's subscriptions; a reset sent
        # in that short window is valid UDP but is dropped by DDS before any
        # subscriber exists.  One bounded discovery window avoids turning the
        # first episode of a long run into a full stack restart.
        time.sleep(1.0)
        ports = ", ".join(str(GATEWAY_PORT + 2 * i)
                          for i in range(self.parallel_pairs))
        print(f"Gateway(s) listening on UDP {ports}.")

    def restart(self) -> None:
        """Cycle the simulator.

        PX4 SITL degrades over long sessions -- ``battery_status`` goes stale
        under lockstep and arming is then refused -- so a long sweep may need a
        clean simulator rather than a lost run.
        """
        with self._restart_lock:
            if self._shutdown_requested:
                return
            self.stop()
            time.sleep(3.0)
            if self._shutdown_requested:
                return
            self.start()
            self.generation += 1

    def restart_if_generation(self, observed_generation: int) -> bool:
        """Restart once when several pair workers observe the same stack fault.

        Returns ``True`` only to the worker that performed the restart.  Other
        pair workers reconnect to the new generation without spending one of
        their own bounded recovery attempts on the same shared interruption.
        """
        with self._restart_lock:
            if (self._shutdown_requested
                    or self.generation != int(observed_generation)):
                return False
            self.stop()
            time.sleep(3.0)
            if self._shutdown_requested:
                return False
            self.start()
            self.generation += 1
            return True

    def shutdown(self) -> None:
        """Prevent recovery workers from relaunching while the pipeline exits."""
        self._shutdown_requested = True
        with self._restart_lock:
            self.stop()

    def stop(self) -> None:
        """Terminate only the processes this object started."""
        while self.managed:
            entry = self.managed.pop()
            print(f"Stopping {entry['name']} (pid {entry['pid']}).")
            try:
                self._kill_group(entry["pid"])
            finally:
                # Long experiments can cycle SITL many times. Keeping every
                # old stdout handle open eventually exhausts the learner's file
                # descriptors even though its child process has exited.
                handle = entry.get("handle")
                if handle is not None and not handle.closed:
                    handle.close()

    def is_ready(self) -> bool:
        return (_udp_port_bound(AGENT_PORT)
                and all(_udp_port_bound(GATEWAY_PORT + 2 * i)
                        for i in range(self.parallel_pairs))
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
