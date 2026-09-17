"""Experiment configuration.

Port of the retired ``config/defaultConfig.m`` plus ``defaultExternalConfig.m``.
The two files were split because the original workspace was read-only; here
there is one function and the external terms are simply part of it.

Everything the in-process MATLAB rigid-body simulator owned is gone: rotor and
motor models, the analytic wind field, the panel aerodynamics and the synthetic
sensor noise. Isaac and PX4 own all four, so keeping a second set of numbers
here would only invite them to disagree. What survives is what the *learning*
side needs: the semantic feature scalings, the landing criteria, the ontology
schema, and the R-GAT/PPO hyperparameters.

``config/system.yaml`` remains the authority for anything the simulator or the
gateway acts on. The two overlap in exactly three places -- the collective
mapping, the relative-speed success criterion and the protocol -- and each
overlap is checked at run time rather than trusted (see ``bridge.PX4Bridge``).
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterator

__all__ = ["Config", "default_config", "WORKSPACE_ROOT"]

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """A nested dict with attribute access, so ``cfg.rgat.lr`` reads naturally.

    MATLAB structs are value types: ``c = cfg; c.external.padScale = s`` left
    the caller's ``cfg`` untouched. Python dicts are not, so the sweeps use
    :meth:`derive` where the MATLAB code relied on copy-on-assign.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, _wrap(value))

    def derive(self, **overrides: Any) -> "Config":
        """Deep copy with dotted-path overrides: ``cfg.derive(**{'external.pad_scale': 2.0})``."""
        out = copy.deepcopy(self)
        for path, value in overrides.items():
            node = out
            parts = path.split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
        return out

    def flatten(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        for key, value in self.items():
            path = f"{prefix}{key}"
            if isinstance(value, Config):
                yield from value.flatten(path + ".")
            else:
                yield path, value


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def default_config(mode: str = "quick", target: str = "sitl") -> Config:
    """Full configuration for one experiment.

    ``mode`` is ``'quick'`` (smoke and iteration) or ``'full'`` (the paper-scale
    sweep). ``target`` is ``'sitl'`` or ``'hardware'``.
    """
    mode = mode.lower()
    target = target.lower()
    if mode not in {"quick", "full"}:
        raise ValueError(f"mode must be 'quick' or 'full', got {mode!r}")
    if target not in {"sitl", "hardware"}:
        raise ValueError(f"target must be 'sitl' or 'hardware', got {target!r}")

    cfg = Config()
    cfg.mode = mode
    cfg.seed = 42

    # ---------------------------------------------------------------- episode
    cfg.sim = {
        "dt": 0.02,                 # s, the 50 Hz control period
        # A lorry at urban speed covers a hundred metres in an episode, and the
        # vehicle has to match its velocity before it can afford to descend.
        "max_time": 20.0,           # s
        "g": 9.80665,
        "world_xy_limit": 30.0,     # m, pad-relative: losing the lorry ends it
        "max_altitude": 18.0,       # m, above the lorry's roof
        "crash_tilt": math.radians(75.0),
        "ground_z": 0.06,           # m above the deck, pad-relative
        "monitor_every": 5,         # control steps between real-time viz updates
    }
    cfg.sim.max_steps = int(round(cfg.sim.max_time / cfg.sim.dt))

    # Only the two terms the semantic features price thrust with survive from
    # the retired airframe model; Isaac flies the vehicle.
    cfg.drone = {"mass": 1.477, "max_total_thrust": 30.0}

    # ------------------------------------------------- semantic feature scales
    cfg.semantic = {
        "wind_speed_thr": 8.0,                  # m/s measured by the UAV sensor
        "wind_accel_thr": 4.0,                  # m/s^2
        "wind_dir_thr": math.radians(50.0),
        # measured speed, inferred aerodynamic load, acceleration, direction
        "wind_risk_w": [1.2, 2.0, 1.4, 1.0],
        "wind_risk_b": -1.4,
        "align_scale": 0.75,
        "att_tilt_scale": math.radians(22.0),
        "att_rate_scale": math.radians(80.0),
        "vz_safe_scale": 0.65,
        # The deck speed at which chasing the target dominates the landing.
        # Urban traffic, so this is a lorry moving with the flow rather than a
        # rover creeping across a laboratory floor.
        "pad_speed_scale": 6.0,                 # m/s
        # Energy margin normalisation, in units of the episode horizon.
        "energy_scale": 1.0,
        # The horizontal position uncertainty at which the fix has stopped
        # being a usable input for a landing. A canyon fix runs 5-25 m; a good
        # open-sky one is under 2 m.
        "gnss_sigma_scale": 12.0,               # m
    }

    # ---------------------------------------------- landing success criteria
    # Evaluation ground truth, never training reward weights. ``rel_speed_xy``
    # matches ``landing.success_rel_speed_xy_m_s`` in config/system.yaml.
    cfg.criteria = {
        "xy": 0.35,                       # m, on the roof of a box lorry
        "vz": 0.55,                       # m/s at contact
        "tilt": math.radians(10.0),
        "rate": math.radians(45.0),
        "rel_speed_xy": 0.45,             # m/s relative to the moving deck
    }

    # ------------------------------------------------------- policy interface
    cfg.rl = {
        # pad-relative pose/velocity, attitude, rates, three semantic channels,
        # deck velocity/motion context, reserve and energy margin,
        # and the receiver's own account of itself.
        "obs_dim": 23,
        "act_dim": 4,                     # collective, roll, pitch, yaw rate
        "max_roll_pitch": math.radians(28.0),
        "max_yaw_rate": math.radians(90.0),
        "collective_span": 0.85,          # hover*(1 + span*a)
    }

    # ------------------------------------------------------------- rewards
    cfg.reward = {
        "manual": {
            "w_pos": 1.3, "w_vel": 0.35, "w_tilt": 0.65, "w_rate": 0.10,
            "w_wind": 0.45, "w_act": 0.025,
            # Hand-tuned weights for the three external factors. The point of
            # the baseline is that these are arbitrary; the proposed arm has to
            # derive the same trade-offs from the ontology instead. w_nav
            # prices flying on a pose nothing can vouch for -- which the policy
            # can act on, by keeping the markers in frame when the fix is bad.
            "w_pad_track": 0.55, "w_energy": 0.40, "w_nav": 0.35,
            "success": 20.0, "failure": -20.0, "timeout": -20.0,
            "battery_depleted": -20.0,
            "time": 0.25,                 # per-second pressure; hovering is not free
            "viol_span": 2.0,             # criteria-ratio span the crash penalty grades over
            "failure_floor": 0.25,        # fraction of the crash penalty a near miss costs
        },
        "sparse": {
            "success": 10.0, "failure": -10.0, "time": -0.15,
            "timeout": -10.0, "battery_depleted": -10.0,
        },
        "pbrs": {
            "lambda": 2.0,
            # Must equal ppo.gamma, or the shaping is no longer policy-invariant.
            "gamma": 0.999,
        },
        # After R-GAT learns a context-dependent safe-landing potential, its
        # counterfactual sensitivities are distilled to these bounded, fixed
        # coefficients. They sum to one and stay frozen for the whole PPO run.
        "fixed": {
            "weight_min": 0.025,
            "weight_max": 0.45,
            "importance_floor": 1e-6,
            "max_attribution_samples": 4096,
        },
    }

    # -------------------------------------------------------- ontology schema
    # PadMotion (11), BatteryReserve (12) and GnssIntegrity (13) are the three
    # nodes the external environment introduces, so SafeLanding is node 14. A
    # model trained against a shorter schema is not loadable: the dimension
    # check in semantic.build_ontology_graph refuses it rather than quietly
    # attending over the wrong nodes.
    cfg.ontology = {
        "node_names": [
            "PositionError", "VerticalSpeed", "TiltAngle", "AngularRate",
            "WindRisk", "MarkerQuality", "VisualStability", "Alignment",
            "AttitudeStability", "TouchdownSafety", "PadMotion",
            "BatteryReserve", "GnssIntegrity", "SafeLanding",
        ],
        "relation_names": ["degrades", "supports", "contributes", "self"],
    }
    cfg.ontology.n_nodes = len(cfg.ontology.node_names)
    cfg.ontology.n_relations = len(cfg.ontology.relation_names)
    cfg.ontology.in_dim = 4 + cfg.ontology.n_nodes

    # ------------------------------------------------------------------ R-GAT
    # The layer follows Busbridge et al. 2019 ("Relational Graph Attention
    # Networks", https://openreview.net/forum?id=Bklzkh0qFm) as implemented by
    # babylonhealth/rgat. The defaults below are the exact configuration the
    # retired MATLAB layer implemented, so numbers stay comparable across the
    # port; the remaining knobs are the parts of that paper the MATLAB code
    # never had. See python/ontology_rgat/rgat/layers.py.
    cfg.rgat = {
        "hidden_dim": 24,
        "rel_dim": 6,                     # relation-embedding width in the logits
        "lr": 2e-3,
        "batch_size": 32,
        "sample_stride": 3,
        # Behaviour-policy perturbation is sampled log-uniformly: the expert
        # success boundary sits near sigma=0.05, so a uniform sweep to 0.65
        # would label ~96% of the dataset negative.
        "noise_range": [0.02, 0.65],
        "val_fraction": 0.2,
        "output_l2": 1e-4,                # weak regularisation on the potential
        # --- Busbridge et al. options -------------------------------------
        "heads": 1,
        "head_aggregation": "mean",       # mean | sum | concat | projection
        "attention_mode": "argat",        # argat | wirgat
        "attention_style": "sum",         # sum (GAT additive) | dot (transformer)
        "attention_units": 1,             # must be 1 for 'sum' style
        "attn_leaky_relu_slope": 0.2,
        "kernel_basis_size": None,        # W_r = sum_i c_{i,r} W'_i; None disables
        "attn_kernel_basis_size": None,
        "feature_dropout": 0.0,
        "support_dropout": 0.0,
        "residual": True,                 # second layer is H2 = tanh(layer(H1) + H1)
        "softmax_floor": 1e-9,            # the MATLAB layer's denominator floor
        "stable_softmax": False,          # True subtracts the per-node max first
    }

    # -------------------------------------------------------------------- PPO
    cfg.ppo = {
        "hidden": 64,
        "gamma": 0.999,                   # horizon >> max_steps, so terminals are visible
        "lambda_gae": 0.95,
        "clip": 0.20,
        "entropy_coef": 0.003,
        "value_coef": 0.5,
        "actor_lr": 2e-4,
        "critic_lr": 7e-4,
        "epochs": 5,
        "minibatch": 64,
        "rollout_steps": 2048,            # batch many episodes before each update
        "init_log_std": -0.55,
        "log_std_bounds": [-3.0, 0.5],
        "grad_clip": 5.0,
        "mu_scale": 1.5,                  # mu = mu_scale*tanh(...) before squashing
    }

    # ----------------------------------------------------------------- device
    # Measured, not assumed: python/ontology_rgat/rgat/benchmark.py re-measures
    # the crossover on the active machine. The graph has 14 nodes and a 24-wide
    # hidden layer, so a small batch is kernel-launch bound rather than FLOP
    # bound and the GPU can still lose. 'auto' therefore declines below
    # ``min_batch_for_gpu`` instead of quietly rebatching -- raising the batch
    # size changes how many Adam steps the potential sees, which is a
    # hyperparameter decision and not a free speedup.
    #
    # On this RTX 4060 laptop, per forward+backward pass (40 repetitions):
    #
    #     batch      CPU        GPU      speedup
    #        32     4.3 ms     4.8 ms      0.9x
    #        64     7.4 ms     5.0 ms      1.5x
    #       128    32.7 ms     4.9 ms      6.6x
    #      1024    88.8 ms     6.6 ms     13.4x
    #      4096   193.9 ms     9.5 ms     20.4x
    #
    # Batch 32 is a tie that varies between runs, 64 is the first clear gain.
    # The retired MATLAB layer needed batch 1024 before the GPU paid for
    # itself, because every relation was a separate traced dlarray operation.
    cfg.device = {
        "rgat": "auto",                   # auto | cuda | cpu
        "min_batch_for_gpu": 64,
        "rgat_precision": "float32",      # float32 | float64 | bfloat16 (autocast)
        "allow_tf32": True,
        "compile": False,                 # torch.compile the potential
        # PPO's networks are 64 wide and one 2048-step update costs ~2 s on the
        # CPU against ~41 s of real-time flight collection, so the GPU cannot
        # shorten this pipeline. Override only to measure it.
        "ppo": "cpu",
    }

    # ----------------------------------------------------------- evaluation
    cfg.eval = {
        "seed0": 5000,
        "wind_scales": [0.5, 1.0, 1.5, 2.0],
        # 0.0 is the parked-lorry control condition, i.e. the original
        # experiment. 1.5 puts the lorry at up to 12 m/s, which is already
        # faster than PX4's position controller holds the entry pose behind.
        "pad_scales": [0.0, 0.5, 1.0, 1.5],
        # 0.0 is open sky in the same city: the buildings still stand and still
        # block the camera's view, but no satellite is lost or reflected. That
        # is what isolates the GNSS effect from the geometry.
        "gnss_scales": [0.0, 0.5, 1.0, 1.5, 2.0],
        "battery_bins_s": [0.0, 10.0, 20.0, 30.0, 50.0],
        # Two independent acceptance gates. Reward quality is the nominal
        # landing success rate. R-GAT robustness is the variation of that rate
        # across wind, pad-motion, GNSS and energy strata; a uniformly bad
        # policy cannot pass merely by being consistently bad.
        "acceptance": {
            "min_success_rate": 0.60,
            "max_success_std": 0.15,
            "min_worst_case_success": 0.35,
            "max_rgat_val_mse": 0.35,
        },
    }

    # ------------------------------------------------------- visualization
    # Every panel the retired MATLAB monitors drew now has a Python owner:
    # RViz 2 for the 3D/real-time view, the web dashboard for unattended
    # progress, matplotlib for the publication figures. See viz/.
    cfg.viz = {
        "training": True,                 # live training telemetry at all
        "realtime": True,                 # per-step episode telemetry
        "live_every": 5,                  # episodes/epochs between snapshot exports
        "live_export": True,              # PNG + CSV under results/live
        "rviz": {
            "enabled": True,              # publish RViz 2 topics when rclpy is present
            "namespace": "/landing_rl",
            "world_frame": "map",
            "pad_frame": "landing_pad",
            "body_frame": "uav_body",
            "trail_length": 900,
            "publish_rate_hz": 5.0,       # visual traffic; control timing is unchanged
            "publish_ontology_graph": True,
            "graph_origin_pad_m": [0.0, -2.6, 1.6],
            "graph_scale_m": 0.42,
        },
        "dashboard": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8770,
            "history": 4000,              # points retained per live series
        },
        # The dashboard's 3D view of the ontology with the R-GAT's attention
        # on its edges. One snapshot is ~40 edges, so the cost is the attention
        # read-out itself, which is why it is throttled rather than per step.
        "graph3d": {
            "enabled": True,
            "every": 5,                   # control steps/epochs between snapshots
        },
        "isaac_overlay": True,            # 3D debug draw inside the Isaac window
    }

    # ------------------------------------------------------- external factors
    cfg.pad = {"enabled": True}
    cfg.gnss = {"enabled": True}
    cfg.battery = {
        "enabled": True,
        # Descent rate the margin calculation prices "can I still land from
        # here" with, deliberately conservative next to the expert's profile.
        "plan_descent_rate": 0.55,        # m/s
    }

    # ------------------------------------------------------- external stack
    cfg.external = {
        "enabled": True,
        "target": target,
        "protocol_version": 1,
        "gateway_host": "127.0.0.1",
        "gateway_port": 14650,
        "local_host": "127.0.0.1",
        "local_port": 14651,
        "timeout": 2.0,
        # Setup traffic only -- first contact after a boot, the episode reset,
        # and the first entry setpoint. A shared two-pair stack is restarted
        # under a worker that did nothing wrong, and Isaac then spends minutes
        # loading the city, the rover and the vehicle before the gateway can
        # answer at all. At the 2 s control budget that reconnect fails and
        # burns one of the worker's bounded recovery attempts on a simulator
        # that is merely still starting. Flight steps keep "timeout": a step
        # that waits minutes is a stall nobody noticed, and a gap that long
        # inside an episode is not a trajectory PPO may learn from.
        "setup_timeout": 120.0,
        "estimator_warmup": 5.0,
        "auto_arm": target == "sitl",
        "start_airborne": True,
        "prestream_count": 24,
        "control_hz": 50.0,
        "reset_settle": 0.15,
        "outcome_settle_timeout": 20.0,
        # PX4 flies the seeded entry pose before the policy takes over. With a
        # moving deck that pose is an offset in the pad frame and the gateway
        # re-aims it at the live deck, so these tolerances are on the
        # pad-relative state and never on a world point. They are looser than
        # the fixed-pad experiment's because PX4's position controller lags a
        # setpoint that is itself driving at 8 m/s, and the entry pose is a
        # starting condition rather than a landing.
        "entry_frame": "pad",
        # Tight again, because the climb is flown and judged on the same
        # simulator-side pose (bridge.entry_state): there is no receiver error
        # between the setpoint and the check, so what is left is how well PX4
        # holds station in the wind. It has to stay well inside the camera
        # footprint -- a few metres at entry altitude -- or the episode cannot
        # be guaranteed to open with the deck in frame.
        "entry_tolerance": 0.90,          # m
        # Pad-relative, and measured on the simulator's own state rather than
        # the receiver's (see bridge.entry_state), so this is the speed the
        # vehicle actually has. PX4 holds station to about 0.1 m/s between
        # gusts, so 0.60 is comfortable again now that the canyon's velocity
        # error is no longer being counted as motion. A starting condition, not
        # a landing criterion: cfg.criteria grades the episode and is unchanged.
        "entry_speed_tolerance": 0.60,    # m/s, pad-relative
        # Hand over only once the deck is actually in the camera frame, so
        # every episode opens with the landing target visible rather than on a
        # GNSS estimate that is tens of metres out in this canyon.
        "require_pad_in_view": True,
        # One definition, and only one: the pad centre projects inside the
        # landing camera's frustum (bridge.entry_view_margin, computed on the
        # same pad-relative pose the climb is judged on). The margin is the
        # admissible fraction of the half field of view; 0.85 keeps the pad
        # centre out of the outer 15 % of the frame on every side. Detector
        # success is deliberately not consulted -- mixing it in gave the
        # experiment two incompatible meanings of "visible".
        "entry_view_margin": 0.85,
        # The rendered landing camera. Copied from the system YAML's
        # vision.camera block by the live runner so the entry gate, the
        # geometric FOV metric and the R-GAT labels all use the camera Isaac
        # actually renders with.
        "landing_camera": {
            "resolution": [512, 320],
            "horizontal_fov_deg": 90.0,
            "pitch_down_deg": 60.0,
            "mount_translation_flu_m": [0.0, 0.0, -0.16],
        },
        # How long a disarmed vehicle may keep being refused arming before the
        # gate gives up. PX4 SITL that has degraded in a long session refuses
        # command 400 indefinitely; waiting out the whole entry budget, eight
        # bounded retries deep, is a quarter of an hour spent confirming it.
        "entry_arm_grace": 25.0,          # s
        # Re-aims allowed when the vehicle holds the commanded offset and speed
        # but the deck is not in frame. The seeded entry pose is never redrawn:
        # both arms must receive the same initial condition.
        "entry_view_retries": 2,
        "entry_settle": 0.5,              # s held inside tolerance
        # Simulated seconds the vehicle gets to reach the seeded entry pose and
        # hold it for entry_settle. This is the budget the gate is actually
        # judged against, and it is on PX4's clock for the same reason the
        # settle streak is: the manoeuvre is a physical duration. A wall-clock
        # budget shrinks it as the stage gets heavier -- on the two-camera
        # rendered city stage the previous 90 s wall budget bought only a
        # fraction of the simulated time the same number bought on a flat
        # plane, and arriving vehicles were cut off mid-settle. Flying a few
        # metres and settling needs roughly ten to twenty simulated seconds;
        # this leaves a wide margin without letting a stuck climb run forever.
        "entry_sim_budget": 60.0,         # s of simulated PX4 time
        # The budget above pays for the manoeuvre, not for the trip. Between
        # episodes the vehicle can be tens of metres from the deck -- a blind
        # or diverging policy ends its episode out there, and the staging hold
        # only brings it back to within nine metres -- so a fixed budget times
        # out vehicles that are still travelling at PX4's own speed limit.
        # Pay for the measured distance at this closing speed, on top of the
        # fixed budget. MPC_XY_VEL_MAX is 2.0 m/s in this SITL profile; the
        # allowance assumes less so acceleration and the climb are covered.
        "entry_travel_speed": 1.2,        # m/s of assumed closing speed
        "entry_travel_budget_max": 60.0,  # s of simulated PX4 time, at most
        # A gateway-classified recoverable SITL link failsafe (OFFBOARD
        # heartbeat loss and its benign status-clear race) clears on its own
        # in well under a second: the gateway keeps streaming setpoints and
        # re-requests OFFBOARD as soon as the flag drops. While it is set PX4
        # ignores the entry setpoints, so the vehicle stands still and the
        # gate reports a stationary vehicle off target. Wait it out for this
        # long before treating it as a fault; a hard failsafe never waits.
        "failsafe_grace": 10.0,           # s of wall clock
        # Unmeasured OFFBOARD position hold between episodes (bridge.
        # hold_for_next_airborne_reset). It must outlast PPO/estimator
        # updates; the gateway lands the vehicle when it expires, which PX4
        # then reports as an offboard-loss failsafe at the next reset.
        "between_episode_hold_s": 900.0,
        # Wall-clock hang guard, no longer the entry budget itself: that is
        # entry_sim_budget above, plus at most entry_travel_budget_max. This
        # bounds a simulator that has stopped publishing time at all, so it
        # must outlast that sum at the slowest stage rate the run is expected
        # to reach -- the measured two-pair city stage advances roughly one
        # simulated second per wall second, so 240 s covers the 120 s ceiling
        # with room to spare -- and PX4's post-boot arm refusal, which
        # entry_arm_grace cuts short anyway.
        "entry_timeout": 240.0,           # s of wall clock
        "arm_retry": 2.0,                 # s between arm attempts during the climb
        # PX4 SITL stops accepting arm commands after hours of lockstep; a
        # pipeline that owns the simulator cycles it rather than losing the run.
        "reset_recoveries": 2,
        "wind_scale": 1.0,
        "pad_scale": 1.0,
        "gnss_scale": 1.0,
    }

    # ------------------------------------------------------------------ sizes
    if mode == "quick":
        cfg.rgat.data_episodes = 24
        cfg.rgat.epochs = 10
        cfg.ppo.train_episodes = 150
        cfg.eval.episodes = 12
        cfg.eval.sweep_episodes = 6
    else:
        cfg.rgat.data_episodes = 400
        cfg.rgat.epochs = 80
        cfg.ppo.train_episodes = 3000
        cfg.eval.episodes = 200
        cfg.eval.sweep_episodes = 50

    # ------------------------------------------------------------------ paths
    root = WORKSPACE_ROOT
    cfg.paths = {
        "root": str(root),
        "system_yaml": str(root / "config" / "system.yaml"),
        "results": str(root / "results"),
        "models": str(root / "results" / "models"),
        "figures": str(root / "results" / "figures"),
        "data": str(root / "results" / "data"),
        "live": str(root / "results" / "live"),
    }
    for key in ("results", "models", "figures", "data", "live"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)
    return cfg
