"""WMON v0.1 (Wave-to-Matter Ontological Network): numpy port of ``run_WMON_demo.m``.

Source: the single-file MATLAB prototype ``run_WMON_demo.m`` (2026-11-10,
``~/Downloads/wmon /``).  Pipeline per time step and per spatial axis:

    L0  causal FFT wave field over the last ``window`` samples of each sensor
        channel; keep DC and harmonics 1..n_modes, mix the reconstruction into
        the current sample by ``fft_mix``.
    L1-L2 information particle confidence = max(0.15, 1/(1 + s * hf_ratio)),
        hf_ratio = high-band power / non-DC power; a missing position
        particle has confidence 0.
    L3  three semantic atoms per axis: position, velocity, acceleration.
    L4  ontology bonds: p - p_prev - dt (v + v_prev)/2 and v - v_prev - dt a.
    L5  equilibrium = minimiser of the weighted on-site + bond energy, i.e. a
        3x3 weighted least-squares solve per axis.

It is a computational analogy, not a simulation of quantum or molecular
physics, and it contains no neural network.  Seven global coefficients are
learned by SPSA + Adam (``spsa_adam``).

``infer_episode`` reproduces the MATLAB ``inferEpisode`` to floating-point
round-off (tests/current_semantics/test_wmon_pad_track.py checks it against
MATLAB R2024b output).  The core is dimension-generic: a state with D spatial
axes is laid out as rows ``[p_1..p_D, v_1..v_D, a_1..a_D]``, as the MATLAB
``[px py vx vy ax ay]``.  ``wmon_track.WMONSpatialTrack`` adapts it to the
spatial ``Measurement`` stream.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np


# MATLAB constants (run_WMON_demo.m), named so nothing restates them.
CONFIDENCE_FLOOR = 0.15
POWER_FLOOR = 1e-4
RIDGE = 1e-8
VELOCITY_LOSS_WEIGHT = 0.35
GRADIENT_CLIP = 12.0
THETA_LOWER = np.array([-1.0, -1.0, -2.0, 0.0, 0.0, -3.0, -2.0])
THETA_UPPER = np.array([6.0, 6.0, 5.0, 7.0, 7.0, 3.0, 3.0])
THETA_NAMES = ("log_w_position", "log_w_velocity", "log_w_acceleration",
               "log_k_position_velocity", "log_k_velocity_acceleration",
               "logit_fft_mix", "log_noise_sensitivity")
ENERGY_TERMS = ("position", "velocity", "acceleration", "position_velocity_bond",
                "velocity_acceleration_bond")

#: theta of the MATLAB script before training.
MATLAB_INITIAL_THETA = np.array([math.log(9.0), math.log(18.0), math.log(4.0),
                                 math.log(90.0), math.log(55.0), math.log(0.3 / 0.7),
                                 math.log(1.3)])
#: theta the MATLAB script selected (WMON_output/WMON_result_data.mat, seed 20261110).
MATLAB_TRAINED_THETA = np.array([2.4342504915745997, 2.4762650361822094, 2.4631679490031715,
                                 4.365379895688865, 3.707446122531034, -2.4140137040961167,
                                 -0.12317111709384246])


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class WMONParams:
    """The seven learned scalars; ``theta`` is the MATLAB log/logit encoding."""
    w_position: float
    w_velocity: float
    w_acceleration: float
    k_position_velocity: float
    k_velocity_acceleration: float
    fft_mix: float
    noise_sensitivity: float

    def __post_init__(self) -> None:
        for name in ("w_position", "w_velocity", "w_acceleration", "k_position_velocity",
                     "k_velocity_acceleration", "noise_sensitivity"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")
        if not 0.0 <= self.fft_mix <= 1.0:
            raise ValueError(f"fft_mix must be in [0, 1], got {self.fft_mix!r}")

    @classmethod
    def from_theta(cls, theta: Sequence[float]) -> "WMONParams":
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.shape != (len(THETA_NAMES),):
            raise ValueError(f"theta needs {len(THETA_NAMES)} entries, got {theta.shape}")
        w = np.exp(theta[:5])
        return cls(*(float(v) for v in w), fft_mix=_sigmoid(float(theta[5])),
                   noise_sensitivity=float(math.exp(theta[6])))

    def to_theta(self) -> np.ndarray:
        mix = min(max(self.fft_mix, 1e-12), 1.0 - 1e-12)
        return np.array([math.log(self.w_position), math.log(self.w_velocity),
                         math.log(self.w_acceleration), math.log(self.k_position_velocity),
                         math.log(self.k_velocity_acceleration), math.log(mix / (1.0 - mix)),
                         math.log(self.noise_sensitivity)])

    @property
    def weights(self) -> np.ndarray:
        """[w_p, w_v, w_a, k_pv, k_va], the MATLAB ``exp(theta(1:5))``."""
        return np.array([self.w_position, self.w_velocity, self.w_acceleration,
                         self.k_position_velocity, self.k_velocity_acceleration])

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


MATLAB_TRAINED = WMONParams.from_theta(MATLAB_TRAINED_THETA)


# ------------------------------------------------------------------ L0-L2 wave

def fourier_mask(window: int, n_modes: int) -> np.ndarray:
    """1 on DC and harmonics 1..n_modes (both signs), 0 on the high band."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if n_modes < 0 or 2 * n_modes + 1 >= window:
        raise ValueError("n_modes must leave a non-empty high band (2 n_modes + 1 < window)")
    mask = np.zeros(window)
    mask[:n_modes + 1] = 1.0
    if n_modes:
        mask[-n_modes:] = 1.0
    return mask


@dataclass(frozen=True, eq=False)
class WaveField:
    sample: np.ndarray       # (..., C) the mixed current sample
    filtered: np.ndarray     # (..., C, W) low-band reconstruction of the window
    hf_ratio: np.ndarray     # (..., C)
    confidence: np.ndarray   # (..., C) particle confidence before any validity mask


def wave_field(segment: np.ndarray, mask: np.ndarray, fft_mix: float,
               noise_sensitivity: float) -> WaveField:
    """L0-L2 of one causal window ``segment`` (..., C, W); the last column is now."""
    segment = np.asarray(segment, dtype=np.float64)
    window = segment.shape[-1]
    if mask.shape != (window,):
        raise ValueError("mask length must equal the window")
    freq = np.fft.fft(segment, axis=-1) / window
    filtered = np.real(np.fft.ifft(freq * mask, axis=-1) * window)
    sample = (1.0 - fft_mix) * segment[..., -1] + fft_mix * filtered[..., -1]
    power = np.abs(freq) ** 2
    high = (power * (1.0 - mask)).sum(axis=-1)
    total = power[..., 1:].sum(axis=-1) + POWER_FLOOR
    hf_ratio = high / total
    confidence = np.maximum(CONFIDENCE_FLOOR, 1.0 / (1.0 + noise_sensitivity * hf_ratio))
    return WaveField(sample=sample, filtered=filtered, hf_ratio=hf_ratio, confidence=confidence)


# ------------------------------------------------------------------- L3-L5 atoms

def bond_design(dt: float) -> np.ndarray:
    """Rows: three on-site particles, the p-v bond and the v-a bond (MATLAB ``A``)."""
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [1.0, -dt / 2.0, 0.0],
                     [0.0, 1.0, -dt]])


def solve_atoms(particles: np.ndarray, previous_position: np.ndarray,
                previous_velocity: np.ndarray, weights: np.ndarray,
                dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-energy atoms of every axis at once.

    ``particles`` (..., 3) = [p, v, a] samples; ``weights`` (..., 5) =
    [w_p, w_v, w_a, k_pv, k_va] already multiplied by the particle confidence.
    Returns (atoms (..., 3), energy (..., 5) = 0.5 W r^2, normal matrix (..., 3, 3)).
    """
    design = bond_design(dt)
    particles = np.asarray(particles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    target = np.concatenate([particles,
                             (previous_position + dt / 2.0 * previous_velocity)[..., None],
                             np.asarray(previous_velocity, dtype=np.float64)[..., None]],
                            axis=-1)
    normal = np.einsum("ri,...r,rj->...ij", design, weights, design) + RIDGE * np.eye(3)
    rhs = np.einsum("ri,...r,...r->...i", design, weights, target)
    atoms = np.linalg.solve(normal, rhs[..., None])[..., 0]
    residual = np.einsum("ri,...i->...r", design, atoms) - target
    return atoms, 0.5 * weights * residual ** 2, normal


# --------------------------------------------------------- MATLAB episode path

def hold_missing_position(observation: np.ndarray, position_valid: np.ndarray) -> np.ndarray:
    """MATLAB ``fillMissingPosition``: causal zero-order hold of the position rows."""
    data = np.array(observation, dtype=np.float64)
    axes = _axes(data)
    valid = np.asarray(position_valid, dtype=bool)
    if valid.shape != (data.shape[1],):
        raise ValueError("position_valid needs one flag per time sample")
    if not valid[0]:
        raise ValueError("the first position sample must be valid (MATLAB forces 1:win)")
    for k in range(1, data.shape[1]):
        if not valid[k]:
            data[:axes, k] = data[:axes, k - 1]
    return data


def _axes(data: np.ndarray) -> int:
    if data.ndim != 2 or data.shape[0] % 3 or data.shape[0] == 0:
        raise ValueError("observation must be (3*D, T) with rows [p(D), v(D), a(D)]")
    return data.shape[0] // 3


@dataclass(frozen=True, eq=False)
class EpisodeEstimate:
    state: np.ndarray        # (3D, T)
    energy: np.ndarray       # (T, 5) summed over axes
    confidence: np.ndarray   # (T, 3) mean particle confidence per particle type
    hf_ratio: np.ndarray     # (T, 3D)


def infer_episode(params: WMONParams, observation: np.ndarray, position_valid: np.ndarray,
                  dt: float, *, window: int = 12, n_modes: int = 3,
                  use_bonds: bool = True) -> EpisodeEstimate:
    """MATLAB ``inferEpisode`` (``use_bonds`` = ``useOntology``), any number of axes."""
    data = hold_missing_position(observation, position_valid)
    axes, steps = _axes(data), data.shape[1]
    valid = np.asarray(position_valid, dtype=bool)
    mask = fourier_mask(window, n_modes)
    weights = params.weights
    bond = weights[3:] if use_bonds else np.zeros(2)
    state = np.zeros_like(data)
    energy = np.zeros((steps, 5))
    confidence = np.zeros((steps, 3))
    hf_ratio = np.zeros((steps, data.shape[0]))
    for k in range(steps):
        segment = data[:, max(0, k - window + 1):k + 1]
        if segment.shape[1] < window:
            segment = np.concatenate(
                [np.repeat(segment[:, :1], window - segment.shape[1], axis=1), segment], axis=1)
        wave = wave_field(segment, mask, params.fft_mix, params.noise_sensitivity)
        particle = wave.confidence.copy()
        particle[:axes] *= float(valid[k])
        confidence[k] = particle.reshape(3, axes).mean(axis=1)
        hf_ratio[k] = wave.hf_ratio
        if k == 0:
            state[:, 0] = wave.sample
            continue
        q = wave.sample.reshape(3, axes).T                       # (D, 3)
        w = np.empty((axes, 5))
        w[:, :3] = weights[:3] * particle.reshape(3, axes).T
        if not valid[k]:
            w[:, 0] = 0.0
        w[:, 3:] = bond
        atoms, terms, _ = solve_atoms(q, state[:axes, k - 1], state[axes:2 * axes, k - 1],
                                      w, dt)
        state[:, k] = atoms.T.reshape(-1)
        energy[k] = terms.sum(axis=0)
    return EpisodeEstimate(state=state, energy=energy, confidence=confidence,
                           hf_ratio=hf_ratio)


def episode_loss(params: WMONParams, observation: np.ndarray, position_valid: np.ndarray,
                 truth: np.ndarray, dt: float, *, window: int = 12, n_modes: int = 3) -> float:
    """One term of MATLAB ``batchLoss``: position MSE + 0.35 velocity MSE from ``win``."""
    estimate = infer_episode(params, observation, position_valid, dt, window=window,
                             n_modes=n_modes)
    axes = _axes(np.asarray(observation))
    ids = slice(window - 1, None)
    position = estimate.state[:axes, ids] - truth[:axes, ids]
    velocity = estimate.state[axes:2 * axes, ids] - truth[axes:2 * axes, ids]
    return float(np.mean(position ** 2) + VELOCITY_LOSS_WEIGHT * np.mean(velocity ** 2))


class WMONStream:
    """``infer_episode`` one sample at a time (same arithmetic), with a per-step dt.

    ``dt_ref`` (optional) rescales the bond weights by (dt_ref/dt)**exponent,
    as in OmniDrones' torch port; None is the MATLAB step form.
    """

    def __init__(self, params: WMONParams, axes: int, *, window: int = 12, n_modes: int = 3,
                 use_bonds: bool = True, dt_ref: float | None = None,
                 bond_dt_exponent: float = 1.0) -> None:
        self.params, self.axes, self.window = params, axes, window
        self.mask = fourier_mask(window, n_modes)
        self.use_bonds, self.dt_ref, self.bond_dt_exponent = use_bonds, dt_ref, bond_dt_exponent
        self.reset()

    def reset(self) -> None:
        self.columns: list[np.ndarray] = []
        self.state = np.zeros(3 * self.axes)
        self.confidence = np.ones(3)
        self.energy = np.zeros(5)
        self.held_position: np.ndarray | None = None

    def step(self, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray,
             position_valid: bool, dt: float) -> np.ndarray:
        axes = self.axes
        first = self.held_position is None
        if first and not position_valid:
            raise ValueError("the first sample needs a valid position (MATLAB keeps 1:win valid)")
        if position_valid:
            self.held_position = np.asarray(position, dtype=np.float64).copy()
        column = np.concatenate([self.held_position, np.asarray(velocity, np.float64),
                                 np.asarray(acceleration, np.float64)])
        self.columns.append(column)
        if len(self.columns) > self.window:
            self.columns.pop(0)
        segment = np.stack(self.columns, axis=1)
        if segment.shape[1] < self.window:
            segment = np.concatenate(
                [np.repeat(segment[:, :1], self.window - segment.shape[1], axis=1), segment],
                axis=1)
        p = self.params
        wave = wave_field(segment, self.mask, p.fft_mix, p.noise_sensitivity)
        particle = wave.confidence.copy()
        particle[:axes] *= float(position_valid)
        self.confidence = particle.reshape(3, axes).mean(axis=1)
        if first:
            self.state = wave.sample.copy()
            return self.state.copy()
        weights = p.weights
        bond = weights[3:].copy() if self.use_bonds else np.zeros(2)
        if self.dt_ref is not None:
            bond *= (self.dt_ref / dt) ** self.bond_dt_exponent
        w = np.empty((axes, 5))
        w[:, :3] = weights[:3] * particle.reshape(3, axes).T
        if not position_valid:
            w[:, 0] = 0.0
        w[:, 3:] = bond
        atoms, terms, _ = solve_atoms(wave.sample.reshape(3, axes).T, self.state[:axes],
                                      self.state[axes:2 * axes], w, dt)
        self.state = atoms.T.reshape(-1)
        self.energy = terms.sum(axis=0)
        return self.state.copy()


# ------------------------------------------------------------------- training

@dataclass(frozen=True, eq=False)
class SPSAResult:
    theta: np.ndarray            # best-validation theta (MATLAB bestTheta)
    final_theta: np.ndarray
    train_loss: np.ndarray       # (epochs,) on every training item
    validation_loss: np.ndarray  # (epochs,) NaN without a validation function
    best_epoch: int


def spsa_adam(batch_loss: Callable[[np.ndarray, Sequence[int]], float], n_train: int,
              theta0: Sequence[float], *, epochs: int = 50, batch_size: int = 4,
              learning_rate: float = 0.045,
              validation_loss: Callable[[np.ndarray], float] | None = None,
              rng: np.random.Generator | None = None,
              lower: np.ndarray = THETA_LOWER, upper: np.ndarray = THETA_UPPER,
              log: Callable[[str], None] | None = None) -> SPSAResult:
    """The MATLAB training loop: two-sided SPSA gradient, Adam, box clamp, best-val pick.

    ``batch_loss(theta, indices)`` returns the mean loss over training items
    ``indices``.  Random streams differ from MATLAB's ``rng``; the update rule,
    perturbation schedule c = 0.085/(1 + 0.025 epoch)^0.2, clip and bounds do not.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = np.clip(np.asarray(theta0, dtype=np.float64).copy(), lower, upper)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    train_history = np.full(epochs, np.nan)
    val_history = np.full(epochs, np.nan)
    best, best_val, best_epoch = theta.copy(), math.inf, 0
    everything = list(range(n_train))
    for epoch in range(1, epochs + 1):
        ids = list(rng.permutation(n_train)[:min(batch_size, n_train)])
        delta = 2.0 * (rng.uniform(size=theta.shape) > 0.5) - 1.0
        c = 0.085 / (1.0 + 0.025 * epoch) ** 0.2
        plus = batch_loss(theta + c * delta, ids)
        minus = batch_loss(theta - c * delta, ids)
        g = np.clip((plus - minus) / (2.0 * c) * delta, -GRADIENT_CLIP, GRADIENT_CLIP)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        theta = theta - learning_rate * (m / (1.0 - beta1 ** epoch)) / (
            np.sqrt(v / (1.0 - beta2 ** epoch)) + eps)
        theta = np.clip(theta, lower, upper)
        train_history[epoch - 1] = batch_loss(theta, everything)
        score = train_history[epoch - 1]
        if validation_loss is not None:
            val_history[epoch - 1] = score = validation_loss(theta)
        if score < best_val:
            best_val, best, best_epoch = score, theta.copy(), epoch
        if log is not None and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            log(f"epoch {epoch:3d}/{epochs} | train {train_history[epoch - 1]:.5f} | "
                f"validation {val_history[epoch - 1]:.5f}")
    return SPSAResult(theta=best, final_theta=theta, train_loss=train_history,
                      validation_loss=val_history, best_epoch=best_epoch)
