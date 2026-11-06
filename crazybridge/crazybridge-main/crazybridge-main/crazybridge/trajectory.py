"""Host-side generator for the tracking-phase trajectory (pure numpy, no ROS).

The controller test uploads the tracking trajectory to the Crazyflie as
piecewise 7th-order polynomials and lets the firmware's high-level commander
execute it. This module builds those pieces.

Shapes (both start at ``spec.goal`` at rest)::

    figure8   x = gx + A sin(phi)    y = gy + B sin(2 phi)    z = gz + C sin^2(phi)
              A 3D Lissajous 1:2 figure-8 (z rises on both lobes). Returns to
              ``goal`` at rest after a whole number of laps.
    spiral    r = R phi / Phi,  x = gx + r cos(phi),  y = gy + r sin(phi),
              z = gz + H phi / Phi.   An expanding conical helix: speed and
              centripetal acceleration grow along the path. Ends at rest on the
              outer turn.

Time law
--------
The path parameter ``phi`` follows a trapezoidal *rate* profile: ``dphi/dt``
ramps 0 -> omega with a quintic smoothstep over ``ramp_s``, cruises, and ramps
back to 0. The ramps make velocity, acceleration and jerk all start and end at
zero, so there is no setpoint step when the trajectory takes over from a hover.
``omega`` is calibrated numerically so the *peak speed* of the resulting
trajectory equals ``spec.peak_speed_mps``; the cruise time is solved so the
parameter ends at exactly ``2 pi laps``.

Piece fitting
-------------
The trajectory is cut into ``n_pieces`` equal-duration pieces. Each axis of each
piece is the unique degree-7 polynomial matching position, velocity,
acceleration and jerk of the analytic trajectory at both ends (a Hermite
interpolant -- 8 conditions, 8 coefficients). Neighbouring pieces share their
boundary values, so the whole trajectory is C3 without any optimisation.

Firmware constraints (``crtp_commander_high_level.c``)
-------------------------------------------------------
A raw poly4d piece is 33 float32 values (132 B) and the trajectory memory is
4096 B, so one trajectory holds at most 31 pieces. The row layout used here is
exactly the one of the cflib examples / crazyswarm2 CSVs::

    [duration, x^0..x^7, y^0..y^7, z^0..z^7, yaw^0..yaw^7]
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SHAPES = ('figure8', 'spiral')
MAX_PIECES = 31          # 4096 B trajectory memory // 132 B per raw poly4d piece
N_COEFF = 8              # degree-7 polynomial
ROW_LEN = 1 + 4 * N_COEFF  # duration + (x, y, z, yaw) * 8

CSV_HEADER = ','.join(
    ['Duration'] + [f'{ax}^{k}' for ax in ('x', 'y', 'z', 'yaw')
                    for k in range(N_COEFF)])


@dataclass(frozen=True)
class TrajectorySpec:
    """Everything that defines a tracking trajectory (JSON-serialisable)."""
    shape: str = 'figure8'
    goal: tuple = (0.5, 0.0, 1.0)             # start point / shape origin [m]
    peak_speed_mps: float = 1.0               # peak |velocity| [m/s]
    laps: int = 2                             # whole laps of the path parameter
    ramp_s: float = 2.0                       # speed ramp-up / ramp-down time [s]
    n_pieces: int = 30                        # polynomial pieces (<= 31)
    figure8_amplitude_xyz: tuple = (0.6, 0.3, 0.3)   # A, B, C [m]
    spiral_radius_m: float = 0.8              # final spiral radius R [m]
    spiral_ascent_m: float = 0.4              # total climb H over the spiral [m]

    def validate(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f'shape must be one of {SHAPES}, got {self.shape!r}')
        if len(self.goal) != 3:
            raise ValueError('goal must be (x, y, z)')
        if len(self.figure8_amplitude_xyz) != 3:
            raise ValueError('figure8_amplitude_xyz must be (A, B, C)')
        if self.peak_speed_mps <= 0.0:
            raise ValueError('peak_speed_mps must be > 0')
        if int(self.laps) < 1:
            raise ValueError('laps must be >= 1')
        if self.ramp_s <= 0.0:
            raise ValueError('ramp_s must be > 0')
        if not 1 <= int(self.n_pieces) <= MAX_PIECES:
            raise ValueError(
                f'n_pieces must be in [1, {MAX_PIECES}] (firmware trajectory '
                f'memory holds {MAX_PIECES} raw pieces), got {self.n_pieces}')
        if self.shape == 'figure8' and min(self.figure8_amplitude_xyz) < 0.0:
            raise ValueError('figure8_amplitude_xyz must be >= 0')
        if self.shape == 'spiral' and self.spiral_radius_m <= 0.0:
            raise ValueError('spiral_radius_m must be > 0')

    def to_dict(self) -> dict:
        d = asdict(self)
        d['goal'] = [float(v) for v in self.goal]
        d['figure8_amplitude_xyz'] = [float(v) for v in self.figure8_amplitude_xyz]
        return d


# --------------------------------------------------------------------------- #
# Path: analytic curve f(phi) and its first three derivatives w.r.t. phi
# --------------------------------------------------------------------------- #
def _curve(spec: TrajectorySpec, phi: np.ndarray):
    """Return ``(f, f', f'', f''')``, each shaped ``(n, 3)``."""
    phi = np.asarray(phi, dtype=float)
    gx, gy, gz = (float(v) for v in spec.goal)
    zero = np.zeros_like(phi)

    if spec.shape == 'figure8':
        A, B, C = (float(v) for v in spec.figure8_amplitude_xyz)
        s1, c1 = np.sin(phi), np.cos(phi)
        s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
        f0 = np.column_stack([gx + A * s1, gy + B * s2, gz + C * s1 ** 2])
        f1 = np.column_stack([A * c1, 2 * B * c2, C * s2])
        f2 = np.column_stack([-A * s1, -4 * B * s2, 2 * C * c2])
        f3 = np.column_stack([-A * c1, -8 * B * c2, -4 * C * s2])
        return f0, f1, f2, f3

    # spiral: r = k phi  (k = R / Phi), constant climb rate H / Phi
    Phi = 2.0 * np.pi * int(spec.laps)
    k = float(spec.spiral_radius_m) / Phi
    dz = float(spec.spiral_ascent_m) / Phi
    r = k * phi
    s, c = np.sin(phi), np.cos(phi)
    f0 = np.column_stack([gx + r * c, gy + r * s, gz + dz * phi])
    f1 = np.column_stack([k * c - r * s, k * s + r * c, zero + dz])
    f2 = np.column_stack([-2 * k * s - r * c, 2 * k * c - r * s, zero])
    f3 = np.column_stack([-3 * k * c + r * s, -3 * k * s - r * c, zero])
    return f0, f1, f2, f3


# --------------------------------------------------------------------------- #
# Time law: phi(t) with quintic-smoothstep ramps on the rate
# --------------------------------------------------------------------------- #
def _S(u):    # smoothstep 0 -> 1 with zero 1st and 2nd derivative at both ends
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def _S1(u):   # dS/du
    return 30 * u ** 2 * (1 - u) ** 2


def _S2(u):   # d2S/du2
    return 60 * u * (1 - u) * (1 - 2 * u)


def _G(u):    # integral of S from 0 to u  (G(1) = 1/2)
    return 2.5 * u ** 4 - 3 * u ** 5 + u ** 6


@dataclass(frozen=True)
class _Plan:
    omega: float   # cruise phase rate [rad/s]
    ramp: float    # ramp time [s]
    phi_end: float  # total phase swept = 2 pi laps
    total: float   # total duration [s]


def _phase(plan: _Plan, t):
    """``phi, dphi, d2phi, d3phi`` at times ``t`` (clipped to ``[0, total]``)."""
    t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, plan.total)
    om, Tr, T = plan.omega, plan.ramp, plan.total
    phi = np.empty_like(t)
    d1 = np.empty_like(t)
    d2 = np.empty_like(t)
    d3 = np.empty_like(t)

    up = t < Tr
    dn = t > T - Tr
    mid = ~(up | dn)

    u = t[up] / Tr
    phi[up] = om * Tr * _G(u)
    d1[up] = om * _S(u)
    d2[up] = om * _S1(u) / Tr
    d3[up] = om * _S2(u) / Tr ** 2

    phi[mid] = om * Tr / 2.0 + om * (t[mid] - Tr)
    d1[mid] = om
    d2[mid] = 0.0
    d3[mid] = 0.0

    w = (T - t[dn]) / Tr          # runs 1 -> 0 over the ramp-down
    phi[dn] = plan.phi_end - om * Tr * _G(w)
    d1[dn] = om * _S(w)
    d2[dn] = -om * _S1(w) / Tr
    d3[dn] = om * _S2(w) / Tr ** 2
    return phi, d1, d2, d3


def _state(spec: TrajectorySpec, plan: _Plan, t):
    """Analytic ``(pos, vel, acc, jerk)`` of the trajectory, each ``(n, 3)``."""
    phi, d1, d2, d3 = _phase(plan, t)
    f0, f1, f2, f3 = _curve(spec, phi)
    col = lambda x: x[:, None]  # noqa: E731
    vel = f1 * col(d1)
    acc = f2 * col(d1 ** 2) + f1 * col(d2)
    jerk = f3 * col(d1 ** 3) + 3.0 * f2 * col(d1 * d2) + f1 * col(d3)
    return f0, vel, acc, jerk


def _peak_speed(spec: TrajectorySpec, plan: _Plan) -> float:
    t = np.linspace(0.0, plan.total, 3001)
    _, v, _, _ = _state(spec, plan, t)
    return float(np.linalg.norm(v, axis=1).max())


def _make_plan(spec: TrajectorySpec) -> _Plan:
    """Calibrate ``omega`` so the trajectory's peak speed matches the spec."""
    phi_end = 2.0 * np.pi * int(spec.laps)
    grid = np.linspace(0.0, phi_end, 4001)
    max_f1 = float(np.linalg.norm(_curve(spec, grid)[1], axis=1).max())
    omega = spec.peak_speed_mps / max_f1      # first guess: ignores the ramps

    plan = None
    for _ in range(60):
        if phi_end / omega < spec.ramp_s:
            raise ValueError(
                f'ramp_s={spec.ramp_s} s leaves no cruise phase: sweeping '
                f'{phi_end:.2f} rad at omega={omega:.3f} rad/s takes only '
                f'{phi_end / omega:.2f} s. Shorten ramp_s, add laps or lower '
                f'peak_speed_mps.')
        plan = _Plan(omega, spec.ramp_s, phi_end, phi_end / omega + spec.ramp_s)
        ratio = spec.peak_speed_mps / _peak_speed(spec, plan)
        if abs(ratio - 1.0) < 1e-5:
            break
        omega *= ratio
    assert plan is not None      # the loop either sets it or raises
    return plan


# --------------------------------------------------------------------------- #
# Piece fitting
# --------------------------------------------------------------------------- #
# Derivatives (orders 0..3) at s = 1 of the basis s^4 .. s^7.
_HERMITE_A = np.array([[1, 1, 1, 1],
                       [4, 5, 6, 7],
                       [12, 20, 30, 42],
                       [24, 60, 120, 210]], dtype=float)


def _hermite7(p0, v0, a0, j0, p1, v1, a1, j1, h) -> np.ndarray:
    """Degree-7 coefficients (ascending, piece-local time) matching p, v, a, j.

    Solved in the normalised variable ``s = tau / h`` for conditioning, then
    rescaled back to ``tau``.
    """
    d = np.zeros(N_COEFF)
    d[0] = p0
    d[1] = h * v0
    d[2] = h ** 2 * a0 / 2.0
    d[3] = h ** 3 * j0 / 6.0
    r = np.array([
        p1 - (d[0] + d[1] + d[2] + d[3]),
        h * v1 - (d[1] + 2 * d[2] + 3 * d[3]),
        h ** 2 * a1 - (2 * d[2] + 6 * d[3]),
        h ** 3 * j1 - 6 * d[3],
    ])
    d[4:] = np.linalg.solve(_HERMITE_A, r)
    return d / h ** np.arange(N_COEFF)


def _build(spec: TrajectorySpec):
    spec.validate()
    plan = _make_plan(spec)
    n = int(spec.n_pieces)
    h = plan.total / n
    p, v, a, j = _state(spec, plan, np.linspace(0.0, plan.total, n + 1))

    pieces = np.zeros((n, ROW_LEN))
    pieces[:, 0] = h
    for i in range(n):
        for ax in range(3):            # yaw (ax 3) stays all-zero
            lo = 1 + ax * N_COEFF
            pieces[i, lo:lo + N_COEFF] = _hermite7(
                p[i, ax], v[i, ax], a[i, ax], j[i, ax],
                p[i + 1, ax], v[i + 1, ax], a[i + 1, ax], j[i + 1, ax], h)
    return plan, pieces


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate(spec: TrajectorySpec) -> tuple[np.ndarray, float]:
    """Build the polynomial pieces.

    Returns ``(pieces, duration_s)``: ``pieces`` is ``(n_pieces, 33)`` in the
    cflib/crazyswarm row layout (see module docstring), ``duration_s`` the total
    duration at ``timescale = 1``.
    """
    plan, pieces = _build(spec)
    return pieces, plan.total


def eval_pieces(pieces: np.ndarray, t, deriv: int = 0) -> np.ndarray:
    """Evaluate the xyz position (or its ``deriv``-th derivative) of the pieces.

    This is what the firmware evaluates, so it is the ground truth for what will
    actually be flown. ``t`` is clipped to the trajectory's time span.
    """
    pieces = np.asarray(pieces, dtype=float)
    if not 0 <= deriv < N_COEFF:
        raise ValueError(f'deriv must be in [0, {N_COEFF - 1}]')
    dur = pieces[:, 0]
    edges = np.concatenate([[0.0], np.cumsum(dur)])
    t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, edges[-1])
    idx = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, len(dur) - 1)
    tau = t - edges[idx]

    out = np.empty((t.size, 3))
    for ax in range(3):
        coef = pieces[:, 1 + ax * N_COEFF:1 + (ax + 1) * N_COEFF]
        if deriv:
            coef = np.polynomial.polynomial.polyder(coef, m=deriv, axis=1)
        c = coef[idx]
        acc = np.zeros(t.size)
        for k in reversed(range(c.shape[1])):    # Horner
            acc = acc * tau + c[:, k]
        out[:, ax] = acc
    return out


def sample_reference(spec: TrajectorySpec, dt: float = 0.02,
                     timescale: float = 1.0) -> np.ndarray:
    """Analytic reference path as ``[t, x, y, z]`` rows (``t`` from trajectory start).

    ``timescale`` stretches time the way the firmware does (> 1 = slower).
    """
    spec.validate()
    plan = _make_plan(spec)
    t = np.arange(0.0, plan.total, dt / max(timescale, 1e-9))
    t = np.append(t, plan.total)
    pos = _state(spec, plan, t)[0]
    return np.column_stack([t * timescale, pos])


def stats(spec: TrajectorySpec) -> dict:
    """Numbers describing the trajectory, including its polynomial fit error."""
    plan, pieces = _build(spec)
    t = np.linspace(0.0, plan.total, 4001)
    p, v, a, j = _state(spec, plan, t)
    fit = eval_pieces(pieces, t)
    goal = np.asarray(spec.goal, dtype=float)
    return {
        'shape': spec.shape,
        'n_pieces': int(spec.n_pieces),
        'duration_s': float(plan.total),
        'omega_rad_s': float(plan.omega),
        'peak_speed_mps': float(np.linalg.norm(v, axis=1).max()),
        'peak_accel_mps2': float(np.linalg.norm(a, axis=1).max()),
        'peak_jerk_mps3': float(np.linalg.norm(j, axis=1).max()),
        'max_dist_from_goal_m': float(np.linalg.norm(p - goal, axis=1).max()),
        'bbox_min': [float(x) for x in p.min(axis=0)],
        'bbox_max': [float(x) for x in p.max(axis=0)],
        'max_fit_error_m': float(np.linalg.norm(fit - p, axis=1).max()),
    }


def to_csv(path: str, pieces: np.ndarray) -> None:
    """Write the pieces in the standard ``Duration,x^0,...,yaw^7`` CSV layout."""
    np.savetxt(path, np.asarray(pieces, dtype=float), delimiter=',',
               header=CSV_HEADER, comments='', fmt='%.9e')
