"""Node-agnostic metrics + report writing for the controller comparison test.

All functions take plain numpy arrays / dicts (no ROS or node dependency) so
they are easy to test and reuse. Sample arrays are laid out as:

* ``thrust``  -> shape (N, 2): columns ``[t, thrust]``
* ``torque``  -> shape (N, 4): columns ``[t, x, y, z]``
* ``pos_err`` -> shape (N, 4): columns ``[t, ex, ey, ez]``

Timestamps are seconds (any consistent clock). Integrals are trapezoidal.
The headline comparison numbers are the *track* phase ``int_thrust_abs`` /
``int_torque_norm`` (control effort) and ``int_err_norm`` (tracking error).

:func:`write_reports` lays a run out as::

    <output_dir>/<config_name>/metrics.json   # metrics + the gains flown + trajectory spec/stats
                              /metrics.csv
                              /traj.npz       # odom, setpoint, analytic reference, ...
                              /trajectory.csv # the polynomial pieces that were uploaded
                              /pid.conf       # verbatim copy of the gain file
"""
from __future__ import annotations

import csv
import json
import os
import shutil

import numpy as np

from crazybridge import trajectory

# Each run gets its own ``<output_dir>/<config_name>/`` folder, and the files
# inside are named identically across runs so tooling never has to interpolate
# the config name into a filename.
FILE_METRICS_JSON = 'metrics.json'
FILE_METRICS_CSV = 'metrics.csv'
FILE_TRAJ_NPZ = 'traj.npz'
FILE_TRAJECTORY_CSV = 'trajectory.csv'
FILE_PID_CONF = 'pid.conf'

# numpy 2.0 renamed trapz -> trapezoid (and later removed the old name).
try:
    from numpy import trapezoid as _np_trapz
except ImportError:  # numpy < 2.0
    from numpy import trapz as _np_trapz

# Phase segments reported. Each is (name, start_key, end_key) into the
# timestamps dict produced by the flight sequence.
SEGMENTS = (
    ('takeoff', 't_start', 't_takeoff_end'),
    ('regulate', 't_takeoff_end', 't_regulate_end'),
    ('track', 't_regulate_end', 't_track_end'),
    ('land', 't_track_end', 't_land_end'),
    ('total', 't_start', 't_land_end'),
)


def _slice(samples: np.ndarray, a: float, b: float) -> np.ndarray:
    """Rows of `samples` (column 0 = time) with timestamp in [a, b]."""
    if samples.size == 0:
        return samples
    mask = (samples[:, 0] >= a) & (samples[:, 0] <= b)
    return samples[mask]


def _trapz_abs(t: np.ndarray, v: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    return float(_np_trapz(np.abs(v), t))


def _trapz(t: np.ndarray, v: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    return float(_np_trapz(v, t))


def _segment_metrics(a: float, b: float, thrust: np.ndarray,
                     torque: np.ndarray, pos_err: np.ndarray) -> dict:
    seg: dict[str, float] = {'t_start': a, 't_end': b, 'duration_s': b - a}

    # --- control signals ---
    th = _slice(thrust, a, b)
    if th.size:
        seg['int_thrust_abs'] = _trapz_abs(th[:, 0], th[:, 1])
        seg['int_thrust_sq'] = _trapz(th[:, 0], th[:, 1] ** 2)
    else:
        seg['int_thrust_abs'] = seg['int_thrust_sq'] = 0.0

    tq = _slice(torque, a, b)
    if tq.size:
        seg['int_torque_x_abs'] = _trapz_abs(tq[:, 0], tq[:, 1])
        seg['int_torque_y_abs'] = _trapz_abs(tq[:, 0], tq[:, 2])
        seg['int_torque_z_abs'] = _trapz_abs(tq[:, 0], tq[:, 3])
        tq_norm = np.linalg.norm(tq[:, 1:4], axis=1)
        seg['int_torque_norm'] = _trapz(tq[:, 0], tq_norm)
        seg['int_torque_norm_sq'] = _trapz(tq[:, 0], tq_norm ** 2)
    else:
        for k in ('int_torque_x_abs', 'int_torque_y_abs', 'int_torque_z_abs',
                  'int_torque_norm', 'int_torque_norm_sq'):
            seg[k] = 0.0

    # --- position error (IAE-style) ---
    pe = _slice(pos_err, a, b)
    if pe.size:
        seg['int_err_x_abs'] = _trapz_abs(pe[:, 0], pe[:, 1])
        seg['int_err_y_abs'] = _trapz_abs(pe[:, 0], pe[:, 2])
        seg['int_err_z_abs'] = _trapz_abs(pe[:, 0], pe[:, 3])
        e_norm = np.linalg.norm(pe[:, 1:4], axis=1)
        seg['int_err_norm'] = _trapz(pe[:, 0], e_norm)     # headline metric
        seg['rmse_norm'] = float(np.sqrt(np.mean(e_norm ** 2)))
        seg['max_err_norm'] = float(np.max(e_norm))
    else:
        for k in ('int_err_x_abs', 'int_err_y_abs', 'int_err_z_abs',
                  'int_err_norm', 'rmse_norm', 'max_err_norm'):
            seg[k] = 0.0

    return seg


def compute_metrics(samples: dict, ts: dict) -> dict:
    """Per-phase and total integral metrics.

    `samples` maps 'thrust'/'torque'/'pos_err' to their arrays; `ts` maps the
    segment boundary keys to timestamps (seconds).
    """
    thrust = np.asarray(samples.get('thrust', np.empty((0, 2))), dtype=float)
    torque = np.asarray(samples.get('torque', np.empty((0, 4))), dtype=float)
    pos_err = np.asarray(samples.get('pos_err', np.empty((0, 4))), dtype=float)

    metrics: dict[str, dict] = {}
    for name, a_key, b_key in SEGMENTS:
        metrics[name] = _segment_metrics(
            ts[a_key], ts[b_key], thrust, torque, pos_err)
    return metrics


def flatten_row(config: str, metrics: dict) -> dict:
    """Flatten nested metrics into a single ``seg.metric`` -> value row."""
    row = {'config': config}
    for seg_name, seg in metrics.items():
        for k, v in seg.items():
            row[f'{seg_name}.{k}'] = v
    return row


# Units for ledger's numeric results, keyed by metric name (the part after the
# segment). Matches the units scripts/summarize.py labels its headline metrics
# with. Control-effort integrals are in raw firmware thrust units -> no unit.
METRIC_UNITS = {
    'duration_s': 's',
    'int_thrust_abs': '', 'int_thrust_sq': '',
    'int_torque_x_abs': 'N·m·s', 'int_torque_y_abs': 'N·m·s',
    'int_torque_z_abs': 'N·m·s', 'int_torque_norm': 'N·m·s',
    'int_torque_norm_sq': 'N²·m²·s',
    'int_err_x_abs': 'm·s', 'int_err_y_abs': 'm·s', 'int_err_z_abs': 'm·s',
    'int_err_norm': 'm·s',
    'rmse_norm': 'm', 'max_err_norm': 'm',
}

# Absolute node-clock timestamps are not metrics; they only locate the window.
_LEDGER_SKIP = ('t_start', 't_end')


def ledger_results(metrics: dict) -> list[dict]:
    """Flatten nested metrics into ledger ``{label, value, unit}`` results.

    Labels are ``<segment>.<metric>``, the same keys as ``metrics.csv`` and
    ``summarize.py``, so they line up across runs.
    """
    results = []
    for seg_name, seg in metrics.items():
        for key, value in seg.items():
            if key in _LEDGER_SKIP:
                continue
            results.append({'label': f'{seg_name}.{key}', 'value': value,
                            'unit': METRIC_UNITS.get(key, '')})
    return results


def run_dir(out_dir: str, config: str) -> str:
    """Directory holding one run's artefacts: ``<out_dir>/<config>/``."""
    return os.path.join(out_dir, config)


def write_reports(out_dir: str, config: str, ts: dict, metrics: dict,
                  traj: dict, gains: dict | None = None,
                  pid_conf_path: str | None = None) -> dict:
    """Write one run's artefacts into ``<out_dir>/<config>/``.

    Filenames are fixed (``metrics.json``, ``metrics.csv``, ``traj.npz``,
    ``pid.conf``) so they are identical across runs; the *directory* carries the
    config name. ``gains`` (the parsed pid.conf, flat ``name -> value``) and a
    verbatim copy of ``pid_conf_path`` are recorded so a run's numbers can
    always be traced back to the gains that produced them.

    Returns the flattened metric row (useful for logging / aggregation).
    """
    dest = run_dir(out_dir, config)
    os.makedirs(dest, exist_ok=True)

    report = {'config': config, 'timestamps': ts, 'metrics': metrics}
    if 'spec' in traj:
        # Spec + achieved peak speed/accel/extent/fit error, so the tracking
        # difficulty is recorded next to the numbers it produced.
        report['trajectory'] = {
            'spec': traj['spec'], 'timescale': float(traj.get('timescale', 1.0)),
            'stats': traj.get('stats', {}),
        }
    if gains is not None:
        report['gains'] = gains
    if pid_conf_path:
        report['pid_conf_path'] = os.path.abspath(pid_conf_path)
    with open(os.path.join(dest, FILE_METRICS_JSON), 'w') as f:
        json.dump(report, f, indent=2)

    row = flatten_row(config, metrics)
    with open(os.path.join(dest, FILE_METRICS_CSV), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    np.savez(
        os.path.join(dest, FILE_TRAJ_NPZ),
        odom=np.asarray(traj.get('odom', np.empty((0, 4))), dtype=float),
        setpoint=np.asarray(traj.get('setpoint', np.empty((0, 4))), dtype=float),
        goal=np.asarray(traj.get('goal', [0.0, 0.0, 0.0]), dtype=float),
        # Analytic reference path [t, x, y, z] on the same clock as odom/setpoint.
        reference=np.asarray(traj.get('reference', np.empty((0, 4))), dtype=float),
        shape=str(traj.get('shape', '')),
        spec_json=json.dumps(traj.get('spec', {})),
        timescale=float(traj.get('timescale', 1.0)),
        t_start=float(ts['t_start']), t_land_end=float(ts['t_land_end']),
        t_track_start=float(ts.get('t_track_start', ts['t_regulate_end'])),
        t_track_end=float(ts['t_track_end']),
    )

    if 'pieces' in traj:
        trajectory.to_csv(os.path.join(dest, FILE_TRAJECTORY_CSV), traj['pieces'])

    # Copy the gain file itself, so the run folder is self-contained even if the
    # source pid.conf is later edited in place.
    if pid_conf_path and os.path.isfile(pid_conf_path):
        shutil.copyfile(pid_conf_path, os.path.join(dest, FILE_PID_CONF))

    return row
