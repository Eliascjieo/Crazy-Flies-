#!/usr/bin/env python3
"""Aggregate controller-comparison runs into a table, charts and a Rerun overlay.

Reads every ``<config>/metrics.json`` (and ``<config>/traj.npz``) written by
``controller_test`` under ``--output-dir`` and produces:

* a ranked comparison **table** of the headline tracking-phase metrics, printed
  and written to ``summary.csv``;
* **bar charts** (``summary.png``) comparing control effort and tracking error;
* a combined **Rerun** recording (``comparison.rrd``) overlaying every config's
  flown path against the shared reference trajectory (figure-8 / spiral).

Matplotlib / Rerun are optional -- if either is missing that output is skipped.

    python3 summarize.py --output-dir ~/ctrl_results
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

# Headline metrics (tracking phase) used for the ranked comparison.
HEADLINE = [
    ('track.int_err_norm', 'int|pos_err| (m·s)'),       # tracking (primary rank)
    ('track.rmse_norm', 'RMSE |e| (m)'),
    ('track.max_err_norm', 'max |e| (m)'),
    ('track.int_thrust_abs', 'int|thrust|'),            # control effort
    ('track.int_torque_norm', 'int|torque| (N·m·s)'),
]
RANK_KEY = 'track.int_err_norm'

# Distinct colors for up to 8 configs in the overlay / charts.
PALETTE = [
    (255, 140, 0), (0, 180, 255), (0, 200, 100), (220, 60, 200),
    (240, 210, 0), (150, 90, 255), (255, 80, 80), (120, 200, 200),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True,
                   help='Directory containing one <config>/ folder per run '
                        '(each with metrics.json and traj.npz).')
    return p.parse_args()


def _get(metrics: dict, dotted: str) -> float:
    seg, key = dotted.split('.', 1)
    return float(metrics.get(seg, {}).get(key, float('nan')))


def load_runs(out_dir: str) -> list[dict]:
    """One run per ``<out_dir>/<config>/metrics.json`` (the layout written by
    ``metrics.write_reports``); the folder name is the fallback config label."""
    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, '*', 'metrics.json'))):
        with open(path) as f:
            data = json.load(f)
        runs.append({
            'config': data.get('config', os.path.basename(os.path.dirname(path))),
            'metrics': data.get('metrics', {}),
            'shape': data.get('trajectory', {}).get('spec', {}).get('shape', ''),
        })
    return runs


def build_table(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        row = {'config': r['config'], 'shape': r['shape']}
        for key, _label in HEADLINE:
            row[key] = _get(r['metrics'], key)
        rows.append(row)
    rows.sort(key=lambda x: (np.isnan(x[RANK_KEY]), x[RANK_KEY]))
    return rows


def print_and_save_table(rows: list[dict], out_dir: str) -> None:
    labels = {k: lbl for k, lbl in HEADLINE}
    headers = ['rank', 'config'] + [labels[k] for k, _ in HEADLINE]
    widths = [max(len(h), 12) for h in headers]

    shapes = sorted({r['shape'] for r in rows if r['shape']})
    phase = f'track phase: {", ".join(shapes)}' if shapes else 'track phase'
    print(f'\nController comparison ({phase}, ranked by tracking error):\n')
    print('  '.join(h.ljust(w) for h, w in zip(headers, widths)))
    print('  '.join('-' * w for w in widths))
    for i, row in enumerate(rows, 1):
        cells = [str(i), row['config']] + [f'{row[k]:.4g}' for k, _ in HEADLINE]
        print('  '.join(c.ljust(w) for c, w in zip(cells, widths)))

    csv_path = os.path.join(out_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'config'] + [k for k, _ in HEADLINE])
        for i, row in enumerate(rows, 1):
            writer.writerow([i, row['config']] + [row[k] for k, _ in HEADLINE])
    print(f'\nsummary.csv -> {csv_path}')


def make_charts(rows: list[dict], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f'[charts skipped: matplotlib unavailable: {exc}]')
        return

    configs = [r['config'] for r in rows]
    colors = [tuple(c / 255 for c in PALETTE[i % len(PALETTE)])
              for i in range(len(configs))]
    metrics_to_plot = [
        ('track.int_err_norm', 'Tracking error  int|e| (m·s)'),
        ('track.rmse_norm', 'RMSE |e| (m)'),
        ('track.int_thrust_abs', 'Control effort  int|thrust|'),
        ('track.int_torque_norm', 'Control effort  int|torque| (N·m·s)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.flat, metrics_to_plot):
        vals = [r[key] for r in rows]
        ax.bar(configs, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
    shapes = sorted({r['shape'] for r in rows if r['shape']})
    fig.suptitle('OOT controller gain comparison — tracking phase'
                 + (f' ({", ".join(shapes)})' if shapes else ''), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(out_dir, 'summary.png')
    fig.savefig(png, dpi=130)
    print(f'summary.png -> {png}')


def make_overlay(out_dir: str) -> None:
    try:
        try:
            from rerun_sdk import rerun as rr
        except ImportError:
            import rerun as rr
    except Exception as exc:  # pragma: no cover
        print(f'[overlay skipped: rerun unavailable: {exc}]')
        return

    trajs = sorted(glob.glob(os.path.join(out_dir, '*', 'traj.npz')))
    if not trajs:
        print('[overlay skipped: no <config>/traj.npz found]')
        return

    rr.init('ctrl_test_comparison')
    rr.save(os.path.join(out_dir, 'comparison.rrd'))

    planned_logged = False
    for i, path in enumerate(trajs):
        data = np.load(path)
        config = os.path.basename(os.path.dirname(path))
        color = list(PALETTE[i % len(PALETTE)])
        odom = data['odom']
        if odom.size:
            rr.log(f'world/actual/{config}',
                   rr.LineStrips3D([odom[:, 1:4]], colors=[color], radii=[0.005]),
                   static=True)
        if not planned_logged:
            # The planned path is identical across configs (same commands), so
            # log one shared reference: the recorded setpoint + the analytic
            # reference trajectory (figure-8 / spiral) stored with the run.
            setp = data['setpoint']
            if setp.size:
                rr.log('world/planned_setpoint',
                       rr.LineStrips3D([setp[:, 1:4]], colors=[[180, 180, 180]],
                                       radii=[0.005]), static=True)
            reference = data['reference'] if 'reference' in data.files \
                else np.empty((0, 4))
            if reference.size:
                rr.log('world/reference_trajectory',
                       rr.LineStrips3D([reference[:, 1:4]],
                                       colors=[[120, 120, 120]], radii=[0.003]),
                       static=True)
            planned_logged = True

    print(f'comparison.rrd -> {os.path.join(out_dir, "comparison.rrd")}')


def main() -> int:
    args = parse_args()
    out_dir = os.path.abspath(args.output_dir)
    runs = load_runs(out_dir)
    if not runs:
        print(f'no <config>/metrics.json found in {out_dir}')
        return 1

    rows = build_table(runs)
    print_and_save_table(rows, out_dir)
    make_charts(rows, out_dir)
    make_overlay(out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
