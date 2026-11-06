#!/usr/bin/env python3
"""Orchestrate the OOT-controller gain comparison across configs.

Relaunches ``crazybridge controller_test.launch.py`` once per gain config,
pausing for the operator between runs (hardware safety: swap battery, re-place
the drone, check props). Each run writes a ``<config>/`` folder (``metrics.json``,
``metrics.csv``, ``traj.npz``, ``trajectory.csv``, ``pid.conf``) into a shared
``--output-dir``; afterwards run ``summarize.py`` on that directory for the
comparison table + overlay.

Example
-------
    # source your workspace first: source install/setup.bash
    ./run_comparison.py \
        --output-dir ~/ctrl_results \
        --config baseline=src/crazybridge/config/pid.conf \
        --config gains_v1=src/crazybridge/config/pid.conf.v1 \
        --config gains_sim=src/crazybridge/config/pid.conf.sim \
        --launch-arg rerun_mode:=connect \
        --launch-arg rerun_addr:=rerun+http://10.43.100.150:9876/proxy

For a CrazySim dry run add:
    --launch-arg uri:=udp://127.0.0.1:19850 --launch-arg use_optitrack:=false

The tracking trajectory is chosen with launch args, identical for every config:
    --launch-arg track_shape:=figure8      # or spiral
    --launch-arg track_peak_speed_mps:=1.0
    --launch-arg track_timescale:=2.0      # first hardware flight: half speed
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--config', action='append', default=[], metavar='NAME=PID_CONF_PATH',
        help='A config to test: label=path/to/pid.conf (repeatable).')
    p.add_argument('--output-dir', required=True,
                   help='Shared directory for all per-config results.')
    p.add_argument(
        '--launch-arg', action='append', default=[], metavar='name:=value',
        help='Extra arg passed verbatim to ros2 launch (repeatable), e.g. '
             'track_shape:=spiral, track_timescale:=2.0 or rerun_addr:=...')
    p.add_argument('--no-prompt', action='store_true',
                   help='Do not pause for the operator between runs (sim only!).')
    p.add_argument('--launch-file', default='controller_test.launch.py')
    p.add_argument('--package', default='crazybridge')
    return p.parse_args()


def parse_configs(items: list[str]) -> list[tuple[str, str]]:
    configs = []
    for item in items:
        if '=' not in item:
            sys.exit(f'error: --config must be NAME=PATH, got {item!r}')
        name, path = item.split('=', 1)
        if not os.path.isfile(path):
            sys.exit(f'error: pid.conf for {name!r} not found: {path}')
        configs.append((name, os.path.abspath(path)))
    if not configs:
        sys.exit('error: provide at least one --config NAME=PATH')
    return configs


def confirm(prompt: str) -> bool:
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        return False
    return ans not in ('q', 'quit', 'n', 'no', 'abort')


def run_config(name: str, pid_conf: str, out_dir: str, launch_file: str,
               package: str, extra: list[str]) -> int:
    cmd = [
        'ros2', 'launch', package, launch_file,
        f'config_name:={name}',
        f'pid_conf_path:={pid_conf}',
        f'output_dir:={out_dir}',
        *extra,
    ]
    print('\n' + '=' * 70)
    print(f'RUN: {name}')
    print('  ' + ' '.join(cmd))
    print('=' * 70, flush=True)
    return subprocess.call(cmd)


def main() -> int:
    args = parse_args()
    configs = parse_configs(args.config)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Comparing {len(configs)} config(s); results -> {out_dir}')
    for name, path in configs:
        print(f'  - {name}: {path}')

    completed, failed = [], []
    for i, (name, pid_conf) in enumerate(configs, 1):
        if not args.no_prompt:
            ok = confirm(
                f'\n[{i}/{len(configs)}] Ready to fly config {name!r}? '
                'Place the drone, check battery/props, then press Enter '
                "(or 'q' to abort): ")
            if not ok:
                print('aborted by operator.')
                break
        rc = run_config(name, pid_conf, out_dir, args.launch_file,
                        args.package, args.launch_arg)
        if rc == 0:
            completed.append(name)
        else:
            failed.append((name, rc))
            print(f'!! config {name!r} exited with code {rc}')
            if not args.no_prompt and not confirm(
                    'Continue with the next config anyway? (Enter=yes, q=stop): '):
                break

    print('\n' + '=' * 70)
    print(f'done: {len(completed)} ok, {len(failed)} failed')
    if failed:
        for name, rc in failed:
            print(f'  FAILED {name} (rc={rc})')
    print(f'\nNext: python3 summarize.py --output-dir {out_dir}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
