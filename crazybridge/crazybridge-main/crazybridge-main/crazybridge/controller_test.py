"""Automated OOT-controller comparison flight test.

Flies a fixed-duration sequence against the ``crazybridge`` node:

    takeoff -> regulate to a fixed point -> track a 3D figure-8 / expanding
    spiral (uploaded polynomial trajectory) -> land

The tracking trajectory is generated on the host by :mod:`crazybridge.trajectory`
(peak speed ~1 m/s by default), uploaded to the Crazyflie's trajectory memory
*before takeoff* via the bridge's ``upload_trajectory`` service, and executed by
the firmware's high-level commander when the tracking phase starts.

Every phase has a fixed duration, so the total test is identical across runs and
the integral metrics are computed over aligned time windows. While flying, the
node records the control signals (``thrust``, ``torque``), the position error
(``pos_error``), the flown path (``odometry``, via the base node's hook) and the
planned reference path (``setpoint`` = firmware ``ctrltarget``). When the
sequence finishes it hands the samples to :mod:`crazybridge.metrics`, which
writes one folder per run::

    <output_dir>/<config_name>/metrics.json   # metrics + the gains flown
                              /metrics.csv
                              /traj.npz
                              /trajectory.csv # the polynomial pieces that were uploaded
                              /pid.conf       # verbatim copy of the gain file

The filenames are identical across runs -- the folder name carries the config.
``pid_conf_path`` should be the same gain file the bridge was launched with; the
node parses it (via :class:`crazybridge.pid_conf.PidConf`) and stores the gains
in ``metrics.json`` so results are never orphaned from the gains behind them.

Ledger: unless ``ledger_enabled`` is false, the run is also registered in the
ledger experiment log (``ledger_experiment``) as a ``pending`` run before
takeoff. On success its metrics are added as numeric results, every file in the
run folder is attached and the outcome becomes ``success``; if the sequence
fails the outcome becomes ``failure`` with the error in the notes. Ledger is
strictly best-effort -- if it is down, slow or erroring, the flight, the local
report files and the exit code are unaffected and only warnings are logged. The
API URL / token come from ``ledger_url`` or ``LEDGER_API_URL`` and
``LEDGER_API_TOKEN``. See :mod:`crazybridge.ledger`.

Visualisation is left to the ``rerun`` node (launched alongside), which draws the
planned-vs-actual paths live from the same ``setpoint`` / ``odometry`` topics.

The node exits when the sequence + reporting finish, so an orchestrator can
relaunch it for the next config. Service clients, request builders, the
odometry subscription and the parameter accessors all come from
:class:`crazybridge.interface.BridgeClientNode`.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime
from time import monotonic, sleep

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PointStamped, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from crazybridge import metrics, trajectory
from crazybridge.interface import BridgeClientNode, Topics
from crazybridge.ledger import LedgerClient
from crazybridge.pid_conf import PidConf


class ControllerTest(BridgeClientNode):
    def __init__(self) -> None:
        # Base declares the service-name + odom_topic params, builds the service
        # clients, subscribes to odometry (-> on_odometry hook below) and gives
        # us _sp/_dp/_bp + the request builders.
        super().__init__('controller_test')

        # identity / output
        self.declare_parameter('config_name', 'default')
        self.declare_parameter('output_dir', '/tmp/ctrl_test')
        # The same pid.conf the bridge loaded; recorded alongside the metrics so
        # a run's numbers can be traced back to the gains that produced them.
        self.declare_parameter('pid_conf_path', '')

        # telemetry topic names (odom_topic comes from the base)
        self.declare_parameter('setpoint_topic', Topics.SETPOINT)
        self.declare_parameter('thrust_topic', Topics.THRUST)
        self.declare_parameter('torque_topic', Topics.TORQUE)
        self.declare_parameter('pos_error_topic', Topics.POS_ERROR)

        # regulate point (also the tracking trajectory's start point)
        self.declare_parameter('takeoff_height_m', 1.0)
        self.declare_parameter('goal_xyz', [0.5, 0.0, 1.0])

        # tracking trajectory (see crazybridge.trajectory.TrajectorySpec)
        self.declare_parameter('track_shape', 'figure8')        # figure8 | spiral
        self.declare_parameter('track_peak_speed_mps', 1.0)
        self.declare_parameter('track_laps', 2)
        self.declare_parameter('track_ramp_s', 2.0)             # speed ramp up/down
        self.declare_parameter('track_n_pieces', 30)            # firmware max 31
        self.declare_parameter('track_timescale', 1.0)          # >1 = slower
        self.declare_parameter('figure8_amplitude_xyz', [0.6, 0.3, 0.3])
        self.declare_parameter('spiral_radius_m', 0.8)
        self.declare_parameter('spiral_ascent_m', 0.4)
        self.declare_parameter('trajectory_id', 1)
        # Reject trajectories that leave this box around the origin. Keep it equal
        # to the bridge's max_distance_from_origin (which drops "bad" markers).
        self.declare_parameter('max_distance_from_origin_m', 3.0)

        # fixed phase durations (seconds) -> identical total across configs
        self.declare_parameter('takeoff_duration_s', 3.0)
        self.declare_parameter('settle_s', 2.0)
        self.declare_parameter('goto_duration_s', 3.0)
        self.declare_parameter('regulate_s', 3.0)
        self.declare_parameter('track_settle_s', 2.0)           # hold after tracking
        self.declare_parameter('land_duration_s', 3.0)
        self.declare_parameter('upload_timeout_s', 60.0)

        self.declare_parameter('startup_timeout_s', 30.0)

        # ledger experiment log (best-effort; see module docstring)
        self.declare_parameter('ledger_enabled', True)
        self.declare_parameter('ledger_url', '')   # '' -> LEDGER_API_URL / default
        self.declare_parameter('ledger_experiment',
                               'Crazyflie OOT Controller Comparison')
        self.declare_parameter('ledger_timeout_s', 3.0)
        self._ledger: LedgerClient | None = None
        self._ledger_run: dict | None = None
        self._ledger_experiment_id: int | None = None

        # Set once takeoff is commanded; gates the emergency land + kill so a
        # pre-flight failure (bad params, failed upload) never latches the
        # Crazyflie into emergency stop while it is still on the ground.
        self.airborne = False
        self._track: dict = {}

        # sample buffers (timestamp = node-clock seconds)
        self._lock = threading.Lock()
        self._thrust: list[list[float]] = []
        self._torque: list[list[float]] = []
        self._pos_err: list[list[float]] = []
        self._odom: list[list[float]] = []
        self._setpoint: list[list[float]] = []

        self.create_subscription(
            Float32, self._sp('thrust_topic'), self._thrust_cb, 50)
        self.create_subscription(
            Vector3, self._sp('torque_topic'), self._torque_cb, 50)
        self.create_subscription(
            Vector3, self._sp('pos_error_topic'), self._pos_err_cb, 50)
        self.create_subscription(
            PointStamped, self._sp('setpoint_topic'), self._setpoint_cb, 50)

    # -- helpers ---------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _fp_array(self, name: str) -> list[float]:
        return list(self.get_parameter(name).get_parameter_value().double_array_value)

    # -- sample callbacks ------------------------------------------------------
    def _thrust_cb(self, msg: Float32) -> None:
        with self._lock:
            self._thrust.append([self._now(), float(msg.data)])

    def _torque_cb(self, msg: Vector3) -> None:
        with self._lock:
            self._torque.append([self._now(), msg.x, msg.y, msg.z])

    def _pos_err_cb(self, msg: Vector3) -> None:
        with self._lock:
            self._pos_err.append([self._now(), msg.x, msg.y, msg.z])

    def _setpoint_cb(self, msg: PointStamped) -> None:
        p = msg.point
        with self._lock:
            self._setpoint.append([self._now(), p.x, p.y, p.z])

    def on_odometry(self, msg: Odometry) -> None:
        # Base already stashed the latest message; we additionally record the
        # flown path for the trajectory overlay.
        p = msg.pose.pose.position
        with self._lock:
            self._odom.append([self._now(), p.x, p.y, p.z])

    # -- blocking service call -------------------------------------------------
    def _call(self, client, request, label: str, timeout: float = 15.0):
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'{label}: service unavailable')
        future = client.call_async(request)
        deadline = monotonic() + timeout
        while not future.done():
            if monotonic() > deadline:
                raise RuntimeError(f'{label}: timed out')
            sleep(0.02)
        resp = future.result()
        if not getattr(resp, 'success', True):
            raise RuntimeError(f'{label}: failed ({resp.message})')
        self.get_logger().info(f'{label}: ok')
        return resp

    # -- the flight sequence ---------------------------------------------------
    def run_sequence(self) -> dict:
        """Execute the fixed-duration sequence, returning phase timestamps."""
        self.get_logger().info('waiting for services and odometry...')
        self.wait_until_ready(self._dp('startup_timeout_s'))
        self.get_logger().info('services + odometry ready')

        height = self._dp('takeoff_height_m')
        goal = self._fp_array('goal_xyz')
        if len(goal) < 3:
            goal = [0.5, 0.0, height]

        t_takeoff = self._dp('takeoff_duration_s')
        t_settle = self._dp('settle_s')
        t_goto = self._dp('goto_duration_s')
        t_regulate = self._dp('regulate_s')
        t_track_settle = self._dp('track_settle_s')
        t_land = self._dp('land_duration_s')
        timescale = self._dp('track_timescale')
        traj_id = self._ip('trajectory_id')

        # Pre-flight: build the tracking trajectory and put it in the Crazyflie's
        # memory while it is still on the ground, so the (multi-second, radio
        # heavy) upload neither competes with flight telemetry nor counts toward
        # the metrics. Any failure here raises before the drone ever moves.
        spec = self._track_spec(goal)
        pieces, traj_duration = trajectory.generate(spec)
        st = trajectory.stats(spec)
        self.get_logger().info(
            f'tracking trajectory: {spec.shape}, {st["n_pieces"]} pieces, '
            f'{traj_duration:.2f} s x timescale {timescale:g}, '
            f'peak v={st["peak_speed_mps"] / timescale:.2f} m/s '
            f'a={st["peak_accel_mps2"] / timescale ** 2:.2f} m/s^2, '
            f'fit error {st["max_fit_error_m"] * 1e3:.3f} mm')
        self._check_extent(st)
        self._call(self.upload_traj_cli,
                   self.upload_trajectory_request(pieces, traj_id),
                   'upload_trajectory', timeout=self._dp('upload_timeout_s'))
        self._track = {'spec': spec, 'pieces': pieces, 'stats': st,
                       'timescale': timescale}

        ts: dict[str, float] = {'t_start': self._now()}

        # Phase 1: takeoff to `height` above current xy, then hold.
        self.airborne = True
        self._call(self.takeoff_cli,
                   self.takeoff_request(height, t_takeoff), 'takeoff')
        sleep(t_takeoff + t_settle)
        ts['t_takeoff_end'] = self._now()

        # Phase 2: regulate to the predetermined point (absolute go_to), hold.
        self._call(self.goto_cli,
                   self.goto_request(goal[0], goal[1], goal[2], 0.0, t_goto,
                                     relative=False), 'go_to')
        sleep(t_goto + t_regulate)
        ts['t_regulate_end'] = self._now()

        # Phase 3: track the uploaded trajectory. It starts at `goal` (absolute
        # coordinates, relative=False) from rest, where phase 2 left the drone.
        # `timescale` > 1 stretches every piece's duration in the firmware.
        self._call(self.start_traj_cli,
                   self.start_trajectory_request(traj_id, timescale),
                   'start_trajectory')
        ts['t_track_start'] = self._now()
        sleep(traj_duration * timescale + t_track_settle)
        ts['t_track_end'] = self._now()

        # Phase 4: land.
        self._call(self.land_cli, self.land_request(0.0, t_land), 'land')
        sleep(t_land)
        ts['t_land_end'] = self._now()

        self._goal = goal
        return ts

    def _track_spec(self, goal) -> trajectory.TrajectorySpec:
        return trajectory.TrajectorySpec(
            shape=self._sp('track_shape'),
            goal=tuple(float(v) for v in goal[:3]),
            peak_speed_mps=self._dp('track_peak_speed_mps'),
            laps=self._ip('track_laps'),
            ramp_s=self._dp('track_ramp_s'),
            n_pieces=self._ip('track_n_pieces'),
            figure8_amplitude_xyz=tuple(self._fp_array('figure8_amplitude_xyz')),
            spiral_radius_m=self._dp('spiral_radius_m'),
            spiral_ascent_m=self._dp('spiral_ascent_m'),
        )

    def _check_extent(self, st: dict) -> None:
        """Refuse to fly a trajectory that leaves the mocap safety box."""
        limit = self._dp('max_distance_from_origin_m')
        worst = max(abs(v) for v in st['bbox_min'] + st['bbox_max'])
        if worst > limit:
            raise RuntimeError(
                f'trajectory reaches {worst:.2f} m from the origin on some axis, '
                f'beyond the {limit:.2f} m safety limit '
                f'(bbox {st["bbox_min"]} .. {st["bbox_max"]})')

    def emergency_stop(self) -> None:
        try:
            self._call(self.land_cli, self.land_request(0.0, 2.0),
                       'emergency-land', timeout=6.0)
            sleep(2.5)
        except Exception as exc:
            self.get_logger().error(f'emergency land failed: {exc}')
        try:
            self._call(self.kill_cli, self.bool_request(True), 'kill', timeout=4.0)
        except Exception as exc:
            self.get_logger().error(f'kill failed: {exc}')

    # -- reporting -------------------------------------------------------------
    def _samples(self) -> dict:
        with self._lock:
            return {
                'thrust': np.array(self._thrust, dtype=float) if self._thrust
                else np.empty((0, 2)),
                'torque': np.array(self._torque, dtype=float) if self._torque
                else np.empty((0, 4)),
                'pos_err': np.array(self._pos_err, dtype=float) if self._pos_err
                else np.empty((0, 4)),
            }

    def _traj(self, ts: dict) -> dict:
        with self._lock:
            odom = np.array(self._odom, dtype=float) if self._odom \
                else np.empty((0, 4))
            setp = np.array(self._setpoint, dtype=float) if self._setpoint \
                else np.empty((0, 4))
        out = {
            'odom': odom, 'setpoint': setp,
            'goal': getattr(self, '_goal', [0.0, 0.0, 0.0]),
        }
        if self._track:
            spec = self._track['spec']
            scale = self._track['timescale']
            # Analytic reference on the same node clock as odom/setpoint, so the
            # overlay can compare them directly.
            ref = trajectory.sample_reference(spec, dt=0.02, timescale=scale)
            ref[:, 0] += ts['t_track_start']
            out.update({
                'shape': spec.shape,
                'spec': spec.to_dict(),
                'stats': self._track['stats'],
                'timescale': scale,
                'reference': ref,
                'pieces': self._track['pieces'],
            })
        return out

    def _gains(self) -> tuple[dict | None, str]:
        """The gains this run flew with, as ``(flat dict | None, path)``."""
        path = self._sp('pid_conf_path')
        if not path:
            self.get_logger().warning(
                'pid_conf_path not set; run will be recorded without gains')
            return None, ''
        try:
            conf = PidConf.load(path)
        except Exception as exc:
            self.get_logger().error(f'could not read gains from {path}: {exc}')
            return None, path
        return conf.as_dict(), path

    def report(self, ts: dict) -> dict:
        m = metrics.compute_metrics(self._samples(), ts)
        gains, pid_conf_path = self._gains()
        config = self._sp('config_name')
        out_dir = self._sp('output_dir')
        metrics.write_reports(out_dir, config, ts, m, self._traj(ts),
                              gains=gains, pid_conf_path=pid_conf_path)
        self.get_logger().info(
            f'wrote run artefacts to {metrics.run_dir(out_dir, config)}')
        return m

    # -- ledger (best-effort: never raises, never affects the flight) -----------
    def _ledger_params(self) -> dict:
        """What this run tested: identity, trajectory, durations and gains."""
        params: dict = {'config_name': self._sp('config_name')}
        gains, pid_conf_path = self._gains()
        if pid_conf_path:
            params['pid_conf_path'] = os.path.abspath(pid_conf_path)
        # Mixed int / float / str parameters, so read the typed value directly.
        for name in ('takeoff_height_m', 'takeoff_duration_s', 'settle_s',
                     'goto_duration_s', 'regulate_s', 'track_settle_s',
                     'land_duration_s', 'track_shape', 'track_peak_speed_mps',
                     'track_laps', 'track_ramp_s', 'track_n_pieces',
                     'track_timescale', 'spiral_radius_m', 'spiral_ascent_m'):
            params[name] = self.get_parameter(name).value
        for name in ('goal_xyz', 'figure8_amplitude_xyz'):
            vec = self._fp_array(name)
            if len(vec) >= 3:
                params[name] = ', '.join(f'{v:g}' for v in vec[:3])
        for name, value in (gains or {}).items():
            params[f'gain.{name}'] = value
        return params

    def ledger_register(self) -> None:
        """Register this run as ``pending`` in ledger, before takeoff."""
        try:
            if not self._bp('ledger_enabled'):
                return
            client = LedgerClient.from_env(
                self._sp('ledger_url') or None,
                timeout_s=self._dp('ledger_timeout_s'),
                log=self.get_logger().warning)
            exp_id = client.ensure_experiment(
                self._sp('ledger_experiment'),
                subject='Crazyflie OOT controller on hardware (OptiTrack)',
                objective='Compare OOT controller gain sets on an identical '
                          'takeoff / regulate / tracking-trajectory / land '
                          'sequence.',
                description='One run per gain config, flown by '
                            'crazybridge controller_test.')
            if exp_id is None:
                return
            config = self._sp('config_name')
            stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            run = client.start_run(exp_id, f'{config} ({stamp})',
                                   self._ledger_params())
            if run is None:
                return
            self._ledger, self._ledger_run = client, run
            self._ledger_experiment_id = exp_id
            self.get_logger().info(
                f'ledger: registered run #{run.get("seq")} (id {run.get("id")}) '
                f'in experiment #{exp_id}')
        except Exception as exc:
            self.get_logger().warning(f'ledger registration failed: {exc}')

    def ledger_finish(self, m: dict | None = None,
                      error: str | None = None) -> None:
        """Close out the ledger run: results + files + outcome (or failure)."""
        if self._ledger is None or self._ledger_run is None:
            return
        run, self._ledger_run = self._ledger_run, None  # finish at most once
        try:
            run_id = run['id']
            if error is not None:
                self._ledger.update_run(
                    run_id, outcome='failure',
                    notes=f'test sequence failed: {error}')
            else:
                if m is not None:
                    self._ledger.add_results(run_id, metrics.ledger_results(m))
                dest = metrics.run_dir(self._sp('output_dir'),
                                       self._sp('config_name'))
                files = sorted(
                    os.path.join(dest, f) for f in os.listdir(dest)
                    if os.path.isfile(os.path.join(dest, f))
                ) if os.path.isdir(dest) else []
                attached = self._ledger.attach_files(run_id, files)
                names = ', '.join(os.path.basename(f) for f in files)
                notes = f'run folder: {os.path.abspath(dest)}\nfiles: {names}'
                if files and attached is None:
                    notes += '\n(attaching files to ledger failed)'
                self._ledger.update_run(run_id, outcome='success', notes=notes)
            if self._ledger.available:
                self.get_logger().info(
                    f'ledger: closed run #{run.get("seq")} (id {run_id}) '
                    f'in experiment #{self._ledger_experiment_id}')
        except Exception as exc:
            self.get_logger().warning(f'ledger update failed: {exc}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerTest()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    exit_code = 0
    try:
        # Best-effort and bounded by ledger_timeout_s; never raises.
        node.ledger_register()
        ts = node.run_sequence()
        m = node.report(ts)          # local files first, always
        node.ledger_finish(m=m)      # then ledger; never raises
        trk = m['track']
        node.get_logger().info(
            f'DONE [{node._sp("config_name")}] track ({node._sp("track_shape")}): '
            f'int|thrust|={trk["int_thrust_abs"]:.3f} '
            f'int|torque|={trk["int_torque_norm"]:.4f} '
            f'int|pos_err|={trk["int_err_norm"]:.4f} '
            f'RMSE={trk["rmse_norm"]:.4f}')
    except Exception as exc:
        node.get_logger().error(f'test sequence failed: {exc}')
        if node.airborne:
            node.emergency_stop()
        else:
            node.get_logger().warning(
                'failed before takeoff: not sending land/kill (the Crazyflie '
                'was never commanded to fly)')
        node.ledger_finish(error=str(exc))
        exit_code = 1
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
