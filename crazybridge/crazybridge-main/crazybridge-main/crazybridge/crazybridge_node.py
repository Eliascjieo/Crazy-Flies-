"""ROS2 bridge for a Crazyflie with Multi-Drone RigidBody & Velocity support."""
from __future__ import annotations

import math
import os
import threading
from logging import getLogger
from time import sleep, time

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement, Poly4D

from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Header
from std_srvs.srv import SetBool

from crazybridge_interfaces.srv import (
    GoTo, Land, Spiral, StartTrajectory, Takeoff, UploadTrajectory)
from crazybridge.interface import BridgePublishers, Create
from crazybridge.pid_conf import PidConf

logger = getLogger()

PARAM_HL_COMMANDER = 'commander.enHighLevel'
PARAM_ESTIMATOR = 'stabilizer.estimator'
PARAM_CONTROLLER = 'stabilizer.controller'
ESTIMATOR_KALMAN = 2
ESTIMATOR_COMPLEMENTARY = 1

MAX_TRAJ_PIECES = 31
NUM_TRAJ_SLOTS = 31
TRAJ_COEFFS = 8


class CrazyBridge(Node):
    def __init__(self) -> None:
        super().__init__('crazybridge')

        self.declare_parameter('uri', 'udp://127.0.0.1:19850')
        self.declare_parameter('rigid_body_name', '')
        self.declare_parameter('send_full_pose', False)
        self.declare_parameter('log_period_ms', 50)
        self.declare_parameter('oot_log_period_ms', 10)
        self.declare_parameter('connection_timeout_s', 10.0)
        self.declare_parameter('odom_frame_id', 'world')
        self.declare_parameter('child_frame_id', 'crazyflie')
        self.declare_parameter('pid_conf_path', '')
        self.declare_parameter('load_pid_conf', True)
        self.declare_parameter('max_distance_from_origin', 3.0)

        self._uri = self.get_parameter('uri').get_parameter_value().string_value
        self._rigid_body_name = (
            self.get_parameter('rigid_body_name').get_parameter_value().string_value.strip()
        )
        self._send_full_pose = (
            self.get_parameter('send_full_pose').get_parameter_value().bool_value
        )
        self._is_sim = self._uri.startswith('udp://')

        self._log_period_ms = self._sanitise_log_period(
            self.get_parameter('log_period_ms').get_parameter_value().integer_value,
            'log_period_ms',
        )
        self._max_distance_from_origin = (
            self.get_parameter('max_distance_from_origin').get_parameter_value().double_value
        )
        self._oot_log_period_ms = self._sanitise_log_period(
            self.get_parameter('oot_log_period_ms').get_parameter_value().integer_value,
            'oot_log_period_ms',
        )
        self._odom_frame = (
            self.get_parameter('odom_frame_id').get_parameter_value().string_value
        )
        self._child_frame = (
            self.get_parameter('child_frame_id').get_parameter_value().string_value
        )
        self._pid_conf_path = self._resolve_pid_conf_path()
        self._load_pid_conf = bool(
            self.get_parameter('load_pid_conf').get_parameter_value().bool_value
        )

        self._pos = [0.0, 0.0, 0.0]
        self._quat = [0.0, 0.0, 0.0, 1.0]
        self._pos_lock = threading.Lock()

        self._oot_lock = threading.Lock()
        self._oot_pos_err = (0.0, 0.0, 0.0)
        self._oot_q_err = (0.0, 0.0, 0.0, 1.0)
        self._oot_qd = (0.0, 0.0, 0.0, 1.0)

        self._marker_id: str | None = self._rigid_body_name if self._rigid_body_name else None
        self._safe_to_fly: bool = False
        self._bad_marker_counter: int = 0
        self._last_seen_marker: float = 0.0
        self._safe_to_fly_lock = threading.Lock()

        self._pubs = BridgePublishers(self)

        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        if self._rigid_body_name:
            self.get_logger().info(
                f'Subscribing to /optitrack/rigid_body for Motive Rigid Body "{self._rigid_body_name}"'
            )
            self._rigid_sub = self.create_subscription(
                PoseStamped, Create.RIGID_BODY, self._rigid_body_cb, qos_best_effort
            )
        else:
            self.get_logger().info(
                'No rigid_body_name set; subscribing to /optitrack/marker (first-seen marker)'
            )
            self._marker_sub = self.create_subscription(
                PointStamped, Create.MARKER, self._marker_cb, qos_best_effort
            )

        # Continuous 3D velocity setpoint subscriber for Q-Table + Lyapunov APF control
        self._cmd_vel_sub = self.create_subscription(
            Vector3, Create.CMD_VEL, self._cmd_vel_cb, 10
        )

        self._takeoff_srv = None
        if self._is_sim:
            self._safe_to_fly = True
            self._takeoff_srv = self.create_service(
                Takeoff, Create.SRV_TAKEOFF, self._takeoff_cb
            )
            self.get_logger().info('Simulation URI detected (udp://): takeoff service enabled immediately')

        self._land_srv = self.create_service(Land, Create.SRV_LAND, self._land_cb)
        self._goto_srv = self.create_service(GoTo, Create.SRV_GOTO, self._goto_cb)
        self._spiral_srv = self.create_service(Spiral, Create.SRV_SPIRAL, self._spiral_cb)

        self._traj_group = MutuallyExclusiveCallbackGroup()
        self._upload_traj_srv = self.create_service(
            UploadTrajectory, Create.SRV_UPLOAD_TRAJ, self._upload_traj_cb,
            callback_group=self._traj_group)
        self._start_traj_srv = self.create_service(
            StartTrajectory, Create.SRV_START_TRAJ, self._start_traj_cb)
        self._kill_srv = self.create_service(SetBool, Create.SRV_KILL, self._kill_cb)

        cflib.crtp.init_drivers()
        self._connected = threading.Event()
        self._connect_failed = threading.Event()
        self._connect_error: str | None = None

        self._cf = Crazyflie(rw_cache='./cache')
        self._cf.fully_connected.add_callback(self._on_connected)
        self._cf.disconnected.add_callback(self._on_disconnected)
        self._cf.connection_failed.add_callback(self._on_connection_failed)
        self._cf.connection_lost.add_callback(self._on_connection_lost)
        self._cf.console.receivedChar.add_callback(self._on_console)

        self.get_logger().info(f'Opening link to {self._uri}')
        self._cf.open_link(self._uri)

        timeout = float(
            self.get_parameter('connection_timeout_s').get_parameter_value().double_value
        )
        if not self._connected.wait(timeout=timeout):
            self._cf.close_link()
            raise RuntimeError(
                f'Timed out connecting to {self._uri} after {timeout}s '
                f'({self._connect_error or "no failure callback fired"})'
            )
        self._safety_interval = self.create_timer(1.0, self._safety_check)

    def _on_console(self, text):
        self.get_logger().info(text)

    def _sanitise_log_period(self, value: int, name: str) -> int:
        v = int(value)
        if v < 10:
            v = 10
        if v % 10 != 0:
            v = (v // 10) * 10 or 10
        if v > 2540:
            v = 2540
        return v

    def _resolve_pid_conf_path(self) -> str:
        path = self.get_parameter('pid_conf_path').get_parameter_value().string_value
        if path:
            return path
        return os.path.join(
            get_package_share_directory('crazybridge'), 'config', 'pid.conf'
        )

    def _on_connected(self, link_uri: str) -> None:
        self.get_logger().info(f'Connected to {link_uri}')
        if self._load_pid_conf:
            try:
                self._apply_pid_conf(self._pid_conf_path)
            except Exception as exc:
                self.get_logger().error(
                    f'Failed to load pid.conf from {self._pid_conf_path}: {exc}')
        try:
            self._cf.param.set_value(PARAM_HL_COMMANDER, 1)
            self._cf.param.set_value(PARAM_ESTIMATOR, ESTIMATOR_KALMAN)
        except Exception as exc:
            self.get_logger().error(f'Failed to set startup params: {exc}')

        self._pos_log = LogConfig(name='kalman_pos', period_in_ms=self._log_period_ms)
        self._pos_log.add_variable('kalman.stateX', 'float')
        self._pos_log.add_variable('kalman.stateY', 'float')
        self._pos_log.add_variable('kalman.stateZ', 'float')

        self._q_log = LogConfig(name='kalman_q', period_in_ms=self._log_period_ms)
        self._q_log.add_variable('kalman.q0', 'float')
        self._q_log.add_variable('kalman.q1', 'float')
        self._q_log.add_variable('kalman.q2', 'float')
        self._q_log.add_variable('kalman.q3', 'float')

        extra_period = self._sanitise_log_period(500, 'extra_log_ms')
        self._pm_log = LogConfig(name='battery', period_in_ms=extra_period)
        self._pm_log.add_variable('pm.batteryLevel', 'uint8_t')

        self._setpoint_log = LogConfig(name='ctrltarget', period_in_ms=self._log_period_ms)
        self._setpoint_log.add_variable('ctrltarget.x', 'float')
        self._setpoint_log.add_variable('ctrltarget.y', 'float')
        self._setpoint_log.add_variable('ctrltarget.z', 'float')

        try:
            self._cf.log.add_config(self._pos_log)
            self._cf.log.add_config(self._q_log)
            self._cf.log.add_config(self._pm_log)
            self._cf.log.add_config(self._setpoint_log)
            self._pos_log.data_received_cb.add_callback(self._on_pos_log)
            self._q_log.data_received_cb.add_callback(self._on_q_log)
            self._pm_log.data_received_cb.add_callback(self._on_pm_log)
            self._setpoint_log.data_received_cb.add_callback(self._on_setpoint_log)
            self._pos_log.start()
            self._q_log.start()
            self._pm_log.start()
            self._setpoint_log.start()
        except Exception as exc:
            self.get_logger().error(f'Failed to register log blocks: {exc}')

        self._setup_oot_logs()
        self._connected.set()

    def _setup_oot_logs(self) -> None:
        period = self._oot_log_period_ms
        self._input_log = LogConfig(name='input', period_in_ms=period)
        self._input_log.add_variable('oot.thrust', 'float')
        self._input_log.add_variable('oot.torque_x', 'float')
        self._input_log.add_variable('oot.torque_y', 'float')
        self._input_log.add_variable('oot.torque_z', 'float')

        self._pos_err_log = LogConfig(name='oot_pos_err', period_in_ms=period)
        self._pos_err_log.add_variable('oot.pos_err_x', 'float')
        self._pos_err_log.add_variable('oot.pos_err_y', 'float')
        self._pos_err_log.add_variable('oot.pos_err_z', 'float')

        self._q_err_log = LogConfig(name='oot_q_err', period_in_ms=period)
        self._q_err_log.add_variable('oot.q_err_x', 'float')
        self._q_err_log.add_variable('oot.q_err_y', 'float')
        self._q_err_log.add_variable('oot.q_err_z', 'float')
        self._q_err_log.add_variable('oot.q_err_w', 'float')

        self._qd_log = LogConfig(name='oot_qd', period_in_ms=period)
        self._qd_log.add_variable('oot.qd_x', 'float')
        self._qd_log.add_variable('oot.qd_y', 'float')
        self._qd_log.add_variable('oot.qd_z', 'float')
        self._qd_log.add_variable('oot.qd_w', 'float')

        self._ang_vel_err_log = LogConfig(name='oot_ang_vel_err', period_in_ms=period)
        self._ang_vel_err_log.add_variable('oot.ang_vel_err_x', 'float')
        self._ang_vel_err_log.add_variable('oot.ang_vel_err_y', 'float')
        self._ang_vel_err_log.add_variable('oot.ang_vel_err_z', 'float')

        try:
            self._cf.log.add_config(self._input_log)
            self._cf.log.add_config(self._pos_err_log)
            self._cf.log.add_config(self._q_err_log)
            self._cf.log.add_config(self._qd_log)
            self._cf.log.add_config(self._ang_vel_err_log)

            self._input_log.data_received_cb.add_callback(self._on_input_log)
            self._pos_err_log.data_received_cb.add_callback(self._on_pos_err_log)
            self._q_err_log.data_received_cb.add_callback(self._on_q_err_log)
            self._qd_log.data_received_cb.add_callback(self._on_qd_log)
            self._ang_vel_err_log.data_received_cb.add_callback(self._on_ang_vel_err_log)

            self._input_log.start()
            self._pos_err_log.start()
            self._q_err_log.start()
            self._qd_log.start()
            self._ang_vel_err_log.start()
            self.get_logger().info('OOT log blocks started')
        except Exception as exc:
            self.get_logger().warning(f'OOT log blocks not active (standard firmware): {exc}')

    def _on_pm_log(self, _timestamp, data, _logconf):
        batt_msg = Float32()
        batt_msg.data = float(data['pm.batteryLevel'])
        self._pubs.battery.publish(batt_msg)

    def _on_setpoint_log(self, _timestamp, data, _logconf):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._odom_frame
        msg.point.x = float(data['ctrltarget.x'])
        msg.point.y = float(data['ctrltarget.y'])
        msg.point.z = float(data['ctrltarget.z'])
        self._pubs.setpoint.publish(msg)

    def _on_input_log(self, _timestamp, data, _logconf):
        thrust_msg = Float32()
        thrust_msg.data = float(data['oot.thrust'])
        self._pubs.thrust.publish(thrust_msg)

        torque_msg = Vector3()
        torque_msg.x = float(data['oot.torque_x'])
        torque_msg.y = float(data['oot.torque_y'])
        torque_msg.z = float(data['oot.torque_z'])
        self._pubs.torque.publish(torque_msg)

    def _apply_pid_conf(self, path: str) -> None:
        conf = PidConf.load(path)
        self.get_logger().info(f'Loading PID gains from {path}')
        for name, value in conf.params():
            self._cf.param.set_value(name, value)
        self.get_logger().info('Done configuring PID')

    def _on_disconnected(self, link_uri: str) -> None:
        self.get_logger().info(f'Disconnected from {link_uri}')

    def _ingest_mocap_xyz(self, x: float, y: float, z: float,
                          qx: float = 0.0, qy: float = 0.0,
                          qz: float = 0.0, qw: float = 1.0,
                          has_quat: bool = False) -> None:
        axis = [x, y, z]
        over_limit = [abs(i) > self._max_distance_from_origin for i in axis]
        if any(over_limit):
            self._bad_marker_counter += 1
            return
        self._last_seen_marker = time()
        self._bad_marker_counter = max(0, self._bad_marker_counter - 1)
        if self._cf.is_connected():
            if has_quat and self._send_full_pose:
                self._cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
            else:
                self._cf.extpos.send_extpos(x, y, z)

    def _rigid_body_cb(self, msg: PoseStamped) -> None:
        frame_id = msg.header.frame_id.strip()
        if frame_id != self._rigid_body_name:
            return
        self._marker_id = frame_id
        p = msg.pose.position
        q = msg.pose.orientation
        self._ingest_mocap_xyz(p.x, p.y, p.z, q.x, q.y, q.z, q.w, has_quat=True)

    def _marker_cb(self, msg: PointStamped) -> None:
        header: Header = msg.header
        if self._marker_id is None:
            self._marker_id = header.frame_id
        if self._marker_id != header.frame_id:
            return
        point: Point = msg.point
        self._ingest_mocap_xyz(point.x, point.y, point.z, has_quat=False)

    def _cmd_vel_cb(self, msg: Vector3) -> None:
        """Streams 3D world velocity setpoint (vx, vy, vz, yawrate=0) to Crazyflie."""
        if not self._safe_to_fly or not self._cf.is_connected():
            return
        try:
            self._cf.commander.send_velocity_world_setpoint(
                float(msg.x), float(msg.y), float(msg.z), 0.0
            )
        except Exception as exc:
            self.get_logger().error(f'cmd_vel failed: {exc}')

    def _safety_check(self):
        if not self._cf.is_connected():
            self.get_logger().error('Disconnected from crazyflie. Trying to reconnect')
            with self._safe_to_fly_lock:
                self._cf.open_link(self._uri)
                if self._connected.wait(timeout=1):
                    self.get_logger().error('LANDING!!')
                    if self._pos[2] > 0.1:
                        self._cf.high_level_commander.land(0, 2)
                        sleep(5)
                    self._cf.supervisor.send_emergency_stop()

        if self._is_sim:
            return

        if self._marker_id is None or self._last_seen_marker == 0.0:
            return
        time_diff = time() - self._last_seen_marker
        if time_diff > 2.0 or self._bad_marker_counter > 10:
            with self._safe_to_fly_lock:
                self.get_logger().error(
                    f'Bad marker/rigid_body "{self._marker_id}": not seen for '
                    f'{time_diff:.2f}s ({self._bad_marker_counter} infractions)'
                )
                if self._safe_to_fly:
                    self._safe_to_fly = False
                    self.get_logger().error('LANDING!!')
                    self._cf.high_level_commander.land(0, 2)
                    sleep(5)
                    self._cf.supervisor.send_emergency_stop()
            return
        with self._safe_to_fly_lock:
            if not self._takeoff_srv:
                self._takeoff_srv = self.create_service(
                    Takeoff, Create.SRV_TAKEOFF, self._takeoff_cb)
                self.get_logger().info(
                    f'OptiTrack lock verified for "{self._marker_id}" -> Takeoff service READY!'
                )
            self._safe_to_fly = True

    def _on_connection_failed(self, link_uri: str, msg: str) -> None:
        self._connect_error = msg
        self.get_logger().error(f'Connection to {link_uri} failed: {msg}')
        self._connect_failed.set()
        self._connected.set()

    def _on_connection_lost(self, link_uri: str, msg: str) -> None:
        self.get_logger().warning(f'Connection to {link_uri} lost: {msg}')

    def _on_pos_log(self, _timestamp, data, _logconf) -> None:
        with self._pos_lock:
            self._pos = [
                float(data['kalman.stateX']),
                float(data['kalman.stateY']),
                float(data['kalman.stateZ']),
            ]
        self._publish_odom()

    def _on_q_log(self, _timestamp, data, _logconf) -> None:
        x, y, z, w = self._normalize_quat(
            float(data['kalman.q1']),
            float(data['kalman.q2']),
            float(data['kalman.q3']),
            float(data['kalman.q0']),
        )
        with self._pos_lock:
            self._quat = [x, y, z, w]
        self._publish_odom()

    def _on_pos_err_log(self, _timestamp, data, _logconf) -> None:
        v = (
            float(data['oot.pos_err_x']),
            float(data['oot.pos_err_y']),
            float(data['oot.pos_err_z']),
        )
        with self._oot_lock:
            self._oot_pos_err = v
        msg = Vector3(x=v[0], y=v[1], z=v[2])
        self._pubs.pos_error.publish(msg)

    def _on_q_err_log(self, _timestamp, data, _logconf) -> None:
        v = self._normalize_quat(
            float(data['oot.q_err_x']),
            float(data['oot.q_err_y']),
            float(data['oot.q_err_z']),
            float(data['oot.q_err_w']),
        )
        with self._oot_lock:
            self._oot_q_err = v
        msg = Quaternion(x=v[0], y=v[1], z=v[2], w=v[3])
        self._pubs.orientation_error.publish(msg)

    def _on_qd_log(self, _timestamp, data, _logconf) -> None:
        v = self._normalize_quat(
            float(data['oot.qd_x']),
            float(data['oot.qd_y']),
            float(data['oot.qd_z']),
            float(data['oot.qd_w']),
        )
        with self._oot_lock:
            self._oot_qd = v
        msg = Quaternion(x=v[0], y=v[1], z=v[2], w=v[3])
        self._pubs.orientation_desired.publish(msg)

    def _on_ang_vel_err_log(self, _timestamp, data, _logconf) -> None:
        pass

    def _publish_odom(self) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._child_frame
        with self._pos_lock:
            pos = tuple(self._pos)
            quat = tuple(self._quat)
        msg.pose.pose.position.x = pos[0]
        msg.pose.pose.position.y = pos[1]
        msg.pose.pose.position.z = pos[2]
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]
        self._pubs.odom.publish(msg)

    @staticmethod
    def _normalize_quat(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        return x / norm, y / norm, z / norm, w / norm

    @staticmethod
    def _duration_to_seconds(duration) -> float:
        return float(duration.sec) + float(duration.nanosec) * 1e-9

    def _takeoff_cb(
        self, request: Takeoff.Request, response: Takeoff.Response
    ) -> Takeoff.Response:
        if not self._safe_to_fly:
            self.get_logger().warning('takeoff refused: not safe to fly')
            response.success = False
            response.message = 'not safe to fly'
            return response
        duration = self._duration_to_seconds(request.duration)
        self.get_logger().info(
            f'takeoff height={request.height:.2f}m duration={duration:.2f}s '
            f'group_mask={request.group_mask}'
        )
        try:
            self._cf.high_level_commander.takeoff(float(request.height), duration)
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'takeoff failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _kill_cb(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        if not request.data:
            response.success = False
            response.message = 'kill not requested (data=false)'
            return response
        self.get_logger().info('Kill!')
        try:
            self._cf.supervisor.send_emergency_stop()
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'Kill failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _land_cb(
        self, request: Land.Request, response: Land.Response
    ) -> Land.Response:
        duration = self._duration_to_seconds(request.duration)
        self.get_logger().info(
            f'land height={request.height:.2f}m duration={duration:.2f}s '
            f'group_mask={request.group_mask}'
        )
        try:
            # Ensure low-level velocity stream is released back to high-level commander
            self._cf.commander.send_notify_setpoint_stop()
            self._cf.high_level_commander.land(
                float(request.height), duration, group_mask=int(request.group_mask)
            )
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'land failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _goto_cb(
        self, request: GoTo.Request, response: GoTo.Response
    ) -> GoTo.Response:
        duration = self._duration_to_seconds(request.duration)
        yaw_rad = math.radians(float(request.yaw))
        self.get_logger().info(
            f'go_to x={request.goal.x:.2f} y={request.goal.y:.2f} '
            f'z={request.goal.z:.2f} yaw={request.yaw:.2f}deg '
            f'duration={duration:.2f}s relative={request.relative}'
        )
        try:
            self._cf.commander.send_notify_setpoint_stop()
            self._cf.high_level_commander.go_to(
                float(request.goal.x),
                float(request.goal.y),
                float(request.goal.z),
                yaw_rad,
                duration,
                relative=bool(request.relative),
                group_mask=int(request.group_mask),
            )
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'go_to failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _spiral_cb(
        self, request: Spiral.Request, response: Spiral.Response
    ) -> Spiral.Response:
        duration = self._duration_to_seconds(request.duration)
        angle_rad = math.radians(float(request.angle))
        try:
            self._cf.high_level_commander.spiral(
                angle_rad, float(request.r0), float(request.rf),
                float(request.ascent), duration,
                sideways=bool(request.sideways),
                clockwise=bool(request.clockwise),
                group_mask=int(request.group_mask),
            )
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'spiral failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _upload_traj_cb(
        self, request: UploadTrajectory.Request,
        response: UploadTrajectory.Response
    ) -> UploadTrajectory.Response:
        n = len(request.pieces)
        try:
            if request.piece_offset != 0:
                raise ValueError('piece_offset != 0 is not supported')
            if not 1 <= n <= MAX_TRAJ_PIECES:
                raise ValueError(f'{n} pieces; firmware holds 1..{MAX_TRAJ_PIECES}')
            if request.trajectory_id >= NUM_TRAJ_SLOTS:
                raise ValueError(f'trajectory_id {request.trajectory_id} >= {NUM_TRAJ_SLOTS}')

            polys = []
            for i, piece in enumerate(request.pieces):
                axes = [list(piece.poly_x), list(piece.poly_y),
                        list(piece.poly_z), list(piece.poly_yaw)]
                if any(len(c) != TRAJ_COEFFS for c in axes):
                    raise ValueError(f'piece {i}: needs {TRAJ_COEFFS} coeffs')
                duration = self._duration_to_seconds(piece.duration)
                polys.append(Poly4D(duration, *(Poly4D.Poly(c) for c in axes)))

            mems = self._cf.mem.get_mems(MemoryElement.TYPE_TRAJ)
            if not mems:
                raise RuntimeError('Crazyflie exposes no trajectory memory')
            mem = mems[0]
            mem.trajectory = polys
            if not mem.write_data_sync():
                raise RuntimeError('trajectory memory write failed')
            self._cf.high_level_commander.define_trajectory(
                int(request.trajectory_id), 0, n)
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'upload_trajectory failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _start_traj_cb(
        self, request: StartTrajectory.Request,
        response: StartTrajectory.Response
    ) -> StartTrajectory.Response:
        try:
            self._cf.high_level_commander.start_trajectory(
                int(request.trajectory_id),
                float(request.timescale),
                relative_position=bool(request.relative),
                reversed=bool(request.reversed),
                group_mask=int(request.group_mask),
            )
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'start_trajectory failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def shutdown(self) -> None:
        for name in (
            '_pos_log', '_q_log', '_setpoint_log',
            '_pos_err_log', '_q_err_log', '_qd_log', '_ang_vel_err_log',
        ):
            log = getattr(self, name, None)
            if log is None:
                continue
            try:
                log.stop()
            except Exception:
                pass
        self._cf.close_link()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CrazyBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()