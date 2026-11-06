"""Shared topic/service names and endpoint helpers for the crazybridge stack."""
from __future__ import annotations

import threading
from time import monotonic, sleep

from rclpy.node import Node

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from std_srvs.srv import SetBool

from crazybridge_interfaces.msg import TrajectoryPolynomialPiece
from crazybridge_interfaces.srv import (
    GoTo, Land, Spiral, StartTrajectory, Takeoff, UploadTrajectory)

BRIDGE_NODE = 'crazybridge'


class Create:
    """Names the bridge node passes to create_publisher / create_service."""
    ODOM = '~/odometry'
    BATTERY = '~/battery'
    THRUST = 'thrust'
    TORQUE = 'torque'
    POS_ERROR = 'pos_error'
    SETPOINT = 'setpoint'
    ORI_DESIRED = 'orientation/desired'
    ORI_ERROR = 'orientation/error'
    MARKER = 'optitrack/marker'
    RIGID_BODY = 'optitrack/rigid_body'
    CMD_VEL = '~/cmd_vel'
    SRV_TAKEOFF = '~/takeoff'
    SRV_LAND = '~/land'
    SRV_GOTO = '~/go_to'
    SRV_SPIRAL = '~/spiral'
    SRV_UPLOAD_TRAJ = '~/upload_trajectory'
    SRV_START_TRAJ = '~/start_trajectory'
    SRV_KILL = '~/kill'


class Topics:
    """Fully-qualified topic names for clients (TUI, test node, rerun)."""
    ODOM = '/crazybridge/odometry'
    BATTERY = '/crazybridge/battery'
    THRUST = '/thrust'
    TORQUE = '/torque'
    POS_ERROR = '/pos_error'
    SETPOINT = '/setpoint'
    ORI_DESIRED = '/orientation/desired'
    ORI_ERROR = '/orientation/error'
    MARKER = '/optitrack/marker'
    RIGID_BODY = '/optitrack/rigid_body'


class Services:
    """Fully-qualified service names for clients."""
    TAKEOFF = '/crazybridge/takeoff'
    LAND = '/crazybridge/land'
    GOTO = '/crazybridge/go_to'
    SPIRAL = '/crazybridge/spiral'
    UPLOAD_TRAJ = '/crazybridge/upload_trajectory'
    START_TRAJ = '/crazybridge/start_trajectory'
    KILL = '/crazybridge/kill'


def seconds_to_duration(seconds: float) -> Duration:
    d = Duration()
    d.sec = int(seconds)
    d.nanosec = int((seconds - int(seconds)) * 1e9)
    return d


class BridgePublishers:
    """All telemetry publishers the bridge node exposes."""

    def __init__(self, node) -> None:
        self.odom = node.create_publisher(Odometry, Create.ODOM, 1)
        self.battery = node.create_publisher(Float32, Create.BATTERY, 1)
        self.thrust = node.create_publisher(Float32, Create.THRUST, 1)
        self.torque = node.create_publisher(Vector3, Create.TORQUE, 1)
        self.pos_error = node.create_publisher(Vector3, Create.POS_ERROR, 1)
        self.setpoint = node.create_publisher(PointStamped, Create.SETPOINT, 1)
        self.orientation_desired = node.create_publisher(
            Quaternion, Create.ORI_DESIRED, 1)
        self.orientation_error = node.create_publisher(
            Quaternion, Create.ORI_ERROR, 1)


class BridgeClientNode(Node):
    """Base node for clients that drive the bridge (TUI, automated test)."""

    def __init__(self, node_name: str, *, subscribe_odometry: bool = True) -> None:
        super().__init__(node_name)

        self.declare_parameter('takeoff_srv', Services.TAKEOFF)
        self.declare_parameter('land_srv', Services.LAND)
        self.declare_parameter('goto_srv', Services.GOTO)
        self.declare_parameter('spiral_srv', Services.SPIRAL)
        self.declare_parameter('upload_traj_srv', Services.UPLOAD_TRAJ)
        self.declare_parameter('start_traj_srv', Services.START_TRAJ)
        self.declare_parameter('kill_srv', Services.KILL)

        self.takeoff_cli = self.create_client(Takeoff, self._sp('takeoff_srv'))
        self.land_cli = self.create_client(Land, self._sp('land_srv'))
        self.goto_cli = self.create_client(GoTo, self._sp('goto_srv'))
        self.spiral_cli = self.create_client(Spiral, self._sp('spiral_srv'))
        self.upload_traj_cli = self.create_client(
            UploadTrajectory, self._sp('upload_traj_srv'))
        self.start_traj_cli = self.create_client(
            StartTrajectory, self._sp('start_traj_srv'))
        self.kill_cli = self.create_client(SetBool, self._sp('kill_srv'))

        self._odom_lock = threading.Lock()
        self._latest_odom: Odometry | None = None
        if subscribe_odometry:
            self.declare_parameter('odom_topic', Topics.ODOM)
            self.create_subscription(
                Odometry, self._sp('odom_topic'), self._ingest_odom, 10)

    def _sp(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _dp(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _bp(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _ip(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _ingest_odom(self, msg: Odometry) -> None:
        with self._odom_lock:
            self._latest_odom = msg
        self.on_odometry(msg)

    def on_odometry(self, msg: Odometry) -> None:
        """Hook for subclasses to record/react to odometry; default no-op."""

    def latest_odom(self) -> Odometry | None:
        with self._odom_lock:
            return self._latest_odom

    def has_odometry(self) -> bool:
        with self._odom_lock:
            return self._latest_odom is not None

    def service_clients(self) -> dict:
        return {
            'takeoff': self.takeoff_cli, 'land': self.land_cli,
            'go_to': self.goto_cli, 'spiral': self.spiral_cli,
            'upload_trajectory': self.upload_traj_cli,
            'start_trajectory': self.start_traj_cli,
            'kill': self.kill_cli
        }

    def service_ready(self) -> dict[str, bool]:
        return {k: c.service_is_ready() for k, c in self.service_clients().items()}

    def wait_until_ready(self, timeout_s: float, require_odom: bool = True) -> None:
        deadline = monotonic() + timeout_s
        while monotonic() < deadline:
            if all(self.service_ready().values()) and (
                    not require_odom or self.has_odometry()):
                return
            sleep(0.1)
        missing = [n for n, r in self.service_ready().items() if not r]
        raise RuntimeError(
            f'startup timeout after {timeout_s}s (missing services: {missing}, '
            f'odometry: {self.has_odometry()})')

    @staticmethod
    def takeoff_request(height: float, duration_s: float, group_mask: int = 0):
        req = Takeoff.Request()
        req.height = float(height)
        req.duration = seconds_to_duration(duration_s)
        req.group_mask = int(group_mask)
        return req

    @staticmethod
    def land_request(height: float, duration_s: float, group_mask: int = 0):
        req = Land.Request()
        req.height = float(height)
        req.duration = seconds_to_duration(duration_s)
        req.group_mask = int(group_mask)
        return req

    @staticmethod
    def goto_request(x: float, y: float, z: float, yaw_deg: float,
                     duration_s: float, relative: bool = False,
                     group_mask: int = 0):
        req = GoTo.Request()
        req.relative = bool(relative)
        req.goal = Point(x=float(x), y=float(y), z=float(z))
        req.yaw = float(yaw_deg)
        req.duration = seconds_to_duration(duration_s)
        req.group_mask = int(group_mask)
        return req

    @staticmethod
    def spiral_request(angle_deg: float, r0: float, rf: float, ascent: float,
                       duration_s: float, sideways: bool = False,
                       clockwise: bool = False, group_mask: int = 0):
        req = Spiral.Request()
        req.angle = float(angle_deg)
        req.r0 = float(r0)
        req.rf = float(rf)
        req.ascent = float(ascent)
        req.duration = seconds_to_duration(duration_s)
        req.sideways = bool(sideways)
        req.clockwise = bool(clockwise)
        req.group_mask = int(group_mask)
        return req

    @staticmethod
    def upload_trajectory_request(pieces, trajectory_id: int = 1,
                                  piece_offset: int = 0):
        req = UploadTrajectory.Request()
        req.trajectory_id = int(trajectory_id)
        req.piece_offset = int(piece_offset)
        for row in pieces:
            piece = TrajectoryPolynomialPiece()
            piece.duration = seconds_to_duration(float(row[0]))
            piece.poly_x = [float(c) for c in row[1:9]]
            piece.poly_y = [float(c) for c in row[9:17]]
            piece.poly_z = [float(c) for c in row[17:25]]
            piece.poly_yaw = [float(c) for c in row[25:33]]
            req.pieces.append(piece)
        return req

    @staticmethod
    def start_trajectory_request(trajectory_id: int = 1, timescale: float = 1.0,
                                 relative: bool = False, reversed: bool = False,
                                 group_mask: int = 0):
        req = StartTrajectory.Request()
        req.group_mask = int(group_mask)
        req.trajectory_id = int(trajectory_id)
        req.timescale = float(timescale)
        req.reversed = bool(reversed)
        req.relative = bool(relative)
        return req

    @staticmethod
    def bool_request(value: bool = True):
        req = SetBool.Request()
        req.data = bool(value)
        return req