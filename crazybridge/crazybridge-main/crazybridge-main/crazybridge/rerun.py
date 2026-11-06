from rerun_sdk import rerun as rr
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Vector3, Quaternion, PoseWithCovariance, Pose, Point, PointStamped

class Rerun(Node):
    def __init__(self):
        super().__init__('rerun')
        # rerun config: mode is one of 'spawn', 'connect', 'save', 'disabled'.
        # 'spawn'   launches a local viewer.
        # 'connect' attaches to an already-running viewer at rerun_addr.
        # 'save'    writes to rerun_save_path (.rrd file).
        # 'disabled' no-ops every log call.
        self.declare_parameter('rerun_mode', 'spawn')
        self.declare_parameter('rerun_app_id', 'crazybridge')
        self.declare_parameter('rerun_addr', '')
        self.declare_parameter('rerun_save_path', '')
        # Name of the run being flown (the controller_test config_name). When
        # set, the rerun application id becomes '<app_id>/<run_name>' so each
        # config shows up as its own entry in the live viewer's recording list
        # instead of every run piling into one 'crazybridge' stream.
        self.declare_parameter('run_name', '')
        # Cap on how many points to keep in each trajectory line strip so a long
        # session does not grow the log unbounded.
        self.declare_parameter('path_max_points', 5000)

        self._path_max = int(
            self.get_parameter('path_max_points').get_parameter_value().integer_value)
        # Accumulated planned (setpoint) and actual (odometry) paths, drawn as
        # 3D line strips so the tracking can be seen live and side by side.
        self._planned_path: list[list[float]] = []
        self._actual_path: list[list[float]] = []

        self._init_rerun()
        self._init_error_logs()
        self._init_orientation_logs()

        self.create_subscription(Vector3, 'pos_error', self._pos_err_cb, 1)
        self.create_subscription(Vector3, 'torque', self._torque_cb, 1)
        self.create_subscription(Float32, 'thrust', self._thrust_cb, 1)
        self.create_subscription(Quaternion, 'orientation/desired', self._qd_cb, 1)
        self.create_subscription(Quaternion, 'orientation/error', self._qe_cb, 1)
        self.create_subscription(Odometry, '/crazybridge/odometry', self._odometry_cb, 1)
        self.create_subscription(PointStamped, 'setpoint', self._setpoint_cb, 1)

    @staticmethod
    def _normalize_quat(q: Quaternion) -> tuple[float, float, float, float]:
        """Return (x, y, z, w) as a unit quaternion.

        The Crazyflie estimator occasionally emits quaternions that are not
        quite unit-length (e.g. right after connect, before the filter has
        settled). Normalizing here keeps the rerun plots and the heading arrow
        well-defined. Falls back to identity when the norm is degenerate.
        """
        x, y, z, w = q.x, q.y, q.z, q.w
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        return x / norm, y / norm, z / norm, w / norm

    def _qd_cb(self, msg: Quaternion):
        x, y, z, w = self._normalize_quat(msg)
        rr.log("/orientation/desired/w", rr.Scalars(w))
        rr.log("/orientation/desired/x", rr.Scalars(x))
        rr.log("/orientation/desired/y", rr.Scalars(y))
        rr.log("/orientation/desired/z", rr.Scalars(z))

    def _qe_cb(self, msg: Quaternion):
        x, y, z, w = self._normalize_quat(msg)
        rr.log("/orientation/error/w", rr.Scalars(w))
        rr.log("/orientation/error/x", rr.Scalars(x))
        rr.log("/orientation/error/y", rr.Scalars(y))
        rr.log("/orientation/error/z", rr.Scalars(z))

    def _thrust_cb(self, msg: Float32):
        rr.log("/input/thrust", rr.Scalars(msg.data))

    def _torque_cb(self, msg: Vector3):
        rr.log("/input/torque/x", rr.Scalars(msg.x))
        rr.log("/input/torque/y", rr.Scalars(msg.y))
        rr.log("/input/torque/z", rr.Scalars(msg.z))


    def _pos_err_cb(self, msg: Vector3):
        if abs(msg.x) > 3:
            msg.x = 0.0
        if abs(msg.y) > 3:
            msg.y = 0.0
        if abs(msg.z) > 3:
            msg.z = 0.0
        rr.log("/errors/pos/x", rr.Scalars(msg.x))
        rr.log("/errors/pos/y", rr.Scalars(msg.y))
        rr.log("/errors/pos/z", rr.Scalars(msg.z))

    def _odometry_cb(self, msg: Odometry):
        poseWC: PoseWithCovariance = msg.pose
        pose: Pose = poseWC.pose
        q: Quaternion = pose.orientation
        x, y, z, w = self._normalize_quat(q)
        rot: Rotation = Rotation.from_quat([x, y, z, w])
        p: Point = pose.position
        start = [p.x, p.y, p.z]
        # Arrow vector is a displacement from the origin, so rotate the drone's
        # body-frame forward axis (+x) into the world frame and scale it. This
        # makes the arrow tip track the drone's heading regardless of position.
        arrow_length = 0.2
        vector = rot.apply([1.0, 0.0, 0.0]) * arrow_length
        rr.log("drone", rr.Arrows3D(origins=[start], vectors=[vector], radii=[0.01]))
        self._append_path(self._actual_path, start,
                          "world/actual_path", [255, 140, 0])
        rr.log("/orientation/w", rr.Scalars(w))
        rr.log("/orientation/x", rr.Scalars(x))
        rr.log("/orientation/y", rr.Scalars(y))
        rr.log("/orientation/z", rr.Scalars(z))

    def _setpoint_cb(self, msg: PointStamped):
        p: Point = msg.point
        self._append_path(self._planned_path, [p.x, p.y, p.z],
                          "world/planned_path", [0, 180, 255])

    def _append_path(self, buf: list, point: list, entity: str, color: list):
        """Append a point to a path buffer and redraw it as a 3D line strip.

        The planned (setpoint) and actual (odometry) paths are drawn as two
        line strips so the tracking error is visible live and side by side.
        """
        buf.append([float(point[0]), float(point[1]), float(point[2])])
        if len(buf) > self._path_max:
            del buf[0]
        if len(buf) >= 2:
            rr.log(entity, rr.LineStrips3D([buf], colors=[color], radii=[0.004]))

    def _init_orientation_logs(self):
        rr.log(
            "/orientation/error/w",
            rr.SeriesLines( colors=[255, 255, 0], names="w"),
            static=True)
        rr.log(
            "/orientation/error/x",
            rr.SeriesLines( colors=[255, 0, 0], names="x"),
            static=True)
        rr.log(
            "/orientation/error/y",
            rr.SeriesLines( colors=[0, 255, 0], names="y"),
            static=True)
        rr.log(
            "/orientation/error/z",
            rr.SeriesLines( colors=[0, 0, 255], names="z",),
            static=True)

        rr.log(
            "/orientation/desired/w",
            rr.SeriesLines( colors=[255, 255, 0], names="w"),
            static=True)
        rr.log(
            "/orientation/desired/x",
            rr.SeriesLines( colors=[255, 0, 0], names="x"),
            static=True)
        rr.log(
            "/orientation/desired/y",
            rr.SeriesLines( colors=[0, 255, 0], names="y"),
            static=True)
        rr.log(
            "/orientation/desired/z",
            rr.SeriesLines( colors=[0, 0, 255], names="z",),
            static=True)

        rr.log(
            "/orientation/w",
            rr.SeriesLines( colors=[255, 255, 0], names="w"),
            static=True)
        rr.log(
            "/orientation/x",
            rr.SeriesLines( colors=[255, 0, 0], names="x"),
            static=True)
        rr.log(
            "/orientation/y",
            rr.SeriesLines( colors=[0, 255, 0], names="y"),
            static=True)
        rr.log(
            "/orientation/z",
            rr.SeriesLines( colors=[0, 0, 255], names="z",),
            static=True)

    def _init_error_logs(self):
        rr.log(
            "/errors/pos/x",
            rr.SeriesLines(
                colors=[255, 0, 0],
                names="x",
                ),
            static=True,
        )
        rr.log(
            "/errors/pos/y",
            rr.SeriesLines(
                colors=[0, 255, 0],
                names="y",
            ),
            static=True,
        )
        rr.log(
            "/errors/pos/z",
            rr.SeriesLines(
                colors=[0, 0, 255],
                names="z",
            ),
            static=True,
        )


    def _init_rerun(self) -> None:
        mode = self.get_parameter('rerun_mode').get_parameter_value().string_value.strip().lower()
        app_id = self.get_parameter(
            'rerun_app_id').get_parameter_value().string_value or 'crazybridge'
        run_name = self.get_parameter(
            'run_name').get_parameter_value().string_value.strip()
        if run_name:
            app_id = f'{app_id}/{run_name}'
        self._rerun_enabled = mode != 'disabled'
        if not self._rerun_enabled:
            return

        self.get_logger().info(f'rerun application id: {app_id}')
        rr.init(app_id)
        try:
            if mode == 'spawn':
                rr.spawn()
            elif mode == 'connect':
                addr = self.get_parameter(
                    'rerun_addr').get_parameter_value().string_value
                if addr:
                    rr.connect_grpc(addr)
                else:
                    rr.connect_grpc()
            elif mode == 'save':
                path = self.get_parameter(
                    'rerun_save_path').get_parameter_value().string_value
                if not path:
                    raise ValueError(
                        'rerun_mode=save requires rerun_save_path to be set'
                    )
                rr.save(path)
            else:
                raise ValueError(f'unknown rerun_mode={mode!r}')
        except Exception as exc:
            # Fall back to the rclpy logger only for setup failures; once we
            # report this we stay disabled so log() calls become no-ops.
            self.get_logger().error(
                f'Failed to initialise rerun (mode={mode!r}): {exc}'
            )
            self._rerun_enabled = False
            return

def main(args=None) -> None:
    rclpy.init(args=args)
    node = Rerun()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
