"""Launch 3-Drone 3D Q-Table + Lyapunov APF Swarm (Sim or OptiTrack Hardware)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    default_pid_conf = os.path.join(
        get_package_share_directory('crazybridge'), 'config', 'pid.conf'
    )

    args = [
        DeclareLaunchArgument(
            'use_sim', default_value='true',
            description='true = CrazySim UDP URIs (no OptiTrack needed); '
                        'false = Crazyradio hardware URIs + OptiTrack Rigid Bodies.'
        ),
        # Hardware Radio URIs (used when use_sim:=false)
        DeclareLaunchArgument('uri_cf1', default_value='radio://0/80/2M/E7E7E7E701'),
        DeclareLaunchArgument('uri_cf2', default_value='radio://0/80/2M/E7E7E7E702'),
        DeclareLaunchArgument('uri_cf3', default_value='radio://0/80/2M/E7E7E7E703'),
        # Simulation UDP URIs (used when use_sim:=true)
        DeclareLaunchArgument('sim_uri_cf1', default_value='udp://127.0.0.1:19850'),
        DeclareLaunchArgument('sim_uri_cf2', default_value='udp://127.0.0.1:19851'),
        DeclareLaunchArgument('sim_uri_cf3', default_value='udp://127.0.0.1:19852'),
        # Motive Rigid Body Asset Names (must match the 4-ball Rigid Body names in Motive)
        DeclareLaunchArgument('rb_cf1', default_value='cf1'),
        DeclareLaunchArgument('rb_cf2', default_value='cf2'),
        DeclareLaunchArgument('rb_cf3', default_value='cf3'),
        # Controller Settings
        DeclareLaunchArgument(
            'control_mode', default_value='goto',
            description="'goto' (safe 10Hz High-Level lookahead) or 'cmd_vel' (20Hz direct velocity)."
        ),
        DeclareLaunchArgument('use_q_table', default_value='true'),
        DeclareLaunchArgument('arena_radius_m', default_value='1.35'),
        DeclareLaunchArgument('base_altitude_m', default_value='1.0'),
        DeclareLaunchArgument('v_max_mps', default_value='0.45'),
        DeclareLaunchArgument('pid_conf_path', default_value=default_pid_conf),
        DeclareLaunchArgument('load_pid_conf', default_value='false'),
    ]

    use_sim = LaunchConfiguration('use_sim')

    def make_bridge_node(name: str, hw_uri_arg: str, sim_uri_arg: str, rb_arg: str) -> Node:
        uri_expr = PythonExpression([
            "'", LaunchConfiguration(sim_uri_arg), "' if '", use_sim,
            "'.lower() == 'true' else '", LaunchConfiguration(hw_uri_arg), "'"
        ])
        return Node(
            package='crazybridge',
            executable='crazybridge',
            name=name,
            output='screen',
            parameters=[{
                'uri': uri_expr,
                'rigid_body_name': LaunchConfiguration(rb_arg),
                'child_frame_id': name,
                'pid_conf_path': LaunchConfiguration('pid_conf_path'),
                'load_pid_conf': ParameterValue(
                    LaunchConfiguration('load_pid_conf'), value_type=bool
                ),
            }],
        )

    cf1_node = make_bridge_node('cf1', 'uri_cf1', 'sim_uri_cf1', 'rb_cf1')
    cf2_node = make_bridge_node('cf2', 'uri_cf2', 'sim_uri_cf2', 'rb_cf2')
    cf3_node = make_bridge_node('cf3', 'uri_cf3', 'sim_uri_cf3', 'rb_cf3')

    optitrack_node = Node(
        package='optitrack_client',
        executable='optitrack_client',
        name='optitrack',
        output='screen',
        condition=UnlessCondition(use_sim),
    )

    swarm_controller_node = Node(
        package='crazybridge',
        executable='swarm_q_apf',
        name='swarm_q_apf_node',
        output='screen',
        parameters=[{
            'drone_names': ['cf1', 'cf2', 'cf3'],
            'priorities': [1.0, 0.5, 0.0],
            'arena_radius_m': ParameterValue(
                LaunchConfiguration('arena_radius_m'), value_type=float
            ),
            'base_altitude_m': ParameterValue(
                LaunchConfiguration('base_altitude_m'), value_type=float
            ),
            'v_max_mps': ParameterValue(
                LaunchConfiguration('v_max_mps'), value_type=float
            ),
            'use_q_table': ParameterValue(
                LaunchConfiguration('use_q_table'), value_type=bool
            ),
            'control_mode': LaunchConfiguration('control_mode'),
        }],
    )

    return LaunchDescription(args + [
        optitrack_node,
        cf1_node,
        cf2_node,
        cf3_node,
        swarm_controller_node,
    ])