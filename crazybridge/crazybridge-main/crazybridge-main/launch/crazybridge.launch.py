import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_pid_conf = os.path.join(
        get_package_share_directory('crazybridge'), 'config', 'pid.conf'
    )

    uri_arg = DeclareLaunchArgument(
        'uri',
        default_value='udp://127.0.0.1:19850',
        description='cflib URI for the Crazyflie (default targets CrazySim SITL).',
    )
    log_period_arg = DeclareLaunchArgument(
        'log_period_ms',
        default_value='50',
        description=(
            'Period in ms for the Kalman state log blocks. cflib quantises '
            'this to 10 ms ticks, so values must be a multiple of 10 and >= 10.'
        ),
    )
    oot_log_period_arg = DeclareLaunchArgument(
        'oot_log_period_ms',
        default_value='100',
        description='Period in ms for the OOT debug log blocks.',
    )
    pid_conf_arg = DeclareLaunchArgument(
        'pid_conf_path',
        default_value=default_pid_conf,
        description='Path to pid.conf with OOT controller gains.',
    )
    load_pid_conf_arg = DeclareLaunchArgument(
        'load_pid_conf',
        default_value='true',
        description='Whether to push pid.conf values to ootParams on connect.',
    )
    rerun_mode_arg = DeclareLaunchArgument(
        'rerun_mode',
        default_value='spawn',
        description=(
            "How to attach to rerun: 'spawn' (launches a viewer), "
            "'connect' (attach to running viewer at rerun_addr), "
            "'save' (write to rerun_save_path), 'disabled'."
        ),
    )
    rerun_addr_arg = DeclareLaunchArgument(
        'rerun_addr',
        default_value='',
        description='gRPC address for rerun_mode=connect (empty = default).',
    )
    rerun_save_path_arg = DeclareLaunchArgument(
        'rerun_save_path',
        default_value='',
        description='Output .rrd path for rerun_mode=save.',
    )
    rerun_oot_decimate_arg = DeclareLaunchArgument(
        'rerun_oot_decimate',
        default_value='2',
        description=(
            'Keep every Nth OOT sample when streaming to rerun. Raise this '
            'if you see re_quota_channel back-pressure warnings.'
        ),
    )

    return LaunchDescription([
        uri_arg,
        log_period_arg,
        oot_log_period_arg,
        pid_conf_arg,
        load_pid_conf_arg,
        rerun_mode_arg,
        rerun_addr_arg,
        rerun_save_path_arg,
        rerun_oot_decimate_arg,
        Node(
            package='crazybridge',
            executable='crazybridge',
            name='crazybridge',
            output='screen',
            parameters=[{
                'uri': LaunchConfiguration('uri'),
                'log_period_ms': LaunchConfiguration('log_period_ms'),
                'oot_log_period_ms': LaunchConfiguration('oot_log_period_ms'),
                'pid_conf_path': LaunchConfiguration('pid_conf_path'),
                'load_pid_conf': LaunchConfiguration('load_pid_conf'),
            }],
        ),
        Node(
            package='crazybridge',
            executable='rerun',
            name='rerun',
            output='screen',
            parameters=[{
                'rerun_mode': LaunchConfiguration('rerun_mode'),
                'rerun_addr': LaunchConfiguration('rerun_addr'),
                'rerun_save_path': LaunchConfiguration('rerun_save_path'),
            }]
        )
    ])
