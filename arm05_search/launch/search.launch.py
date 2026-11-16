from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NAV2_FAST_PARAMS = [
    ('/controller_server', 'FollowPath.max_vel_x', '0.5'),
    ('/controller_server', 'FollowPath.max_speed_xy', '0.5'),
    ('/controller_server', 'FollowPath.max_vel_theta', '1.5'),
]


def generate_launch_description():
    fast = LaunchConfiguration('fast')
    set_params = [
        ExecuteProcess(
            cmd=['ros2', 'param', 'set', node, name, value],
            output='screen',
            condition=IfCondition(fast),
        )
        for node, name, value in NAV2_FAST_PARAMS
    ]
    return LaunchDescription([
        DeclareLaunchArgument('fast', default_value='true', description='przyspiesz Nav2 do 0.5 m/s'),
        *set_params,
        Node(
            package='arm05_search',
            executable='aruco_detector.py',
            output='screen',
            parameters=[{'use_sim_time': True, 'process_rate_hz': 10.0}],
        ),
        Node(
            package='arm05_search',
            executable='search_server.py',
            output='screen',
            emulate_tty=True,
            parameters=[{'use_sim_time': True}],
        ),
    ])
