from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, Command
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition
from launch_ros.parameter_descriptions import ParameterValue
from launch.event_handlers import OnProcessStart  # 确保正确导入事件处理器

def generate_launch_description():
    pkg_path = FindPackageShare('ubt_left_hand_v3_description')
    
    urdf_path = PathJoinSubstitution([
        pkg_path,
        'urdf',
        'hand3_v1/hand3_v1.urdf'
    ])
    
    rviz_config_path = PathJoinSubstitution([
        pkg_path,
        'config',
        'view_robot.rviz'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock'
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz'
        ),
        DeclareLaunchArgument(
            'urdf_file',
            default_value=urdf_path,
            description='Absolute path to URDF file'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Launch joint_state_publisher_gui'
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                pkg_path,
                'config',
                'rviz',
                'view_robot.rviz'
            ]),
            description='RViz config file path'
        ),

        # 修正后的URDF验证（使用check_urdf替代）
        ExecuteProcess(
            cmd=['check_urdf', LaunchConfiguration('urdf_file')],
            output='screen',
            name='urdf_validation',
            shell=True
        ),

        # 机器人状态发布节点
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': ParameterValue(
                    Command(['cat ', LaunchConfiguration('urdf_file')]),
                    value_type=str
                ),
                'publish_frequency': 50.0,
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }],
            remappings=[
                ('/joint_states', 'joint_states'),
                ('/robot_description', 'robot_description')
            ]
        ),

        # 修正后的关节状态GUI发布器配置
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'rate': 50
            }],
            condition=IfCondition(LaunchConfiguration('gui'))
        ),

        # 修正后的RViz节点配置（移除错误的事件处理器）
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('rviz')),
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),

        # 独立配置的静态TF发布器（不再依赖事件触发）
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'L_hand_base_link'],
            output='screen'
        )
    ])
