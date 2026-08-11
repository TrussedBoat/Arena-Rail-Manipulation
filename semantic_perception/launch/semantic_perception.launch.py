import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    default_config_file = os.path.join(
        get_package_share_directory("semantic_perception"),
        "config",
        "semantic_perception.yaml",
    )
    arguments = [
        DeclareLaunchArgument("model_path"),
        DeclareLaunchArgument(
            "config_file", default_value=TextSubstitution(text=default_config_file)
        ),
        DeclareLaunchArgument("registry_path", default_value="semantic_objects.json"),
        DeclareLaunchArgument(
            "legacy_coordinates_path", default_value="semantic_distances_dynamic.json"
        ),
        DeclareLaunchArgument("write_legacy_coordinates", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("publish_annotated_debug", default_value="false"),
    ]
    node = Node(
        package="semantic_perception",
        executable="semantic_perception_node",
        name="semantic_perception",
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
            {
                "detector.model_path": LaunchConfiguration("model_path"),
                "registry.path": LaunchConfiguration("registry_path"),
                "registry.legacy_coordinates_path": LaunchConfiguration(
                    "legacy_coordinates_path"
                ),
                "registry.write_legacy_coordinates": ParameterValue(
                    LaunchConfiguration("write_legacy_coordinates"), value_type=bool
                ),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "debug.publish_annotated": ParameterValue(
                    LaunchConfiguration("publish_annotated_debug"), value_type=bool
                ),
            }
        ],
    )
    return LaunchDescription(arguments + [node])
