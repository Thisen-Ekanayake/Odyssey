"""Live, cooperative 4-drone mapping: merge every drone's LiDAR into ONE map.

    ros2 launch airsim_swarm_bridge cooperative_mapping.launch.py
    ros2 launch airsim_swarm_bridge cooperative_mapping.launch.py auto_maneuver:=edge_to_center

Poses come from ground truth (via the TF chain bridge_node already publishes),
not from an estimator -- this demonstrates cooperative MAPPING, four sensors
building one structure, not cooperative SLAM. For "how well would 4 independent
estimators agree on one map", each drone would need its own SLAM front end; that
is a harder, different exercise and not what this launch file does.

`auto_maneuver` fires the named `/swarm/*` service a few seconds after the bridge
comes up, purely so a demo recording can be one command instead of two terminals
racing each other. Leave it unset to trigger flight yourself, e.g. after RViz has
had a chance to open and you have started your screen recording:

    ros2 service call /swarm/edge_to_center airsim_swarm_msgs/srv/Maneuver "{name: edge_to_center}"

Recording the result: `tools/window_recorder.py` (X11 window capture) works on
the RViz window exactly as it does on AirSim's own window; `tools/to_gif.sh`
turns the output into a LinkedIn-sized GIF.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "airsim_swarm_bridge"


def _launch_setup(context, *args, **kwargs):
    cfg = lambda n: LaunchConfiguration(n).perform(context)      # noqa: E731
    truthy = lambda s: s.lower() in ("1", "true", "yes")          # noqa: E731
    share = get_package_share_directory(PKG)

    drones = [d.strip() for d in cfg("drones").split(",") if d.strip()]

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, "launch", "bridge.launch.py")),
            launch_arguments={
                "drones": cfg("drones"),
                "host": cfg("host"), "port": cfg("port"),
                "stereo": "false",          # a mapping demo needs LiDAR, not images
                "swarm_state": "true",      # formation markers, drawn alongside the map
                "maneuvers": "true",        # so a maneuver can be triggered at all
            }.items(),
        ),
        Node(
            package=PKG, executable="cloud_merger_node", name="cloud_merger",
            output="screen",
            parameters=[{"drones": drones, "output_topic": "/swarm/merged_points"}],
        ),
        Node(
            package="octomap_server", executable="octomap_server_node", name="octomap_server",
            output="screen",
            parameters=[os.path.join(share, "config", "octomap.yaml"),
                       {"frame_id": "map", "filter_ground_plane": False}],
            remappings=[("cloud_in", "/swarm/merged_points")],
        ),
    ]

    if truthy(cfg("rviz")):
        actions.append(Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", os.path.join(share, "rviz", "swarm_live.rviz")],
        ))

    maneuver = cfg("auto_maneuver")
    if maneuver:
        delay = float(cfg("auto_maneuver_delay"))
        actions.append(TimerAction(period=delay, actions=[
            ExecuteProcess(cmd=[
                "ros2", "service", "call", f"/swarm/{maneuver}",
                "airsim_swarm_msgs/srv/Maneuver", f"{{name: {maneuver}}}",
            ], output="screen"),
        ]))
    return actions


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("drones", default_value="Drone1,Drone2,Drone3,Drone4"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="41451"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("auto_maneuver", default_value="",
                              description="e.g. 'edge_to_center' or 'converge' -- "
                                          "fired automatically after auto_maneuver_delay. "
                                          "Empty (default): trigger it yourself."),
        DeclareLaunchArgument("auto_maneuver_delay", default_value="8.0",
                              description="seconds to wait for the bridge + RViz to "
                                          "settle before firing auto_maneuver"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
