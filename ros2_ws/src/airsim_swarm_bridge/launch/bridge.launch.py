"""Bring up one bridge node per drone, plus the swarm state/marker node.

    ros2 launch airsim_swarm_bridge bridge.launch.py                    # all 4
    ros2 launch airsim_swarm_bridge bridge.launch.py drones:=Drone1     # just one
    ros2 launch airsim_swarm_bridge bridge.launch.py drones:=Drone1,Drone2 stereo:=false

One process per drone is deliberate: msgpack-rpc is not thread-safe, and
simGetImages is slow enough that sharing a connection would serialise the whole
swarm behind one camera. See bridge_node's module docstring.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os

DEFAULT_DRONES = "Drone1,Drone2,Drone3,Drone4"


def _launch_setup(context, *args, **kwargs):
    cfg = lambda n: LaunchConfiguration(n).perform(context)      # noqa: E731
    drones = [d.strip() for d in cfg("drones").split(",") if d.strip()]
    params_file = os.path.join(
        get_package_share_directory("airsim_swarm_bridge"), "config", "drones.yaml")

    truthy = lambda s: s.lower() in ("1", "true", "yes")          # noqa: E731
    overrides = {
        "host": cfg("host"),
        "port": int(cfg("port")),
        "enable_stereo": truthy(cfg("stereo")),
        "enable_lidar": truthy(cfg("lidar")),
        "gt_owns_base_link": truthy(cfg("gt_owns_base_link")),
    }

    nodes = []
    for d in drones:
        nodes.append(Node(
            package="airsim_swarm_bridge",
            executable="bridge_node",
            # No namespace= here: bridge_node builds absolute topic names from
            # vehicle_name itself, so adding one would nest them (/drone1/drone1/imu).
            name=f"bridge_{d.lower()}",
            output="screen",
            emulate_tty=True,
            parameters=[params_file, dict(overrides, vehicle_name=d)],
        ))

    if truthy(cfg("swarm_state")) and len(drones) > 1:
        nodes.append(Node(
            package="airsim_swarm_bridge",
            executable="swarm_state_node",
            name="swarm_state",
            output="screen",
            emulate_tty=True,
            parameters=[{"host": cfg("host"), "port": int(cfg("port")),
                         "drones": drones, "rate": 10.0}],
        ))

    if truthy(cfg("maneuvers")):
        nodes.append(Node(
            package="airsim_swarm_bridge",
            executable="maneuver_node",
            name="swarm_maneuver",
            output="screen",
            emulate_tty=True,
            parameters=[{"host": cfg("host"), "port": int(cfg("port")),
                         "drones": drones}],
        ))
    return nodes


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("drones", default_value=DEFAULT_DRONES,
                              description="comma-separated vehicle names from settings.json"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="41451"),
        DeclareLaunchArgument("stereo", default_value="true",
                              description="publish stereo images (the expensive RPC)"),
        DeclareLaunchArgument("lidar", default_value="true"),
        DeclareLaunchArgument("swarm_state", default_value="true",
                              description="publish /swarm/state and formation markers"),
        DeclareLaunchArgument("maneuvers", default_value="true",
                              description="expose /swarm/* manoeuvre services"),
        DeclareLaunchArgument("gt_owns_base_link", default_value="true",
                              description="false when a SLAM node will own odom->base_link"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
