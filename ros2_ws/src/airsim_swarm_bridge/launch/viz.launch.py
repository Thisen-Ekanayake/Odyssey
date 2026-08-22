"""The visualisation layer: RViz2, Foxglove, PlotJuggler.

    ros2 launch airsim_swarm_bridge viz.launch.py                  # all of it
    ros2 launch airsim_swarm_bridge viz.launch.py plotjuggler:=false

Foxglove serves ws://localhost:8765 -- open https://app.foxglove.dev, choose
"Open connection", and paste that. It is the easiest way to capture clips without
the Open3D window-recorder path in tools/window_recorder.py.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "airsim_swarm_bridge"


def _launch_setup(context, *args, **kwargs):
    cfg = lambda n: LaunchConfiguration(n).perform(context)      # noqa: E731
    truthy = lambda s: s.lower() in ("1", "true", "yes")          # noqa: E731
    share = get_package_share_directory(PKG)
    sim_time = {"use_sim_time": truthy(cfg("use_sim_time"))}

    actions = []
    if truthy(cfg("rviz")):
        actions.append(Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", os.path.join(share, "rviz", cfg("rviz_config"))],
            parameters=[sim_time],
        ))
    if truthy(cfg("foxglove")):
        actions.append(Node(
            package="foxglove_bridge", executable="foxglove_bridge", name="foxglove_bridge",
            output="screen",
            parameters=[sim_time, {
                "port": int(cfg("foxglove_port")),
                "address": "0.0.0.0",
                # PointCloud2 at 10 Hz x 4 drones is a lot of websocket traffic;
                # the default 8 MB cap drops whole clouds silently.
                "send_buffer_limit": 100_000_000,
                "use_compression": True,
            }],
        ))
    if truthy(cfg("plotjuggler")):
        actions.append(Node(
            package="plotjuggler", executable="plotjuggler", name="plotjuggler",
            output="log", arguments=["--nosplash"], parameters=[sim_time],
        ))
    if truthy(cfg("image_view")):
        actions.append(Node(
            package="rqt_image_view", executable="rqt_image_view", name="rqt_image_view",
            output="log", arguments=[f"/{cfg('drone').lower()}/stereo/left/image_raw"],
        ))
    return actions


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value="swarm_live.rviz"),
        DeclareLaunchArgument("foxglove", default_value="true"),
        DeclareLaunchArgument("foxglove_port", default_value="8765"),
        DeclareLaunchArgument("plotjuggler", default_value="true"),
        DeclareLaunchArgument("image_view", default_value="false"),
        DeclareLaunchArgument("drone", default_value="Drone1"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
