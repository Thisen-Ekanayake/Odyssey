"""Replay a converted dataset bag through the LiDAR SLAM stack -- no simulator.

This is the reproducible half of the study. The bag holds the exact bytes
`flight/record_dataset.py` recorded, so every rerun and every method sees
identical input, which is the same guarantee `tools/run_benchmark.py` gives the
Python SLAM package.

    ros2 run airsim_swarm_bridge dataset_to_rosbag datasets/clear --out bags/clear
    ros2 launch airsim_swarm_bridge slam_replay.launch.py bag:=bags/clear

    # and to score it against ground truth afterwards, on the HOST:
    ./airsim_venv/bin/python tools/compare_ros_slam.py --condition clear

`use_sim_time` is forced on: message stamps come from the recording's own AirSim
clock, and running the estimator on wall-clock instead would mis-associate every
IMU interval.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

PKG = "airsim_swarm_bridge"


def _launch_setup(context, *args, **kwargs):
    cfg = lambda n: LaunchConfiguration(n).perform(context)      # noqa: E731
    share = get_package_share_directory(PKG)

    bag = cfg("bag")
    if not os.path.isabs(bag):
        repo = os.environ.get("AIRSIM_SWARM_REPO", "/ml/airsim_swarm")
        bag = os.path.join(repo, bag)

    play = ["ros2", "bag", "play", bag, "--clock", "--rate", cfg("rate")]
    if cfg("loop").lower() in ("1", "true", "yes"):
        play.append("--loop")
    if cfg("start_offset") != "0":
        play += ["--start-offset", cfg("start_offset")]

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, "launch", "slam_lidar.launch.py")),
            launch_arguments={
                "drone": cfg("drone"),
                "bridge": "false",          # the bag is the source; no live sim
                "use_sim_time": "true",
                "rviz": cfg("rviz"),
                "octomap": cfg("octomap"),
                "ekf": cfg("ekf"),
                "record_trajectory": cfg("record_trajectory"),
                "trajectory_out": cfg("trajectory_out"),
            }.items(),
        ),
        # Delayed by a couple of seconds so rtabmap has its subscriptions up
        # before the first messages arrive -- otherwise the opening scans are
        # dropped and odometry initialises mid-flight.
        ExecuteProcess(cmd=["bash", "-c", f"sleep 3; exec {' '.join(play)}"],
                       output="screen"),
    ]


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("bag", description="rosbag2 directory from dataset_to_rosbag"),
        DeclareLaunchArgument("drone", default_value="Drone1"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("start_offset", default_value="0"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("octomap", default_value="false"),
        DeclareLaunchArgument("ekf", default_value="false"),
        DeclareLaunchArgument("record_trajectory", default_value="true"),
        DeclareLaunchArgument("trajectory_out", default_value="results/ros/rtabmap.txt"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
