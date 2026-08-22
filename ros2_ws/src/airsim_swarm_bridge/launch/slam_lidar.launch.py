"""3D LiDAR SLAM over one drone: icp_odometry -> rtabmap -> octomap.

Works identically on a live bridge and on a replayed bag, because
dataset_to_rosbag writes the same topic names and frame ids the bridge publishes.

    # live
    ros2 launch airsim_swarm_bridge slam_lidar.launch.py
    # over a bag (see slam_replay.launch.py, which also starts `ros2 bag play`)
    ros2 launch airsim_swarm_bridge slam_lidar.launch.py use_sim_time:=true bridge:=false

Note `bridge:=false` when replaying -- rtabmap must not have a live AirSim
connection competing with the bag for the same topics.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "airsim_swarm_bridge"


def _launch_setup(context, *args, **kwargs):
    cfg = lambda n: LaunchConfiguration(n).perform(context)      # noqa: E731
    truthy = lambda s: s.lower() in ("1", "true", "yes")          # noqa: E731

    share = get_package_share_directory(PKG)
    rtabmap_cfg = os.path.join(share, "config", "rtabmap_lidar.yaml")
    octomap_cfg = os.path.join(share, "config", "octomap.yaml")
    ekf_cfg = os.path.join(share, "config", "ekf.yaml")

    drone = cfg("drone")
    ns = drone.lower()
    sim_time = {"use_sim_time": truthy(cfg("use_sim_time"))}

    # Frame ids are per-drone, so the shared config file's defaults (drone1) get
    # overridden here rather than duplicated into four near-identical yaml files.
    frames = {
        "frame_id": f"{ns}/base_link",
        "odom_frame_id": f"{ns}/odom",
        "map_frame_id": "map",
    }
    scan_topic = f"/{ns}/lidar/points"
    imu_topic = f"/{ns}/imu"

    actions = []

    if truthy(cfg("bridge")):
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, "launch", "bridge.launch.py")),
            launch_arguments={
                "drones": drone,
                "stereo": cfg("stereo"),
                # rtabmap's icp_odometry publishes odom -> base_link, so the
                # bridge must not: two publishers on one TF edge corrupts the tree.
                "gt_owns_base_link": "false",
                "swarm_state": "false",
            }.items(),
        ))

    actions.append(Node(
        package="rtabmap_odom", executable="icp_odometry", name="icp_odometry",
        output="screen",
        parameters=[rtabmap_cfg, frames, sim_time,
                    {"subscribe_scan_cloud": True, "wait_imu_to_init": True}],
        remappings=[("scan_cloud", scan_topic), ("imu", imu_topic)],
        arguments=["--ros-args", "--log-level", cfg("log_level")],
    ))

    actions.append(Node(
        package="rtabmap_slam", executable="rtabmap", name="rtabmap",
        output="screen",
        parameters=[rtabmap_cfg, frames, sim_time, {"subscribe_scan_cloud": True}],
        remappings=[("scan_cloud", scan_topic), ("imu", imu_topic)],
        # --delete_db_on_start: otherwise a second run silently resumes the first
        # run's graph, and the ATE you report is of a session you did not fly.
        arguments=(["--delete_db_on_start"] if truthy(cfg("delete_db")) else []) +
                  ["--ros-args", "--log-level", cfg("log_level")],
    ))

    if truthy(cfg("ekf")):
        actions.append(Node(
            package="robot_localization", executable="ekf_node", name="ekf_filter_node",
            output="screen",
            parameters=[ekf_cfg, sim_time, {
                "odom_frame": f"{ns}/odom", "base_link_frame": f"{ns}/base_link",
                "world_frame": f"{ns}/odom", "imu0": imu_topic}],
        ))

    if truthy(cfg("octomap")):
        actions.append(Node(
            package="octomap_server", executable="octomap_server_node", name="octomap_server",
            output="screen",
            parameters=[octomap_cfg, sim_time, {"base_frame_id": f"{ns}/base_link"}],
            # rtabmap publishes its assembled map cloud on /cloud_map. The node name
            # does NOT namespace its topics, so there is no /rtabmap/ prefix.
            remappings=[("cloud_in", "/cloud_map")],
        ))

    if truthy(cfg("record_trajectory")):
        actions.append(Node(
            package=PKG, executable="traj_recorder", name="traj_recorder",
            output="screen",
            parameters=[sim_time, {
                # icp_odometry publishes on /odom. /rtabmap/odom exists but is a
                # different, lower-rate republication -- not what to score.
                "topic": "/odom",
                "out": cfg("trajectory_out"),
                "to_ned": True,
                "header": f"rtabmap icp_odometry, drone={drone}",
            }],
        ))

    if truthy(cfg("rviz")):
        actions.append(Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", os.path.join(share, "rviz", "slam_single.rviz")],
            parameters=[sim_time],
        ))
    return actions


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("drone", default_value="Drone1"),
        DeclareLaunchArgument("bridge", default_value="true",
                              description="start the AirSim bridge (false when replaying a bag)"),
        DeclareLaunchArgument("stereo", default_value="false",
                              description="LiDAR SLAM does not need it; off by default to save RPC"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("ekf", default_value="false",
                              description="fuse IMU with ICP odometry via robot_localization"),
        # rtabmap builds an octomap internally when Grid/3D is on and publishes
        # /octomap_full, /octomap_binary, /octomap_occupied_space and /cloud_map
        # itself. A second octomap_server on top re-voxelises the same data and
        # regenerates the whole cloud on every graph change, so it is off by
        # default; turn it on only to build a map independent of rtabmap's graph.
        DeclareLaunchArgument("octomap", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("delete_db", default_value="true"),
        DeclareLaunchArgument("record_trajectory", default_value="false"),
        DeclareLaunchArgument("trajectory_out", default_value="results/ros/rtabmap.txt"),
        DeclareLaunchArgument("log_level", default_value="info"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
