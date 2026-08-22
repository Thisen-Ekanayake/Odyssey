"""Optional 2D comparison: pointcloud_to_laserscan -> slam_toolbox.

Included because it is cheap, but read the caveat before reporting anything from
it. slam_toolbox is a 2D SLAM system: it wants a planar LiDAR on a robot that
stays level. This drone flies a fixed-altitude circuit, so slicing a horizontal
band out of the 3D cloud does produce a usable scan -- but it throws away the
vertical structure that is most of what the sensor measured, and any bank angle
during a turn tilts the slice through the world.

Treat it as a sanity check that the topics and frames are sane, not as a result
to put next to rtabmap.

    ros2 launch airsim_swarm_bridge slam_2d.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    cfg = lambda n: LaunchConfiguration(n).perform(context)      # noqa: E731
    truthy = lambda s: s.lower() in ("1", "true", "yes")          # noqa: E731
    ns = cfg("drone").lower()
    sim_time = {"use_sim_time": truthy(cfg("use_sim_time"))}

    return [
        Node(
            package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node",
            name="pointcloud_to_laserscan", output="screen",
            remappings=[("cloud_in", f"/{ns}/lidar/points"), ("scan", "/scan")],
            parameters=[sim_time, {
                "target_frame": f"{ns}/base_link",
                # A 4 m band around the sensor: thick enough to catch returns
                # while the drone banks, thin enough to stay a "slice".
                "min_height": -2.0,
                "max_height": 2.0,
                "angle_min": -3.14159,
                "angle_max": 3.14159,
                "angle_increment": 0.0087,      # ~0.5 deg
                "range_min": 1.0,
                "range_max": 90.0,
                "use_inf": True,
            }],
        ),
        Node(
            package="slam_toolbox", executable="async_slam_toolbox_node",
            name="slam_toolbox", output="screen",
            parameters=[sim_time, {
                "odom_frame": f"{ns}/odom",
                "base_frame": f"{ns}/base_link",
                "map_frame": "map",
                "scan_topic": "/scan",
                "mode": "mapping",
                "resolution": 0.5,
                "max_laser_range": 90.0,
                "minimum_travel_distance": 1.0,
                "minimum_travel_heading": 0.2,
            }],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("drone", default_value="Drone1"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        OpaqueFunction(function=_launch_setup),
    ])
