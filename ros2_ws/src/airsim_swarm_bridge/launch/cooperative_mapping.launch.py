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
    if not drones:
        raise RuntimeError("cooperative_mapping: 'drones' resolved to an empty list")

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, "launch", "bridge.launch.py")),
            launch_arguments={
                "drones": cfg("drones"),
                "host": cfg("host"), "port": cfg("port"),
                "stereo": cfg("stereo"),    # a mapping demo needs LiDAR, not images
                "swarm_state": "true",      # formation markers, drawn alongside the map
                "maneuvers": "true",        # so a maneuver can be triggered at all
                # Four drones hammering the RPC is what drags AirSim's clock below
                # real time, and a slow sim clock is precisely what broke this demo
                # (see bridge_node._stamp). None of this is needed to build a map:
                # GPS off, and IMU kept only fast enough to refresh the sim-clock
                # offset that odom/TF are stamped from -- 20 Hz is 5x the LiDAR
                # rate, so TF stays comfortably denser than the clouds it brackets.
                "gps": "false",
                "imu_rate": "20.0",
                "odom_rate": "20.0",
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
            # base_frame_id must name a drone that is actually being bridged --
            # config/octomap.yaml hardcodes drone1, so `drones:=Drone2,Drone3`
            # would otherwise point octomap at a frame nobody publishes.
            parameters=[os.path.join(share, "config", "octomap.yaml"),
                        {"base_frame_id": f"{drones[0].lower()}/base_link"}],
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
        # Retry rather than fire once. maneuver_node refuses to move anything
        # until every bridge has published odometry, because each one calibrates
        # its world offset from a stationary pose at startup -- flying early
        # misplaces that drone on the shared map with no error. A fixed delay
        # cannot be right for both a warm sim and a cold UE4 start (shader
        # compilation can hold a bridge in confirmConnection for a minute), so
        # poll until the service accepts instead of guessing.
        actions.append(TimerAction(period=delay, actions=[
            ExecuteProcess(cmd=["bash", "-c", f"""
                for i in $(seq 60); do
                  out=$(ros2 service call /swarm/{maneuver} \
                          airsim_swarm_msgs/srv/Maneuver '{{name: {maneuver}}}' 2>&1)
                  case "$out" in
                    *success=True*) echo "$out"; exit 0 ;;
                  esac
                  echo "waiting for /swarm/{maneuver} to accept ($i/60): \
$(echo "$out" | tail -1)"
                  sleep 2
                done
                echo "auto_maneuver {maneuver} never accepted; trigger it by hand" >&2
                exit 1
            """], output="screen"),
        ]))
    return actions


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("drones", default_value="Drone1,Drone2,Drone3,Drone4"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="41451"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("stereo", default_value="false",
                              description="publish stereo images too. Off by default: "
                                          "simGetImages is the expensive RPC and a "
                                          "slow sim clock is what breaks this demo. "
                                          "Turn on (and enable the camera panels in "
                                          "RViz) if you want a camera view in the clip."),
        DeclareLaunchArgument("auto_maneuver", default_value="",
                              description="e.g. 'edge_to_center' or 'converge' -- "
                                          "fired automatically after auto_maneuver_delay. "
                                          "Empty (default): trigger it yourself."),
        DeclareLaunchArgument("auto_maneuver_delay", default_value="8.0",
                              description="seconds to wait for the bridge + RViz to "
                                          "settle before firing auto_maneuver"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
