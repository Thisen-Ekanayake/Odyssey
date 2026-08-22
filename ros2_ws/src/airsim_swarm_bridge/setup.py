from glob import glob

from setuptools import find_packages, setup

package_name = "airsim_swarm_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Thisen Ekanayake",
    maintainer_email="thisenekanayake0330@gmail.com",
    description="ROS 2 bridge between AirSim and the airsim_swarm research workspace.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "bridge_node = airsim_swarm_bridge.bridge_node:main",
            "swarm_state_node = airsim_swarm_bridge.swarm_state_node:main",
            "maneuver_node = airsim_swarm_bridge.maneuver_node:main",
            "traj_recorder = airsim_swarm_bridge.traj_recorder_node:main",
            "cloud_merger_node = airsim_swarm_bridge.cloud_merger_node:main",
            "dataset_to_rosbag = airsim_swarm_bridge.dataset_to_rosbag:main",
            "airsim_rpc_check = airsim_swarm_bridge.rpc:main",
        ],
    },
)
