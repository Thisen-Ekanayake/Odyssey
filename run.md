# 0. HARD GATE — do not skip, do not proceed on failures
./airsim_venv/bin/python tools/probe_setup.py

# 1. record ONE condition first and check it
./airsim_venv/bin/python flight/record_dataset.py --condition clear
./airsim_venv/bin/python tools/verify_dataset.py datasets/clear

# 2. only then record the rest
./airsim_venv/bin/python flight/record_dataset.py --condition all --overwrite
./airsim_venv/bin/python tools/verify_dataset.py --all

# 3. benchmark (offline — sim can be shut down now)
./airsim_venv/bin/python tools/run_benchmark.py --no-loop-closure --repeats 2

# 4. optional demo
./airsim_venv/bin/python flight/slam_live.py --method lidar --weather fog_heavy

# 5. ROS 2 cooperative mapping demo (live 4-drone merged-LiDAR octomap)
# GOTCHA: the `ros2` distrobox container has no NVIDIA GPU access (no driver,
# no Vulkan ICD -- confirmed via glxinfo/vulkaninfo). RViz's OGRE renderer
# falls back to zink-over-Mesa, which fails to create a swapchain and leaves
# the 3D view permanently black even though ROS data is flowing fine. `distrobox
# create --nvidia` to fix this HANGS indefinitely on the nvidia-integration step
# (a real distrobox bug, not worth re-attempting) -- the workaround is to force
# Mesa's llvmpipe software renderer instead. Always export this before
# `ros2 launch` in that container until the --nvidia hang is resolved:
#   export LIBGL_ALWAYS_SOFTWARE=1
PROFILE=swarm ./scripts/run_swarm.sh AirSimNH
./scripts/ros_enter.sh bash -lc 'export LIBGL_ALWAYS_SOFTWARE=1 && \
  ros2 launch airsim_swarm_bridge cooperative_mapping.launch.py auto_maneuver:=edge_to_center'
