"""
SLAM for the AirSim drone rig.

Two front ends over a shared pose-graph back end:

  * ``lidar_slam``  -- LiDAR-inertial: IMU-primed GICP scan-to-submap.
  * ``stereo_slam`` -- stereo-inertial: SGBM depth + ORB/PnP, windowed BA.

Both consume a ``source.SensorSource`` (live AirSim, or a recorded dataset)
and emit a trajectory + map that ``evaluate`` scores against ground truth.

Everything here is CPU-only numpy/Open3D/OpenCV/scipy -- no new dependencies
beyond what ``airsim_venv`` already ships, and nothing competes with UE4 for
the GPU.

Frame conventions (see ``geometry``):
  * All poses are 4x4 SE(3) in AirSim's NED world frame (x north, y east,
    z DOWN), with the vehicle-spawn point as the origin.
  * Quaternions are ``[x, y, z, w]`` (scipy order) everywhere. AirSim's
    ``Quaternionr`` field order is different -- convert at the boundary with
    ``geometry.airsim_pose_to_matrix``.
  * Camera *optical* frames (z forward, x right, y down) are related to the
    body frame by ``config.R_BODY_FROM_OPTICAL``.
"""
