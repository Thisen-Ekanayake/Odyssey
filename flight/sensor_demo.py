#!/usr/bin/env python3
"""
Single-drone sensor demo for AirSim/Blocks.
Reads: RGB camera, depth camera, LiDAR, IMU, GPS, magnetometer, barometer.

Requires LidarSensor1 in settings.json for Drone1 (already configured).

Run after the Blocks sim is up:
    ./airsim_venv/bin/python flight/sensor_demo.py
"""
import numpy as np
import airsim

DRONE = "Drone1"
ALTITUDE = -5.0  # NED: negative = up


def read_rgb(client):
    resp = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
        vehicle_name=DRONE,
    )[0]
    img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8).reshape(
        resp.height, resp.width, 3
    )
    return img, resp.width, resp.height


def read_depth(client):
    resp = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)],
        vehicle_name=DRONE,
    )[0]
    depth = airsim.list_to_2d_float_array(
        resp.image_data_float, resp.width, resp.height
    )
    return depth


def read_lidar(client):
    data = client.getLidarData(lidar_name="LidarSensor1", vehicle_name=DRONE)
    if len(data.point_cloud) < 3:
        return np.empty((0, 3))
    return np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    client.enableApiControl(True, DRONE)
    client.armDisarm(True, DRONE)

    print("Taking off...")
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(0, 0, ALTITUDE, 3, vehicle_name=DRONE).join()
    print(f"Hovering at {abs(ALTITUDE):.0f} m — reading all sensors...\n")

    # RGB Camera
    rgb, w, h = read_rgb(client)
    print(f"[RGB Camera]   {w}x{h}  "
          f"mean R={rgb[:,:,0].mean():.1f}  G={rgb[:,:,1].mean():.1f}  B={rgb[:,:,2].mean():.1f}")

    # Depth Camera
    depth = read_depth(client)
    print(f"[Depth Camera] {depth.shape[1]}x{depth.shape[0]}  "
          f"min={depth.min():.2f}m  max={depth.max():.2f}m  mean={depth.mean():.2f}m")

    # LiDAR
    pts = read_lidar(client)
    sample = pts[:2].tolist() if len(pts) >= 2 else []
    print(f"[LiDAR]        {len(pts)} points  sample={sample}")

    # IMU
    imu = client.getImuData(imu_name="", vehicle_name=DRONE)
    av, la = imu.angular_velocity, imu.linear_acceleration
    print(f"[IMU]          ang_vel=({av.x_val:.3f}, {av.y_val:.3f}, {av.z_val:.3f}) rad/s")
    print(f"               lin_acc=({la.x_val:.3f}, {la.y_val:.3f}, {la.z_val:.3f}) m/s²")

    # GPS
    gps = client.getGpsData(gps_name="", vehicle_name=DRONE)
    geo, vel = gps.gnss.geo_point, gps.gnss.velocity
    print(f"[GPS]          lat={geo.latitude:.6f}  lon={geo.longitude:.6f}  alt={geo.altitude:.2f}m")
    print(f"               vel=({vel.x_val:.3f}, {vel.y_val:.3f}, {vel.z_val:.3f}) m/s")

    # Magnetometer
    mag = client.getMagnetometerData(magnetometer_name="", vehicle_name=DRONE)
    mf = mag.magnetic_field_body
    print(f"[Magnetometer] ({mf.x_val:.4f}, {mf.y_val:.4f}, {mf.z_val:.4f}) T")

    # Barometer / altimeter
    baro = client.getBarometerData(barometer_name="", vehicle_name=DRONE)
    print(f"[Barometer]    altitude={baro.altitude:.2f}m  pressure={baro.pressure:.1f}Pa")

    print("\nLanding...")
    client.landAsync(vehicle_name=DRONE).join()
    client.armDisarm(False, DRONE)
    client.enableApiControl(False, DRONE)
    print("Done.")


if __name__ == "__main__":
    main()
