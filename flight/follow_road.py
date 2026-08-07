#!/usr/bin/env python3
"""
Road-following demo: Drone1 flies forward at a fixed altitude, steering by
yaw-rate to keep a road centered beneath it -- so it follows curves and takes
turns instead of flying in a straight line.

No ML model here: COCO-pretrained YOLO (used elsewhere in flight/) has no
"road" class, so this uses classical HSV color thresholding + largest-contour
centroid on camera "3" (bottom_center, looks straight down) instead. The
control loop only reacts to the LEFT/RIGHT pixel offset of that centroid --
never up/down -- because whether "up" in a downward image maps to the drone's
forward or backward direction isn't guaranteed, while left/right reliably
maps to the drone's own left/right for any camera mounted without roll.

Road-color thresholds (ROAD_HSV_LOW/HIGH below) are environment-specific and
almost certainly need tuning. Calibrate first, without flying into anything:

    ./airsim_venv/bin/python flight/follow_road.py --calibrate

That hovers in place and shows the camera + binary mask windows so you can
adjust the thresholds until the road is a single clean white blob. Then:

    ./airsim_venv/bin/python flight/follow_road.py

Position the drone over a road before running (or during --calibrate) --
it does not search for a road to begin with, only to reacquire one it loses.
Press 'q' in the debug window (or Ctrl+C) to land and exit.
"""
import argparse
import time

import airsim
import cv2
import numpy as np

DRONE = "Drone1"
CAMERA = "3"        # bottom_center -- looks straight down
ALTITUDE = -12.0      # NED: negative = up (12 m -- low enough to resolve the road clearly)
FORWARD_SPEED = 3.0   # m/s, body-frame forward
SEARCH_SPEED = 1.5     # m/s, slower forward speed while the road is lost
CONTROL_HZ = 10

# HSV thresholds for "road" (asphalt): low saturation, mid-range brightness.
# Starting guess for a typical gray road vs. green lawn -- tune with --calibrate.
ROAD_HSV_LOW = (0, 0, 40)
ROAD_HSV_HIGH = (180, 60, 160)

MIN_ROAD_AREA = 400   # px; ignore tiny noise blobs

KP_YAW = 0.12          # yaw-rate (deg/s) per pixel of lateral error
MAX_YAW_RATE = 45.0     # deg/s cap
LOST_YAW_RATE = 15.0   # deg/s slow search-turn when the road isn't visible


def find_road_centroid(frame_bgr: np.ndarray):
    """Return ((x, y) pixel centroid, mask) for the largest road-colored blob, or (None, mask)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ROAD_HSV_LOW, ROAD_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_ROAD_AREA:
        return None, mask

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None, mask
    return (m["m10"] / m["m00"], m["m01"] / m["m00"]), mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true",
                         help="Hover in place and only show camera+mask windows, for tuning ROAD_HSV_LOW/HIGH")
    args = parser.parse_args()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    client.enableApiControl(True, DRONE)
    client.armDisarm(True, DRONE)

    print("Taking off...")
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(0, 0, ALTITUDE, 3.0, vehicle_name=DRONE).join()
    print(f"Hovering at {abs(ALTITUDE):.0f} m.")

    if args.calibrate:
        client.hoverAsync(vehicle_name=DRONE).join()
        print("Calibration mode: hovering, no flight. Adjust ROAD_HSV_LOW/HIGH in this file "
              "and re-run until the mask window shows the road as a clean white blob.")
    else:
        print("Following the road beneath the drone. Press 'q' in the debug window "
              "(or Ctrl+C) to land and quit.")

    try:
        while True:
            resp = client.simGetImages(
                [airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False)],
                vehicle_name=DRONE,
            )[0]
            if resp.width == 0 or resp.height == 0:
                continue
            frame = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            frame = frame.reshape(resp.height, resp.width, 3)
            h, w = frame.shape[:2]

            centroid, mask = find_road_centroid(frame)

            if centroid is not None:
                cx, _cy = centroid
                error = cx - w / 2.0   # + = road is to the right of center
                yaw_rate = float(np.clip(KP_YAW * error, -MAX_YAW_RATE, MAX_YAW_RATE))
                speed = FORWARD_SPEED
                status = f"road at x={cx:.0f} error={error:+.0f}px yaw_rate={yaw_rate:+.1f} deg/s"
            else:
                # Road lost -- turn slowly to search rather than flying blind in a straight line.
                yaw_rate = LOST_YAW_RATE
                speed = SEARCH_SPEED
                status = "road not visible -- searching"

            print(f"\r{status}" + " " * 10, end="", flush=True)

            if not args.calibrate:
                client.moveByVelocityZBodyFrameAsync(
                    speed, 0, ALTITUDE, 1.0 / CONTROL_HZ,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(True, yaw_rate),
                    vehicle_name=DRONE,
                )

            debug = frame.copy()
            if centroid is not None:
                cv2.circle(debug, (int(centroid[0]), int(centroid[1])), 8, (0, 255, 0), -1)
            cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
            cv2.imshow("Road-follow: camera", debug)
            cv2.imshow("Road-follow: mask", mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1.0 / CONTROL_HZ)

    finally:
        cv2.destroyAllWindows()
        print("\nLanding...")
        client.landAsync(vehicle_name=DRONE).join()
        client.armDisarm(False, DRONE)
        client.enableApiControl(False, DRONE)
        print("Done.")


if __name__ == "__main__":
    main()
