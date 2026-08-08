#!/usr/bin/env python3
"""
Live instance segmentation on Drone1's onboard FPV camera feed, using a YOLO26
-seg model (Ultralytics), COCO-pretrained -- no training needed.

Note on terminology: Ultralytics' "-seg" models are *instance* segmentation
(a separate pixel mask per detected object, same 80 COCO classes as
detect_objects.py) rather than pure semantic segmentation (one mask per class,
no per-instance distinction). This is what "a segmentation model from YOLO"
means in practice -- Ultralytics doesn't ship a semantic-only architecture.

The drone takes off, hovers at altitude, and continuously rotates in place
(slow yaw-rate scan) so the camera sweeps the surroundings. Every frame from
camera "0" (the vehicle's own front-facing camera -- not the ChaseCam used by
lidar_viz.py) is run through YOLO and shown with mask overlays in an OpenCV
window.

Weights auto-download to models/yolo26x-seg.pt on first run.

Run after the sim is up:
    ./airsim_venv/bin/python flight/segment_objects.py

Press 'q' in the video window (or Ctrl+C) to land and exit.
"""
import sys
import threading
from pathlib import Path

import airsim
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import window_recorder  # noqa: E402,F401

DRONE = "Drone1"
CAMERA = "0"      # front-center FPV camera (vehicle-mounted, not ChaseCam)
ALTITUDE = -8.0    # NED: negative = up (8 m)

MODEL_PATH = "models/yolo26x-seg.pt"   # auto-downloads here on first run
CONF_THRESHOLD = 0.35

YAW_RATE_DEG_S = 10.0   # slow continuous scan while hovering


def scan(stop_event: threading.Event):
    """Continuously rotate in place so the camera sweeps the whole surroundings."""
    # msgpackrpc uses a Tornado IOLoop that is not thread-safe -- this thread
    # must own its own client connection, separate from main()'s.
    client = airsim.MultirotorClient()
    client.confirmConnection()
    while not stop_event.is_set():
        client.rotateByYawRateAsync(YAW_RATE_DEG_S, 36.0, vehicle_name=DRONE).join()


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    client.enableApiControl(True, DRONE)
    client.armDisarm(True, DRONE)

    print("Taking off...")
    client.takeoffAsync(vehicle_name=DRONE).join()
    client.moveToPositionAsync(0, 0, ALTITUDE, 3.0, vehicle_name=DRONE).join()
    print(f"Hovering at {abs(ALTITUDE):.0f} m.")

    print(f"Loading {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    stop_event = threading.Event()
    threading.Thread(target=scan, args=(stop_event,), daemon=True).start()
    print("Scanning — press 'q' in the video window (or Ctrl+C) to land and quit.")

    last_classes = None
    try:
        while not stop_event.is_set():
            resp = client.simGetImages(
                [airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False)],
                vehicle_name=DRONE,
            )[0]
            if resp.width == 0 or resp.height == 0:
                continue

            frame = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            frame = frame.reshape(resp.height, resp.width, 3)  # BGR, matches OpenCV

            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated = results[0].plot()   # masks + boxes + labels together

            classes = sorted({model.names[int(c)] for c in results[0].boxes.cls})
            if classes != last_classes:
                print(f"[Segmented] {', '.join(classes) if classes else '(nothing)'}")
                last_classes = classes

            cv2.imshow("Drone FPV - YOLO26 instance segmentation", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        print("\nLanding...")
        client.landAsync(vehicle_name=DRONE).join()
        client.armDisarm(False, DRONE)
        client.enableApiControl(False, DRONE)
        print("Done.")


if __name__ == "__main__":
    main()
