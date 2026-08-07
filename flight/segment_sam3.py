#!/usr/bin/env python3
"""
Live open-vocabulary video segmentation + tracking on Drone1's onboard FPV
camera feed, using Meta's SAM 3 video model (Promptable Concept Segmentation)
via Hugging Face Transformers.

Unlike detect_objects.py / segment_objects.py (YOLO26, fixed 80 COCO classes,
stateless per-frame), SAM 3 is prompted with free-form text concepts
(TEXT_PROMPTS below) and is *not* stateless -- frames are fed one at a time
into a streaming Sam3VideoInferenceSession that tracks each detected instance
across frames, so the same physical object keeps the same object id (and
overlay color) as the drone scans past it.

The drone takes off, hovers at altitude, and continuously rotates in place
(slow yaw-rate scan) so the camera sweeps the surroundings, exactly like the
YOLO scripts. Every frame from camera "0" (the vehicle's own front-facing
camera) is run through SAM 3 and shown with mask + box overlays in an OpenCV
window.

Checkpoint: loaded locally from models/sam_3 (gated Meta weights, downloaded
separately -- see https://huggingface.co/facebook/sam3). No download happens
at run time.

Requires transformers>=5 (Sam3VideoModel/Sam3VideoProcessor) and a CUDA GPU
with a few GB free -- this model is much heavier than YOLO26 and will run at
a noticeably lower frame rate, especially while sharing the GPU with the UE4
sim itself.

VRAM note: the tracker keeps a growing per-object memory bank (each tracked
instance stays "alive" in memory for ~30 frames after leaving view, per the
checkpoint's max_trk_keep_alive), so GPU usage climbs with how many objects
are simultaneously tracked, not just with frame count. A broad concept like
"tree" can match dozens of instances in one AirSimNH frame and reliably OOMs
an 8GB laptop GPU once the UE4 sim (~1.5GB) and desktop/browser GPU usage are
also accounted for -- confirmed by hand on this repo's dev machine. Keep
TEXT_PROMPTS short/specific, close other GPU-heavy apps, and/or export
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before running if you widen it.

Run after the sim is up:
    ./airsim_venv/bin/python flight/segment_sam3.py

Press 'q' in the video window (or Ctrl+C) to land and exit.
"""
import threading

import airsim
import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

DRONE = "Drone1"
CAMERA = "0"      # front-center FPV camera (vehicle-mounted, not ChaseCam)
ALTITUDE = -8.0    # NED: negative = up (8 m)

MODEL_PATH = "models/sam_3"   # local checkpoint, gitignored -- no auto-download
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

TEXT_PROMPTS = ["car"]   # open-vocabulary concepts to track -- see VRAM note below

YAW_RATE_DEG_S = 10.0   # slow continuous scan while hovering

# Fixed BGR palette, cycled by object id so each tracked instance keeps a
# stable color across frames.
PALETTE = [
    (60, 180, 220), (80, 200, 80), (220, 90, 60), (200, 180, 40),
    (200, 60, 200), (60, 220, 220), (140, 90, 220), (90, 140, 255),
]


def scan(stop_event: threading.Event):
    """Continuously rotate in place so the camera sweeps the whole surroundings."""
    # msgpackrpc uses a Tornado IOLoop that is not thread-safe -- this thread
    # must own its own client connection, separate from main()'s.
    client = airsim.MultirotorClient()
    client.confirmConnection()
    while not stop_event.is_set():
        client.rotateByYawRateAsync(YAW_RATE_DEG_S, 36.0, vehicle_name=DRONE).join()


def draw_overlay(frame_bgr, processed, obj_id_to_text):
    masks = processed["masks"]
    boxes = processed["boxes"]
    object_ids = processed["object_ids"]
    scores = processed["scores"]

    annotated = frame_bgr.copy()
    for mask, obj_id in zip(masks, object_ids):
        color = PALETTE[int(obj_id) % len(PALETTE)]
        mask_np = mask.cpu().numpy()
        annotated[mask_np] = (0.5 * annotated[mask_np] + 0.5 * np.array(color)).astype(np.uint8)

    for box, obj_id, score in zip(boxes, object_ids, scores):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        color = PALETTE[int(obj_id) % len(PALETTE)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{obj_id_to_text.get(int(obj_id), '?')} #{int(obj_id)} {float(score):.2f}"
        cv2.putText(annotated, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


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

    print(f"Loading {MODEL_PATH} on {DEVICE} ({DTYPE}) ...")
    model = Sam3VideoModel.from_pretrained(MODEL_PATH, dtype=DTYPE).to(DEVICE)
    model.eval()
    processor = Sam3VideoProcessor.from_pretrained(MODEL_PATH)

    session = processor.init_video_session(
        inference_device=DEVICE,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=DTYPE,
    )
    session = processor.add_text_prompt(inference_session=session, text=TEXT_PROMPTS)
    print(f"Tracking concepts: {', '.join(TEXT_PROMPTS)}")

    stop_event = threading.Event()
    threading.Thread(target=scan, args=(stop_event,), daemon=True).start()
    print("Scanning — press 'q' in the video window (or Ctrl+C) to land and quit.")

    last_summary = None
    obj_id_to_text = {}
    try:
        while not stop_event.is_set():
            resp = client.simGetImages(
                [airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False)],
                vehicle_name=DRONE,
            )[0]
            if resp.width == 0 or resp.height == 0:
                continue

            frame_bgr = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            frame_bgr = frame_bgr.reshape(resp.height, resp.width, 3)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            inputs = processor(images=frame_rgb, device=DEVICE, return_tensors="pt")
            with torch.no_grad():
                model_outputs = model(inference_session=session, frame=inputs.pixel_values[0], reverse=False)
            processed = processor.postprocess_outputs(
                session, model_outputs, original_sizes=inputs.get("original_sizes"),
            )

            for text, obj_ids in processed["prompt_to_obj_ids"].items():
                for obj_id in obj_ids:
                    obj_id_to_text[int(obj_id)] = text

            annotated = draw_overlay(frame_bgr, processed, obj_id_to_text)

            counts = {}
            for obj_id in processed["object_ids"].tolist():
                text = obj_id_to_text.get(int(obj_id), "?")
                counts[text] = counts.get(text, 0) + 1
            summary = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())) or "(nothing)"
            if summary != last_summary:
                print(f"[Tracked] {summary}")
                last_summary = summary

            cv2.imshow("Drone FPV - SAM 3 concept segmentation", annotated)
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
