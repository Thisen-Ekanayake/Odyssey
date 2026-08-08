#!/usr/bin/env python3
"""Merge 5 video clips into one 1920x1440 grid via ffmpeg.

Layout:
    +---------------------------+
    |                           |
    |     clip 1 (1920x1080)    |
    |                           |
    +------+------+------+------+
    |  2   |  3   |  4   |  5   |   each 480x360
    +------+------+------+------+

Each clip is scaled to fit its cell (letterboxed with black bars if its
aspect ratio doesn't match) rather than stretched or cropped. Output
duration is clamped to the shortest input clip.

Usage:
    tools/merge_clips.py --one a.mp4 --two b.mp4 --three c.mp4 \
        --four d.mp4 --five e.mp4 --output merged.mp4
"""
import argparse
import shutil
import subprocess
import sys

TOP_W, TOP_H = 1920, 1080
CELL_W, CELL_H = 480, 360


def scale_pad(label_in: str, label_out: str, width: int, height: int) -> str:
    return (
        f"[{label_in}]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[{label_out}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--one", required=True, help="large top clip (1920x1080)")
    parser.add_argument("--two", required=True, help="bottom clip 1 of 4 (480x360)")
    parser.add_argument("--three", required=True, help="bottom clip 2 of 4 (480x360)")
    parser.add_argument("--four", required=True, help="bottom clip 3 of 4 (480x360)")
    parser.add_argument("--five", required=True, help="bottom clip 4 of 4 (480x360)")
    parser.add_argument("--output", default="merged.mp4", help="output path (default: merged.mp4)")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 1

    clips = [args.one, args.two, args.three, args.four, args.five]

    filters = [scale_pad("0:v", "top", TOP_W, TOP_H)]
    bottom_labels = []
    for i in range(1, 5):
        label = f"b{i}"
        filters.append(scale_pad(f"{i}:v", label, CELL_W, CELL_H))
        bottom_labels.append(f"[{label}]")
    filters.append(f"{''.join(bottom_labels)}hstack=inputs=4[bottom]")
    filters.append("[top][bottom]vstack=inputs=2[outv]")
    filter_complex = ";".join(filters)

    cmd = ["ffmpeg", "-y"]
    for clip in clips:
        cmd += ["-i", clip]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "0:a?",
        "-shortest",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        args.output,
    ]

    print("running:", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
