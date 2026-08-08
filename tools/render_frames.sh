#!/usr/bin/env bash
# Render each <run-dir>/<N>/frame_*.jpg folder into <run-dir>/<N>.mp4, then
# delete that frame folder once its render succeeds. Mirrors the ffmpeg
# settings tools/window_recorder.py uses for its own atexit render, for a
# manual retry when a folder's automatic render didn't happen (e.g. after
# "Fix window_recorder cascading render failure").
#
# Usage: tools/render_frames.sh <run-dir>
#   e.g. tools/render_frames.sh recordings/20260808_234359
set -euo pipefail

FPS=30
WIDTH=1920
HEIGHT=1080

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run-dir>" >&2
    exit 1
fi

run_dir="$1"
if [[ ! -d "$run_dir" ]]; then
    echo "error: not a directory: $run_dir" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "error: ffmpeg not found on PATH" >&2
    exit 1
fi

vf="scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease,pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
err_log="$(mktemp)"
trap 'rm -f "$err_log"' EXIT

shopt -s nullglob
for folder in "$run_dir"/*/; do
    folder="${folder%/}"
    name="$(basename "$folder")"
    frames=("$folder"/frame_*.jpg)
    if [[ ${#frames[@]} -eq 0 ]]; then
        continue
    fi

    out_path="$run_dir/${name}.mp4"
    echo "rendering $folder -> $out_path (${#frames[@]} frames)"

    if ffmpeg -y -start_number 1 -framerate "$FPS" \
        -i "$folder/frame_%06d.jpg" -r "$FPS" \
        -vf "$vf" -preset veryfast \
        -c:v libx264 -pix_fmt yuv420p "$out_path" \
        </dev/null >/dev/null 2>"$err_log"; then
        if [[ -s "$out_path" ]]; then
            echo "rendered $out_path"
            rm -rf "$folder"
        else
            echo "ffmpeg produced empty output for $folder" >&2
            rm -f "$out_path"
        fi
    else
        echo "ffmpeg failed for $folder: $(tail -n1 "$err_log")" >&2
        rm -f "$out_path"
    fi
done
