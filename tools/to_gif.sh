#!/usr/bin/env bash
# Convert an mp4 to a high-quality GIF via ffmpeg's two-pass palette workflow
# (palettegen + paletteuse with Sierra dithering) rather than a naive single-pass
# encode, which banding/dithers badly on GIF's 256-color palette.
#
# GIF has no inter-frame compression, so full source resolution/fps blows up
# fast on anything longer than a few seconds (a 73s 1920x1440@30fps clip came
# out to 567MB) -- WIDTH/FPS default to a shareable size; override for a
# closer-to-source result if the clip is short.
#
# Usage: tools/to_gif.sh <input.mp4> [output.gif] [width] [fps]
#   e.g. tools/to_gif.sh merged.mp4
#        tools/to_gif.sh merged.mp4 merged.gif 960 20
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: $0 <input.mp4> [output.gif] [width] [fps]" >&2
    exit 1
fi

input="$1"
output="${2:-${input%.*}.gif}"
width="${3:-640}"
fps="${4:-15}"

if [[ ! -f "$input" ]]; then
    echo "error: not a file: $input" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "error: ffmpeg not found on PATH" >&2
    exit 1
fi

palette="$(mktemp --suffix=.png)"
trap 'rm -f "$palette"' EXIT

vf="fps=${fps},scale=${width}:-1:flags=lanczos"

echo "generating palette from $input ..."
ffmpeg -y -i "$input" -vf "${vf},palettegen=stats_mode=diff" "$palette" \
    </dev/null >/dev/null 2>&1

echo "encoding $output (width=${width}, fps=${fps}) ..."
ffmpeg -y -i "$input" -i "$palette" \
    -lavfi "${vf}[x];[x][1:v]paletteuse=dither=sierra2_4a" \
    "$output" </dev/null >/dev/null 2>&1

echo "wrote $output ($(du -h "$output" | cut -f1))"
