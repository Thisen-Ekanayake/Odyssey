#!/usr/bin/env bash
# Download + extract the AirSim v1.8.1 Linux environments into this workspace.
#
#   ./fetch_envs.sh                 # all envs (prompts before the big download)
#   ./fetch_envs.sh Africa_Savannah AirSimNH   # just these
#   YES=1 ./fetch_envs.sh           # skip the confirmation prompt
#   KEEP_ZIPS=1 ./fetch_envs.sh     # don't delete the .zip files after extracting
#
# Downloads are resumable: rerun to continue a partial/failed fetch. Each env
# extracts to <Env>/LinuxNoEditor/<Env>.sh — run one with:  ./run_swarm.sh <Env>
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="https://github.com/microsoft/AirSim/releases/download/v1.8.1"
ENVS_DIR="$WORKDIR/envs"            # all downloaded/extracted environments live here
DL="$ENVS_DIR/_env_downloads"        # staging dir for the zips

# name -> approx download size, for the summary. Building_99 is a broken 22-byte
# asset upstream, so it's intentionally absent. TrapCamera is a split archive.
ALL_ENVS=(Blocks Africa_Savannah MSBuild2018 ZhangJiajie LandscapeMountains AbandonedPark AirSimNH TrapCamera)
declare -A SIZE=(
  [Blocks]="0.14 GB" [Africa_Savannah]="1.11 GB" [MSBuild2018]="0.79 GB"
  [ZhangJiajie]="0.88 GB" [LandscapeMountains]="1.15 GB" [AbandonedPark]="1.66 GB"
  [AirSimNH]="2.11 GB" [TrapCamera]="3.85 GB (2 parts)"
)

# Which envs to fetch: CLI args if given, else all of them.
if [[ $# -gt 0 ]]; then ENVS=("$@"); else ENVS=("${ALL_ENVS[@]}"); fi

# --- dependency + summary -------------------------------------------------
for bin in curl unzip; do
  command -v "$bin" >/dev/null || { echo "ERROR: '$bin' is required but not installed." >&2; exit 1; }
done

echo "Will download to: $ENVS_DIR"
echo "Environments:"
for e in "${ENVS[@]}"; do printf '  %-20s %s\n' "$e" "${SIZE[$e]:-?}"; done
echo "(~12 GB total for the full set; needs roughly double that free during extraction)"
if [[ "${YES:-0}" != "1" ]]; then
  read -r -p "Proceed? [y/N] " ans
  [[ "$ans" == [yY] ]] || { echo "Aborted."; exit 0; }
fi
mkdir -p "$DL"

# --- helpers --------------------------------------------------------------
fetch() {  # fetch <filename> : resumable, only a complete download gets the final name
  local f="$1"
  if [[ -f "$DL/$f" ]]; then echo "   have $f"; return; fi
  echo "   downloading $f ..."
  curl -fL --retry 5 --retry-delay 3 -C - -o "$DL/$f.part" "$BASE/$f"
  mv "$DL/$f.part" "$DL/$f"
}

extract() {  # extract <zipfile> into ENVS_DIR, make launcher + binary executable
  local z="$1"
  echo "   extracting $(basename "$z") ..."
  unzip -q -o "$z" -d "$ENVS_DIR"
}

# --- main loop ------------------------------------------------------------
for e in "${ENVS[@]}"; do
  echo ">> $e"
  if [[ "$e" == "TrapCamera" ]]; then
    # split archive: download both parts, concatenate, then unzip
    fetch TrapCamera.zip.001
    fetch TrapCamera.zip.002
    [[ -f "$DL/TrapCamera.zip" ]] || cat "$DL/TrapCamera.zip.001" "$DL/TrapCamera.zip.002" > "$DL/TrapCamera.zip"
    extract "$DL/TrapCamera.zip"
    [[ "${KEEP_ZIPS:-0}" == "1" ]] || rm -f "$DL/TrapCamera.zip" "$DL/TrapCamera.zip.001" "$DL/TrapCamera.zip.002"
  else
    fetch "$e.zip"
    extract "$DL/$e.zip"
    [[ "${KEEP_ZIPS:-0}" == "1" ]] || rm -f "$DL/$e.zip"
  fi
done

# UE4 launchers/binaries sometimes lose the exec bit through zipping.
echo ">> fixing permissions on launchers + binaries"
find "$ENVS_DIR" -maxdepth 5 -name '*.sh' -path '*LinuxNoEditor*' -exec chmod +x {} \; 2>/dev/null || true
find "$ENVS_DIR" -maxdepth 6 -path '*/Binaries/Linux/*' -type f ! -name '*.so' -exec chmod +x {} \; 2>/dev/null || true

rmdir "$DL" 2>/dev/null || true   # remove staging dir if now empty
echo "Done. Launch one with:  ./run_swarm.sh <Env>   e.g.  ./run_swarm.sh Africa_Savannah"
