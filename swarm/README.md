# swarm/

Multi-drone (Drone1-4) AirSim client scripts. All cross-import each other by flat module name
(e.g. `from swarm_comms import ...`), so they must stay siblings in this one directory.

| Script | What it does |
|---|---|
| `swarm_comms.py` | `SwarmPositions` — shared-GPS polling across Drone1-4 + local→world offset calibration; run standalone for a takeoff/share/land demo |
| `swarm_demo.py` | 3-drone line formation demo |
| `swarm_circle.py` | 5-drone ring orbit demo |
| `swarm_converge.py` | shrinks the 4-drone formation onto Drone1's live position via a uniform scale-down |
| `swarm_converge_viz.py` | 5-window live view of `swarm_converge.py`: 4 chase cams + a merged, height-shaded LiDAR map with a rotation panel |
| `swarm_lines_viz.py` | live Open3D view segmenting the 4-drone formation into 4 small squares (perimeter + center-to-midpoint lines) |
| `swarm_edge_to_center.py` | each drone flies its perimeter edge to the midpoint, turns left, then heads toward the shared center |
| `swarm_edge_to_center_viz.py` | 6-window live view of that maneuver: 4 chase cams + quadrilateral/dot tracker + merged LiDAR map |

## Run

```bash
./scripts/run_swarm.sh AirSimNH          # sim must be up first, with Drone1-4 in settings.json
./airsim_venv/bin/python swarm/<script>.py
```

## Common instructions

- All scripts assume Drone1-4 exist in `settings.json` — the current single-drone settings.json
  will make these fail; re-add the vehicles first (see root `CLAUDE.md`).
- Close a window / Ctrl+C lands the swarm and exits.
- The `_viz.py` scripts fly in a background thread and add live Open3D/chase-cam windows on top —
  closing a window only stops the viewer, the flight keeps going to completion.
