"""3D panoptic mapping of an AirSim environment + map-guided car driving.

Pipeline (each step is its own script, all read/write ``datasets_panoptic/``):

    capture.py   fly a lawnmower over the whole env recording LiDAR +
                 segmentation + depth from a nadir camera (needs the sim,
                 PROFILE=panoptic)
    fuse.py      -> panoptic_map.npz  (voxels with class + instance ids)
    view_map.py  Open3D viewer, class / instance colouring
    drive_car.py plan a road route on that map and drive a spawned car along
                 it, checking every step against the map (needs the sim)

Scripts import each other by flat module name, like ``swarm/``, so they must
stay siblings in this directory.
"""
