# The whole swarm's shared position picture for one tick.
#
# Published by swarm_state_node at a fixed rate; this is the ROS-native form of
# what swarm/swarm_comms.py::SwarmPositions.refresh() returns.

std_msgs/Header header            # stamp + frame_id ("map")

DroneState[] drones

# Formation centroid in `map` (ENU), the mean of the drone positions.
geometry_msgs/Point centroid

# Flattened row-major NxN matrix of pairwise horizontal distances (metres),
# ordered to match `drones`. N*N entries; the diagonal is zero.
float64[] distance_matrix
