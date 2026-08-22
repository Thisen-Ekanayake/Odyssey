# One drone's last-known state, as shared with the rest of the swarm.
#
# Mirrors swarm/swarm_comms.py::DronePosition, with the world position already
# converted from AirSim's NED into the ROS ENU `map` frame. Note this is shared
# through the single AirSim client, not over a simulated radio -- AirSim has no
# RF model, so the sharing is perfect and instant. See swarm_comms.py.

string vehicle_name

# Position/orientation in `map` (ENU), spawn offset already applied.
geometry_msgs/Pose pose

# Body-frame (FLU) velocity.
geometry_msgs/Vector3 velocity

# Raw GNSS, straight from AirSim's GPS sensor.
float64 latitude
float64 longitude
float64 altitude          # metres, up positive (GPS convention, not NED)

bool armed
bool has_collided
