# 0. HARD GATE — do not skip, do not proceed on failures
./airsim_venv/bin/python tools/probe_setup.py

# 1. record ONE condition first and check it
./airsim_venv/bin/python flight/record_dataset.py --condition clear
./airsim_venv/bin/python tools/verify_dataset.py datasets/clear

# 2. only then record the rest
./airsim_venv/bin/python flight/record_dataset.py --condition all --overwrite
./airsim_venv/bin/python tools/verify_dataset.py --all

# 3. benchmark (offline — sim can be shut down now)
./airsim_venv/bin/python tools/run_benchmark.py --no-loop-closure --repeats 2

# 4. optional demo
./airsim_venv/bin/python flight/slam_live.py --method lidar --weather fog_heavy
