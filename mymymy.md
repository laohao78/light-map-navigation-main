```sh
# Run the robot's navigation system and simulation environment
source install/setup.bash
ros2 launch classic_nav_bringup bringup_sim.launch.py \
world:=MEDIUM_OSM \
mode:=nav \
lio:=fastlio \
localization:=icp \
lio_rviz:=False \
nav_rviz:=True \
use_sim_time:=True

# Run delivery-related nodes
source install/setup.bash
ros2 launch delivery_bringup delivery_system_sim.launch.py

# Run client to send delivery requests
source install/setup.bash
ros2 run delivery_executor delivery_executor_action_client
```


```sh
source install/setup.bash
ros2 launch gazebo_simulation gazebo_simulation.launch.py
```