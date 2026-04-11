colcon build --symlink-install --packages-select llm_delivery
colcon build --symlink-install --packages-select delivery_executor
colcon build --base-paths src/gazebo_simulation
colcon build --base-paths src/sensor_driver