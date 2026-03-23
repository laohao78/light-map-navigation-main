colcon build --symlink-install --packages-select grounded_sam2
colcon build --base-paths src/gazebo_simulation
colcon build --base-paths src/sensor_driver