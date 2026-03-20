#!/bin/bash

echo -e "\n=== Building Project ==="
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers 1
echo -e "\n=== Installation Complete ==="
