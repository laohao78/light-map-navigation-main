### 1. gazebo 路径配置
Gazebo 会按顺序查找：
1. $GAZEBO_MODEL_PATH
2. $GAZEBO_RESOURCE_PATH
3. 默认路径（比如 ~/.gazebo/models）
```sh
# 注意修改路径
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:/usr/share/gazebo-11
# 注意修改路径
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/Desktop/ROS2_LightMap_Delivery/src/gazebo_simulation/gazebo_simulation/models
# ---------copy到系统----------
cd ~/Desktop/ROS2_LightMap_Delivery/src/gazebo_simulation/gazebo_simulation/models
sudo cp -r * /usr/share/gazebo-11/models/
```

### 2. gazebo 启动
```sh
source install/setup.bash
ros2 launch gazebo_simulation gazebo_simulation.launch.py
# ----------------gazebo 直接启动-----------------
gazebo src/gazebo_simulation/gazebo_simulation/world/medium_osm/medium_osm.world
gazebo src/gazebo_simulation/gazebo_simulation/world/small_osm/small_osm.world
```

### 3. 启动 osrm 并测试
```sh
bash sh/osrm_first_bringup.sh
bash sh/osrm_bringup.sh
bash sh/osrm_test.sh
bash sh/osrm_test1.sh
# ------------------可视化---------------------
cd src/utils_pkg/resource/osm/script/
python3 osm_to_image.py
# 记得改
src/delivery_executor/delivery_executor/delivery_client_node.py
src/delivery_bringup/config/delivery_bringup_sim.yaml
# 里面有：
osm_routing_url: "http://10.219.235.175:5001/route/v1/driving/"
```

### 4. 启动快递系统
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
# 终端1：启动仿真环境
source install/setup.bash
ros2 launch classic_nav_bringup bringup_sim.launch.py world:=MEDIUM_OSM mode:=nav lio:=fastlio localization:=icp lio_rviz:=False nav_rviz:=True use_sim_time:=True

# 等待20秒后，终端2：启动配送系统
source install/setup.bash
ros2 launch delivery_bringup delivery_system_sim.launch.py

# 等待5秒后，终端3：启动任务记录
cd src/delivery_benchmark
source ../../install/setup.bash
python3 task_record.py --filename result/test.csv

# 终端4：启动client
source install/setup.bash
ros2 run delivery_executor delivery_client_node " Please deliver this box into unit1, building14. Please deliver this parcel into unit2, building1."
```
