# 🚚 ROS2 LightMap Delivery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![ROS2](https://img.shields.io/badge/ROS2-Humble-green)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange)


基于 ROS 2、OpenStreetMap、OSRM、Nav2、LLM/VLM 的轻量级末端配送系统。项目面向“自然语言下单 -> 任务解析 -> 全局导航 -> 局部重规划 -> 入口识别 -> 配送完成”的完整闭环，支持 Gazebo 仿真与真实系统扩展。

开源协议：MIT ｜ 许可证文件见 [LICENSE](LICENSE)

## ✨ 项目简介

这个仓库整合了以下能力：

- 自然语言任务解析：将用户输入拆解为可执行任务序列。
- 地图驱动导航：基于 OSM 地图数据和 OSRM 生成全局路径。
- 局部路径优化：在动态障碍场景下对下一关键点进行重规划。
- 入口探索与识别：通过 VLM 辅助识别建筑单元门牌号。
- 仿真验证：支持 Gazebo + ROS 2 + Nav2 的完整仿真链路。

如果你想快速了解系统流程，建议先看 [0326.md](0326.md) 和 [0315.md](0315.md)。

## 🧩 核心模块

- [src/task_planning](src/task_planning) 负责把自然语言转成任务序列。
- [src/delivery_executor](src/delivery_executor) 负责执行任务状态机与业务逻辑。
- [src/delivery_bringup](src/delivery_bringup) 负责系统级启动与参数汇总。
- [src/classic_nav_bringup](src/classic_nav_bringup) 负责导航与定位系统的启动。
- [src/replanner](src/replanner) 负责局部路径重规划。
- [src/entrance_exploration](src/entrance_exploration) 负责围绕建筑物探索入口。
- [src/building_entrance_recognition](src/building_entrance_recognition) 负责入口识别服务。
- [src/utils_pkg](src/utils_pkg) 负责 OSM 数据、坐标和路径相关工具。
- [src/grounded_sam2](src/grounded_sam2) 提供开放词汇检测与实例分割能力。
- [src/delivery_benchmark](src/delivery_benchmark) 用于任务记录与评测。

## 🔁 系统流程

1. 用户通过 `delivery_executor` 输入自然语言配送请求。
2. `task_planning` 调用 LLM，将请求解析为 `NAVIGATION` 和 `EXPLORATION` 等任务。
3. `delivery_executor` 根据任务类型查询 OSM 坐标，并通过 OSRM 生成全局路径。
4. Nav2 执行逐点导航；遇到局部障碍时，`replanner` 对下一关键点进行优化。
5. 若目标只定位到楼栋，`entrance_exploration` 会沿建筑外围探索，并调用 `building_entrance_recognition` 识别单元门。
6. 任务完成后，结果可由 `delivery_benchmark` 记录和评估。

更详细的模块说明可以参考 [0326.md](0326.md)。

## 🛠️ 环境要求

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11
- Python 3.10 或兼容环境

推荐先安装 ROS 2、构建工具和项目所需的 Python 依赖，再进行编译和运行。

## 📦 安装

### 1. 初始化仓库

```sh
git clone https://github.com/EI-Nav/light-map-navigation.git --depth=1
git submodule init
git submodule update

cd src/classic_localization/FAST_LIO/
git submodule init
git submodule update
```

### 2. 安装系统依赖

项目依赖会随不同功能模块变化，通常需要先安装 ROS 2 相关依赖，再安装 Python 包：

```sh
rosdep install -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y
pip3 install -r requirements.txt
```

如果你使用源码安装，还需要按 [doc/README_old.md](doc/README_old.md) 中的说明配置 Livox SDK2、Conda 环境和感知相关依赖。

### 3. 编译

仓库提供了构建脚本，也可以直接使用 colcon：

```sh
./build.sh
```

或者使用分步脚本：

```sh
./sh/build_dependencies.sh
./sh/build_project.sh
```

## 🗺️ Gazebo 资源配置

Gazebo 会按顺序查找模型和资源路径。你可以先把仓库内模型目录加入环境变量：

```sh
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:/usr/share/gazebo-11
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/Desktop/ROS2_LightMap_Delivery/src/gazebo_simulation/gazebo_simulation/models
```

也可以直接复制到系统默认模型目录：

```sh
cd ~/Desktop/ROS2_LightMap_Delivery/src/gazebo_simulation/gazebo_simulation/models
sudo cp -r * /usr/share/gazebo-11/models/
```

## ⚙️ 配置说明

### LLM API

编辑 [src/task_planning/config/api_config.yaml](src/task_planning/config/api_config.yaml)：

```yaml
llm_api:
  key: "your_key"
  base_url: "your_base_url"
  model_name: "your_model_name"
```

### VLM API

编辑 [src/building_entrance_recognition/config/api_config.yaml](src/building_entrance_recognition/config/api_config.yaml)：

```yaml
vlm_api:
  key: "your_key"
  base_url: "your_base_url"
  model_name: "your_model_name"
```

### 常用地图数据

建筑物与单元坐标数据可以参考 [0412.md](0412.md)。如果你要更新地图或者补充楼栋信息，建议同步修改 `src/utils_pkg` 中的 OSM 相关数据源。

## 🚀 运行方式

### 1. 启动仿真导航系统

```sh
source install/setup.bash
ros2 launch classic_nav_bringup bringup_sim.launch.py \
world:=MEDIUM_OSM \
mode:=nav \
lio:=fastlio \
localization:=icp \
lio_rviz:=False \
nav_rviz:=True \
use_sim_time:=True
```

### 2. 启动配送系统

```sh
source install/setup.bash
ros2 launch delivery_bringup delivery_system_sim.launch.py
```

### 3. 发送配送任务

```sh
source install/setup.bash
ros2 run delivery_executor delivery_executor_action_client
```

### 4. Benchmark 示例

```sh
# 终端1
source install/setup.bash
ros2 launch classic_nav_bringup bringup_sim.launch.py world:=MEDIUM_OSM mode:=nav lio:=fastlio localization:=icp lio_rviz:=False nav_rviz:=True use_sim_time:=True

# 终端2
source install/setup.bash
ros2 launch delivery_bringup delivery_system_sim.launch.py

# 终端3
cd src/delivery_benchmark
source ../../install/setup.bash
python3 task_record.py --filename result/test.csv

# 终端4
source install/setup.bash
ros2 run delivery_executor delivery_client_node "Please deliver this box into unit1, building14. Please deliver this parcel into unit2, building1."
```

## 📌 典型流程

- 只做导航：发送带有明确楼栋/单元坐标的任务，系统会直接规划到目标点。
- 楼栋到单元：如果只有楼栋信息，系统会先导航到楼栋，再进入探索识别流程。
- 动态障碍：局部路径被阻挡时，`replanner` 会尝试修正下一关键点，而不是重算整条路径。

## 🗂️ 项目结构

```text
src/
	building_entrance_recognition/
	classic_localization/
	classic_navigation/
	delivery_bringup/
	delivery_executor/
	entrance_exploration/
	grounded_sam2/
	replanner/
	task_planning/
	utils_pkg/
```

## ❓ 常见问题

### Gazebo 模型找不到

确认 `GAZEBO_MODEL_PATH` 和 `GAZEBO_RESOURCE_PATH` 已正确设置，或者已将模型复制到 `/usr/share/gazebo-11/models/`。

### LLM 或 VLM 调用失败

确认 API Key、Base URL 和模型名已经正确填写，并且网络可以访问对应服务。

### 任务规划结果不合理

检查 OSM 数据、楼栋单元坐标以及任务输入是否匹配。你可以先在 [0412.md](0412.md) 中核对坐标，再回到 `task_planning` 调整提示词或解析逻辑。

## 📄 开源协议

本项目采用 MIT License。你可以自由使用、修改和分发，但需要保留原始版权和许可证声明。完整协议请查看 [LICENSE](LICENSE)。

## 🙏 致谢

本项目基于并借鉴了 ROS 2、Gazebo、Nav2、OpenStreetMap、OSRM、GroundingDINO、SAM2 等开源项目的工作。
