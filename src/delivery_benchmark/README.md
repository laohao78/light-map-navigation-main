# Benchmark使用说明

## 安装依赖

```sh
apt install psmisc
```

## 生成配送任务指令

```sh
python3 delivery_request_generator.py --num 3 --per-block 3 --output-file data/medium0411_reqs.txt --coords-file data/medium0411_coords.txt
```

## 自动化执行配送任务指令

```sh
./start_delivery.sh data/medium0411_reqs.txt
```

## 地图离线评测探索算法

这个脚本不需要启动 ROS 仿真，只读取已知 OSM 地图，直接对比 `baseline` 和 `adaptive` 的探索顺序与代价指标。

```sh
source ../../install/setup.bash
python3 evaluation/map_exploration_eval.py \
	--osm-file ../../src/utils_pkg/resource/osm/medium.osm \
	--csv-out result/map_exploration_eval.csv \
	--plot-out result/map_exploration_eval.png
```

常用参数：

```sh
# 只评估单个建筑
python3 evaluation/map_exploration_eval.py \
	--osm-file ../../src/utils_pkg/resource/osm/medium.osm \
	--building building14

# 指定机器人起点（投影坐标）
python3 evaluation/map_exploration_eval.py \
	--osm-file ../../src/utils_pkg/resource/osm/medium.osm \
	--robot-x 500120.0 \
	--robot-y 4483120.0

# 只输出 JSON，便于后处理
python3 evaluation/map_exploration_eval.py \
	--osm-file ../../src/utils_pkg/resource/osm/medium.osm \
	--json
```

输出说明：

- `result/map_exploration_eval.csv`：原始结果和 baseline/adaptive 汇总对比
- `result/map_exploration_eval.png`：路径长度差值和转向代价差值图

## 论文对比图表

`evaluation/eval1.py`、`evaluation/eval2.py`、`evaluation/eval3.py` 都支持 `--plot-out`，会在输出 CSV/JSON 的同时生成图表。

```sh
cd evaluation
python3 eval1.py --plot-out ../result/eval1_plot.png
python3 eval2.py --plot-out ../result/eval2_plot.png
python3 eval3.py --plot-out ../result/eval3_plot.png
```

图表含义：

- `eval1`：前 3 步局部代价对比
- `eval2`：目标代价、总路径代价和提升量对比
- `eval3`：消融实验各变体的均值对比

## 多尺度离线总评测（small/medium/large）

使用 `evaluation/eval_multiscale_mfvs.py` 可以一次性对三种规模地图做：

- baseline 对比
- MFVS 整体模型对比
- 关键模块消融对比

推荐在仓库根目录运行：

```sh
source install/setup.bash
python3 src/delivery_benchmark/evaluation/eval_multiscale_mfvs.py \
	--output-dir src/delivery_benchmark/result/multiscale_eval
```

也可以显式指定三张地图：

```sh
python3 src/delivery_benchmark/evaluation/eval_multiscale_mfvs.py \
	--small-osm src/utils_pkg/resource/osm/small.osm \
	--medium-osm src/utils_pkg/resource/osm/medium.osm \
	--large-osm src/utils_pkg/resource/osm/large.osm \
	--output-dir src/delivery_benchmark/result/multiscale_eval
```

输出文件说明（默认在 `src/delivery_benchmark/result/multiscale_eval/`）：

- `multiscale_detailed.csv`：按建筑逐条记录（地图规模、变体、target/full cost、steps、候选数）
- `multiscale_map_summary.csv`：按地图规模汇总，并给出相对 baseline 的差值
- `multiscale_overall_summary.csv`：跨三种地图的总体均值汇总，并给出相对 baseline 的差值
- `multiscale_summary.json`：完整结构化结果，便于论文后处理
- `multiscale_overall.png`：总体对比图（所有地图合并）
- `multiscale_per_map.png`：分地图对比图（small/medium/large）

变体命名说明：

- `baseline_fixed_order`：固定顺序基线
- `mfvs_full`：完整 MFVS
- `ablate_corner_augmentation`：去掉角点增强
- `ablate_dynamic_scoring`：去掉动态评分
- `ablate_coverage_novelty`：去掉 coverage/novelty 项
- `ablate_pruning`：去掉局部剪枝