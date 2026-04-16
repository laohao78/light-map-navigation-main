# 多尺度 MFVS 离线评测分析

## 1. 图表与变量说明

本实验使用离线多尺度评测脚本 `eval_multiscale_mfvs.py`，对三种地图规模（small、medium、large）上的入口探索策略进行对比。评测对象包括固定顺序基线 `baseline_fixed_order`、完整 MFVS `mfvs_full`，以及四个消融版本：`ablate_corner_augmentation`、`ablate_dynamic_scoring`、`ablate_coverage_novelty` 和 `ablate_pruning`。

图中的所有数值均表示“相对 baseline 的差值”，即

$$
\Delta = \text{metric}_{\text{variant}} - \text{metric}_{\text{baseline}}
$$

因此，图中 baseline 对应 0 轴，负值表示优于 baseline，正值表示劣于 baseline。

变量含义如下：

- `mean_target_cost`：到达目标入口参考点之前的平均累计探索代价，越小越好。
- `mean_full_cost`：遍历完整个候选视点序列的平均总代价，越小越好。
- `mean_target_steps`：到达目标入口参考点前经过的平均候选点数量，越小越好。
- `mean_candidate_count`：平均候选视点数量，反映算法生成的搜索空间规模。
- `target_cost_delta_vs_baseline`：目标代价相对 baseline 的差值。
- `full_cost_delta_vs_baseline`：总路径代价相对 baseline 的差值。
- `steps_delta_vs_baseline`：目标步数相对 baseline 的差值。
- `candidates_delta_vs_baseline`：候选数相对 baseline 的差值。

## 2. 总体结果解读

![Overall Delta Comparison](result/multiscale_eval/multiscale_overall.png)

总体图对应三种地图规模下所有建筑的均值汇总，可以看作是对算法全局性能的概括性展示。

从结果上看，完整 MFVS `mfvs_full` 在两类核心代价上均优于 baseline：目标代价和完整路径代价都明显下降，说明该方法不仅能更快找到目标入口附近的有效视点，而且能在整体上减少无效巡航开销。与此同时，目标步数也下降，说明 MFVS 能更早地将探索顺序引导到更有价值的视点上。

消融结果进一步说明，动态评分模块是该方法的核心贡献。`ablate_dynamic_scoring` 在目标代价和总路径代价上几乎退化到 baseline，甚至略有变差，说明仅依赖角点增强或局部剪枝并不能单独带来稳定收益；真正决定视点优先级的是多因素动态评分机制。

从候选数量来看，MFVS 相关变体普遍生成更多候选视点，这意味着方法并不是通过缩小搜索空间获得收益，而是通过更丰富的候选生成与更合理的排序策略，在更大的可选空间中实现更优的探索决策。

## 3. 分地图结果解读

![Per-map Delta Comparison](result/multiscale_eval/multiscale_per_map.png)

分地图图将 small、medium、large 三种规模分开展示，能够更清楚地观察算法在不同场景复杂度下的稳定性。

在 large 地图上，`mfvs_full` 的收益最明显，目标代价与完整路径代价均显著低于 baseline。这说明当场景空间更大、候选视点更多、入口分布更复杂时，MFVS 的动态评分和局部剪枝能够更充分地发挥作用，表现出更强的全局排序能力。

在 medium 地图上，MFVS 仍然保持稳定优势，且整体趋势与 large 地图一致。这表明该方法并非只对某一特定地图结构有效，而是在中等复杂度场景下也具有较好的泛化能力。

在 small 地图上，MFVS 的收益相对有限，部分消融版本甚至出现轻微代价交换。这说明当场景较小时，候选空间本身已经有限，复杂评分机制的边际收益下降，方法在“小场景”中更容易体现为局部权衡，而不是压倒性优势。

## 4. 结论

综合来看，MFVS 的主要优势体现在中大规模地图中。它通过角点增强、动态评分和局部剪枝共同提高了视点调度效率，使探索顺序更贴近“优先接近高价值入口线索”的目标。与固定顺序 baseline 相比，完整 MFVS 在目标命中效率和全局路径开销上都取得了更优结果；而消融实验则表明，动态评分是该方法最关键的组成部分。

如果需要将这部分写入论文正文，可以将其作为“多尺度离线评测结果分析”小节直接使用。