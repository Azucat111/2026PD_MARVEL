# Rule Baseline 审计

当前最新跨场景验证的 Rule 分支从 `artifacts/gppo_donor_validation_20261111/evaluation/evaluate_arm.py` 进入 `scripts/benchmark_fair_search.py::run`，后者在 allocator 的 `assign` 位置明确分支。Rule 使用 `integrations/rule_scheduling.py::choose_rule_assignment`，GPPO 使用调度器配置的生产 `ExtendedGPPOInferenceAdapter.assign`。最新实验中三臂加载同一个 6D 模型来构造环境和配对条件。Rule 分支实际选择由规则函数负责。

**先由共同执行器确定机会，再选配对。** `scheduler._high_level_update` 在控制时钟更新公开延迟状态并刷新任务。Search 构造 pending heat slot，Relay 从已知位置、连通状态和 helper/anchor 需求构造 slot。共同逻辑限制角色、容量、不可抢占分配、Safety 紧急优先级、可观测路线/可达性/时限。分配器在两个选择器之前构造同种 `TaskGraph.action_mask`。同图时基线仅在 `~action_mask` 的有限耗时项上比较，实际可行集合随双方轨迹分叉而变化。正式运行未开启额外 `candidate_budget`。单独的 donor 实验臂在纯 Search 图附加自身过滤，普通 Rule 和 GPPO 都不调用这段逻辑。

**排序是词典序。** `(-priority, raw_total_time, subtask_id, uav_id, task_index, uav_index)` 取最小值。raw total time 是 `graph.edge_features[ti,ui,2]`，来自观测旅行时间、处理时间和图的同一估计口径。非有限值跳过。空图或无合法候选返回 `None`。公开任务 ID、UAV ID 平局后才使用图内索引。Rule 的 value 为 0，选择不含采样。GPPO 则用带 mask 的 pair logits 做确定性 argmax。低层 PolicyNet、override、物理、Safety、任务生命周期、通信和评分共用。

| 容易混淆的名称 | 已核实的含义 |
|---|---|
| NearestUAV / 最近无人机 | 历史文字和某些路径选择的通俗名称。Rule 当前不是按全局欧氏最近排序。近期独立入口没有调用同名独立选择器。 |
| ShortestTotalTime | 描述当前 Rule 的第二排序键。优先级先于 raw total time，ID 决定平局。未发现最新入口以该名字导入另一选择器。 |
| `default_actions()` | `utils/simulation_runtime.py` 的低层动作入口及 PolicyNet 不可用时的回退。`scripts/run_scenario.py` 及 `scripts/evaluate_scales.py` 可使用它。公平基准检查 PolicyNet 已加载，并只替换高层配对。 |
| Native GPPO | 冻结高层 3D checkpoint 和混合配置。可用于建立环境与 native parity，不是传统规则调度器。 |
| 旧正式评测 | 历史 `artifacts/extended_round8_independent_mission_acceptance_20261015_attempt34/run_acceptance.py` 调度 `evaluate_arm.py` 的 rule/GPPO 配对，沿用 benchmark 的选择分支。其固定种子、风险门槛、统计与使用的执行器源码快照与今天不同。 |
| 旧通用单对入口 | `scripts/evaluate_extended_gppo_joint_acceptance.py` 可同时写 Search/JCT，`--final` 在 `gppo_acceptance_protocol.py` 中是保留 Search seed 的守门。其早期 heading 包装与最近 production adapter 路径不同，并行版历史 run 也不自动更新为新验收。 |

最新 joint 评分要求真实连通比至少 0.9、所有非 Relay 任务完成、Safety 已处理且连续稳定 3 秒。H300 为 300 秒，0.1 秒一 tick，3000 tick 全窗口无碰撞/禁飞违规，最终 Safety 完成。完整但失败按 300 秒，截断则不可作为已完成窗口。主比较按 `100*(1-sum(C_GPPO)/sum(C_Rule))`，不要把 Search latency、wall time 或反比速度收益混为同一指标。

新环境的配对验证使用 `handoff/cloud/paired_eval.py`，再用预先冻结、未参与选择的全套场景执行项目验收。它复用现有 Rule 及生产 adapter，验证初始公开状态、隐藏布局、配置和权重；隐藏布局只作对照核验。此前对“多个历史 Rule 别名”的结论仅限实际核对的源码和已归档记录，未知旧临时脚本不推定为正式实现。
