# 源码与实验清单

路径以项目目录为基准。Git 根目录比项目目录高两级。实际本地跟踪修改涉及 12 个 Python 文件及 3 个编译缓存。Python 变更已按原始字节复制，缓存沿用基底的 tracked 状态，未纳入本地缓存改动。新代码和配置同样按哈希复制。两工作树及各类文件数见 [验证记录](HANDOFF_VALIDATION.md)，逐项 SHA-256 见 [快照清单](snapshot_manifest.json)。

| 位置 | 职责与接口 | 来源/状态 |
|---|---|---|
| `utils/policy_adapter.py`、`utils/model.py`、`load_model/MARVEL/checkpoint.pth` | PolicyNet、批量低层动作、调度器接管、冻结 actor | adapter 跟踪修改，网络和 actor 沿用基底 |
| `integrations/gppo/scheduler.py` | 触发、控制时钟、Search/Relay/Safety 预占与释放、override、物理后同步 | 跟踪修改，本地最新 |
| `integrations/gppo/event_sources.py`、`event_allocator.py` | 公开热区、Relay demand，候选建图、时限/路线/可达性过滤，配对到预占 | 跟踪修改，本地最新 |
| `integrations/gppo/task_graph.py`、`gppo_model.py` | 有效掩码与 ID 映射，AHGNN、配对 actor、critic | Git 基底，两者仍参与当前评测 |
| `integrations/gppo/heading_adapter.py`、`heading_features.py`、`opportunity_features.py` | 公开延迟航向和可选 Search 机会成本，6D/9D 模型协议 | 本地新增 |
| `integrations/gppo/mesh_relay.py`、`relay_frontier.py`、`relay_release.py` | 公开连通分量、anchor、供体连通检查与释放稳定条件 | mesh/frontier 本地新增 |
| `integrations/gppo/extended_travel.py`、`search_recovery.py`、`observed_reachability.py` | 公开路线时间、合法恢复前缀及可达性遮罩 | 本地新增 |
| `utils/simulation_runtime.py`、`dynamics_models.py`、`geometry.py`、`safety_shield.py`、`sensor_models.py`、`task_manager.py` | 连续物理、栅格碰撞、动态障碍、观测、Safety、完成事件 | 跟踪修改 |
| `scripts/benchmark_fair_search.py`、`integrations/rule_scheduling.py` | 两臂共用执行链，替换合法配对选择 | SHA 见交接清单或 Git |
| `scripts/gppo_joint_completion_metric.py`、`scripts/gppo_benchmark_domains.py` | 完整任务安全与通信计时，静态/动态域装载 | 本地评测配置和实现 |
| `scripts/train_extended_gppo_batched.py`、`scripts/gppo_parallel_rollouts.py`、`scripts/gppo_rollout_batch.py` | 真闭环采样、冻结行为、PPO 更新与重用 | 本地训练实现 |
| `configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml` | 80 m 连通地图、缓存观测、物理步长 0.1 秒；实验覆盖 30/60、静态/动态 | 本地配置与地图资产 |
| `artifacts/gppo_learnability_two_hours_20261111/guarded/` | 组内 leave-one-out、逐 epoch KL 事务回退和验证 | 冻结诊断用；原采样与本地合同路径另需恢复 |
| `artifacts/gppo_donor_validation_20261111/evaluation/` | 最新四种子三臂、原始过滤函数、复核脚本和冻结协议 | 历史运行目录保留精选文件 |
| `artifacts/extended_round8_independent_mission_acceptance_20261015_attempt34/` | 原目标的历史正式规则对照和跨 seed 聚合 | 原执行器与历史合同，成绩不能和最新套件混算 |

调用链为 `benchmark_fair_search.run → MARVELPolicyAdapter.get_actions → GPPOMissionScheduler.apply → _high_level_update`。先 `_refresh_search`，再 `_refresh_relay`。Search 调用 `SearchSlotBuilder.build`，Relay 从观测连通性与 demand/anchor 形成 slot；`EventAllocator.allocate` 构造 TaskGraph、应用候选约束，委派 Rule 或 `ExtendedGPPOInferenceAdapter.assign`。后者返回 task/UAV 配对，调度器落实预占和 override，低层动作进入 `SimulationRuntime.step`。物理前后分别走 `before_physics_tick`、`post_physics_step`，Safety、碰撞、任务完成和物理连通状态按物理时钟记录。H300 由 `joint_completion_metric` 从完整物理样本重算。

历史正式与当前诊断的逐入口差异见 [规则审计](RULE_BASELINE_AUDIT.md)。重要训练协议包括历史 `joint_physical_v1`、目标对齐后的 `joint_makespan_physical_h300_v2`、以及同 seed 组内对照。前者的成绩或 critic 权重不适合直接解释后者。最近模型输入为 UAV 13、task 12、edge 6，9D 边新增 Search 机会字段而非 radio 拓扑。最新模型身份证明、原始 optimizer 是否保留见 [模型清单](checkpoint_manifest.json)。

## 证据位置和限制

- 完整历史导览在 `handoff/codex/MARVEL_GPPO_CONVERSATION_CONTEXT_20261121.md`，另有交接计划和验收台账。
- 物理停滞恢复、Round 8/17/19 公平比较、三小时修复与验收、训练重审、组内学习与 donor 对照的精选原始摘要位于 `artifacts/` 对应目录。哪些副本删除了私人路径、原件与副本哈希各是什么，见 [快照清单](snapshot_manifest.json)。
- [原始轨迹索引](raw_trace_index.json) 含最近四套实验的原始路径、大小、SHA-256，并为完整版物理指标序列保留状态变化摘要。原始姿态、模型输入、隐藏目标布局、采样张量和失败长日志继续由本地保存。公共压缩摘要不足以逐 tick 重现全部动作。
- 公开归档计划中的路径替换会改变原始哈希，原截止时间也已过期。先读结果和源码，再用 [可移植入口](CLOUD_REPRODUCTION.md) 和新环境配置建自己的冻结实验。
- 依赖外部旧 worktree、未迁移采样包或旧 deadline 的历史入口仅作代码溯源。论文正文和原 PDF 按原授权来源获取。
