# 工作进展日志

## 2026-09-04

- 读取并应用 `planning-with-files` 技能要求。
- 确认项目根目录包含 `.codegraph`，使用 `cmd.exe /c codegraph explore` 成功查询核心符号和调用路径。
- 读取 `question.txt`、README、论文文本提取文件、训练/测试入口和核心 `utils` 模块。
- 创建 `task_plan.md`、`findings.md`、`progress.md`，完成需求—论文—代码基线映射。
- 当前阶段：阶段 1（需求与基线固化）已完成；下一阶段为可扩展仿真平台设计。
- 仿真平台审查结果另存于 `simulation_gaps.md`，识别出 12 类不足，并按 P0/P1/P2 排定优先级。
- 对 `zhuoyi_cup` 参考平台完成架构对比，结论记录于 `zhuoyi_lessons.md`：重点借鉴平台边界、配置驱动、seed/日志归档、难度矩阵、公开 ROS 接口、watchdog 和评测分离。

## 2026-09-10

- 参考 `仿真平台设计文档.md` 开始阶段 2 落地，新增 `configs/`、`scripts/`、`platform_tests/` 和新的 `utils` 仿真模块。
- 配置加载器支持相对路径、必填字段校验、机器人 ID 范围重叠校验和子模块配置加载。
- 实现 `KinematicSimple`、`KinematicWithAccel`、`IdealSensor`、`CommunicationModel`、`ObstacleManager`、`TaskManager`、`SimulationRuntime`、`Evaluator`。
- `python -m pytest -q platform_tests` 结果：3 passed。
- `urban_rescue_simple.yaml` 校验结果：30 机、3 任务、2 个动态障碍。
- 完成 30 机、300 步 headless 运行；日志目录：`MARVEL-main (1)/MARVEL-main/logs/20260910_163645_urban_rescue_simple/`。
- 该结果仅验证平台闭环，不代表算法达标；当前默认动作下探索率约 0.502、发现目标 2/5，碰撞事件较多，后续需接入 MARVEL 策略、安全屏蔽和协同初始布局。
- 修复 Windows `logs/latest` Junction 清理问题；再次运行 30 机 10 步成功，确认 `logs/latest` 指向最新场景目录。
- 4 个平台测试全部通过，包含 60 机 reset + 10 步压力测试。
- 已补充 `assigned_robot_types`，并让 T1/T2/T3 的任务分配字段参与任务执行。
- Runtime `info` 已提供 `terminated` / `truncated`，评测器按整段 episode 累计通信连通率。
- Runtime `info` 宸叉彁渚?`terminated` / `truncated`锛岃瘎娴嬪櫒鎸夋暣娈?episode 绱閫氱巼銆?

## 2026-09-14 进度核验

- 使用 CodeGraph 检查 `SimulationRuntime`、`MARVELPolicyAdapter`、`SafetyShield` 调用链：安全屏蔽器已接入 Runtime，策略适配器已接入 `run_scenario.py`，但暂无专门的 PolicyAdapter 单元测试。
- 运行 `python -m pytest -q platform_tests`：`10 passed`。
- 运行 `python scripts/run_scenario.py --scenario urban_rescue_simple.yaml --seed 42 --max-steps 10 --use-marvel-policy`：30 机运行成功，生成日志目录 `logs/20260914_114341_urban_rescue_simple`。
- 本次运行指标：探索率 `0.2048`、发现目标 `0`、碰撞 `0`、通信连通率 `0`；当前 checkpoint 不存在时使用 `default_actions()` 回退，因此不能视为真实 MARVEL 策略性能。
- 阶段 2 仍为 `in_progress`。下一优先级是补齐真实 MARVEL 输入适配与 checkpoint 推理验证，再扩展动态障碍安全约束、NoisySensor、真实地图加载和 60 机压力基准。
- 
## 2026-09-19 接手与策略适配修复

- 原始 MARVEL 环境测试首次因共享图中机器人位置未精确落在节点上触发 `Dijkstra` 断言；已在 `NodeManager.get_all_node_graph()` 中将规划起点和多机器人占用位置吸附到最近图节点。
- 修复后原始环境可完整运行 128 步，并成功加载 `load_model/marvel/checkpoint.pth`（episode 43872）；一次运行最终探索率达到 `63.79%`，说明 checkpoint 可以真实执行，问题不在 checkpoint 完全失效。
- 发现 30 机适配层的第二个根因：共享图节点数达到约 `1000~1200`，超过 MARVEL 固定输入上限 `NODE_PADDING_SIZE=360`，导致负 padding 并逐动作回退。
- 已在 `Agent.get_observation()` 增加局部图裁剪与索引重映射：保留当前节点附近最多 360 个节点，并同步重映射邻接矩阵、邻居索引和 current index。
- 验证 `python scripts/run_scenario.py --scenario urban_rescue_simple.yaml --seed 42 --max-steps 10 --use-marvel-policy` 成功，日志显示 30 个机器人均生成了 policy waypoint，不再出现负维度异常或 default action 回退。
- 最新 10 步策略运行指标：探索率 `0.2344`、目标发现 `0`、碰撞 `0`、通信连通率 `0`。这说明策略推理链已打通，但多任务场景的通信/目标搜索效果仍未达标。
- `python -m pytest -q platform_tests`：`10 passed`。
- 
## 2026-09-20 多任务、通信与规模评测

- 通信模型新增最大连通分量比例 `connectivity_ratio`；Evaluator 改为累计真实连通比例，而不是只记录“是否全连通”。
- 场景支持 `ensure_initial_connectivity: true`：初始位置采用通信半径约束生成，固定种子 `42/43/44/45/46` 的 30 机初始化连通率均为 `1.0`。
- Relay 任务改为持续状态：连通率达到阈值产生 `relay_connectivity_ok`，下降时产生 `relay_connectivity_lost`，不会一次成功后永久完成。
- 新增 `utils/task_scheduler.py`：Target Search 激活后覆盖指定 rescue/explorer 机器的 waypoint；relay 机器向远端节点移动维持通信骨干；MARVEL 仍负责未被高层任务接管的探索机器人。
- 动态障碍管理器现在记录 `dynamic_obstacle_spawned` / `dynamic_obstacle_updated` 事件。
- 新增 `scripts/evaluate_scales.py`，支持固定种子、多规模、baseline/policy、策略调用间隔和 JSON 结果归档。
- 新增两项平台测试后，`python -m pytest -q platform_tests`：`12 passed`。
- 10 步固定种子评测已完成：baseline 4/30/60 机和 MARVEL policy 4/30 机均运行成功，通信连通率均为 `1.0`。
- 60 机 300 步 baseline（seed 42）：探索率 `0.6742`、目标发现 `2/5`、平均连通率 `0.9519`、碰撞 `1803`、总航程 `2011.17 m`、耗时 `54.36 s`。
- 60 机 300 步 MARVEL + 调度层（seed 42，policy interval=100）：探索率 `0.5150`、目标发现 `4/5`、平均连通率 `0.9992`、碰撞 `1155`、总航程 `922.50 m`、耗时 `185.58 s`。策略版本目标搜索和通信明显改善，但探索率下降、碰撞仍高，不能宣称整体优于 baseline。
- 评测汇总已整理到根目录 `scale_evaluation_report.md`；代码编译检查通过，平台测试当前为 `13 passed`。
