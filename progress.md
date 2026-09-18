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
