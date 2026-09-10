# 卓翼杯仿真平台对 MARVEL 的启发

参考目录：`zhuoyi_cup/zhuoyi_cup`。该平台由 RflySim/UE、CopterSim、PX4 SITL、rostrans、ROS 公开接口和独立裁判运行时组成。

## 值得借鉴

### 1. 平台边界清楚：仿真、接口、裁判、参赛算法分离

- 通过 `run.sh` 统一启动和停止 ROS、CopterSim、裁判和数据桥接进程。
- 参赛程序只通过公开 ROS topic/service 与平台交互，不能直接访问裁判内部状态。
- `competition_runtime` 独立负责比赛计时、命中、突围、安全区和最终成绩。

对 MARVEL 的借鉴：建立 `sim_runtime`、`agent_adapter`、`evaluator` 三个边界。训练环境可以访问真值，但部署/评测策略必须只接收公开观测；评分逻辑不应混在策略代码里。

### 2. 配置驱动而非硬编码

参考平台将场景 YAML、车辆/碰撞 JSON、视觉传感器 JSON、rostrans 参数分开，并在启动时校验数量和队伍配置。

对 MARVEL 的借鉴：引入版本化 `scenario.yaml`，统一描述地图、机器人、任务、障碍物、通信、传感器、动力学、随机种子和评测指标；启动时做 schema 校验并把最终配置归档。

### 3. 随机种子和运行归档

`run.sh --seed` 固定随机性，每次运行创建 `logs/<timestamp>_<scene>_<difficulty>`，保存 seed、场景、配置、进程日志，并提供 `logs/latest` 快速定位。

对 MARVEL 的借鉴：每个 episode/批次记录 `scenario_id`、seed、模型 commit/hash、配置快照、软件版本、指标和异常事件；修复当前 `os.listdir()` 未排序、随机起点/航向未统一记录的问题。

### 4. 难度分级和场景矩阵

参考平台通过 `low/mid/high` 逐级改变目标运动难度，并区分 `3v3/10v10` 赛道。

对 MARVEL 的借鉴：设计可组合的难度维度，而不是单一 difficulty：机器人规模、障碍密度/速度、FoV、定位噪声、通信延迟/丢包、故障率、任务紧迫度。训练和测试用场景矩阵自动生成。

### 5. 真实接口优先，支持仿真到现实

平台公开 ENU 坐标、里程计、飞控状态、雷达目标、IMU、相机图像和吊舱状态/控制；示例程序展示位置、速度、偏航和吊舱控制。

对 MARVEL 的借鉴：定义与仿真无关的 `VehicleState`、`SensorObservation`、`ActionCommand` 和 `TaskEvent` 接口，并提供 ROS2/ROS1、Gymnasium 和离线回放适配器。MARVEL 策略接收适配后的观测，不直接依赖 Env 内部数组。

### 6. 运行时监控、健康检查和故障处理

启动脚本会检查关键进程是否存活、端口是否占用、裁判是否完成初始化，并在异常时转发关键日志、清理进程。

对 MARVEL 的借鉴：加入仿真 watchdog、进程健康状态、超时、端口隔离、自动清理、故障事件记录和 episode 级恢复；这些能力对 30–60 机长时间压力测试非常关键。

### 7. 视觉化和可观测性

平台通过 UE 标签、队伍颜色、场景对象和日志让运行状态可检查；3v3 提供相机/吊舱公开接口，10v10 则限制视觉接口以保持公平。

对 MARVEL 的借鉴：演示层与算法层分离，提供统一 dashboard/回放，显示任务状态、机器人轨迹、局部图、障碍物、通信链路、冲突和重规划事件；评测时可关闭高成本视觉输出。

## 不应直接照搬

- 黑盒二进制裁判：竞赛公平需要，但研发阶段必须提供可测试的开源评分器和事件 API。
- 固定 3v3/10v10 队伍结构：我们的目标是 30–60 架和异构集群，应使用动态 agent registry。
- 只公开位置/雷达接口：MARVEL 需要探索地图、前沿和任务观测，必须定义观测权限等级。
- 只按命中数/比赛时间评分：揭榜需求还要评估任务完成率、makespan、航程、能耗、冲突、通信和故障恢复。
- 依赖 UE/CopterSim/PX4 的单一运行栈：应保留轻量 headless 栅格仿真用于训练和 CI，再提供高保真 ROS/UE 适配器。

## 建议的平台分层

```text
Scenario & Seed Manager
        |
Simulation Runtime (地图/动力学/障碍/传感器/通信/故障)
        |
Adapter Layer (Gymnasium + ROS + replay)
        |-------------------|
Policy / Task Planner    Evaluator / Referee
        |-------------------|
Logging, Metrics, Replay, Dashboard
```

## 对当前 MARVEL 的优先改造

1. 先做 `ScenarioConfig` 和统一运行器，替代 `parameter.py` 全局常量。
2. 增加独立 `Evaluator`，把终止条件和指标从 `MultiAgentWorker` 中移出。
3. 建立 headless `SimRuntime`，先支持动态障碍、通信/故障注入和 4→10→30→60 机压测。
4. 定义 ROS/Gymnasium 双适配接口，保留原始 MARVEL episode 作为兼容模式。
5. 增加按场景目录归档的日志、seed、配置、模型和指标快照。
