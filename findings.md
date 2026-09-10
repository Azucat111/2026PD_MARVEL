# MARVEL 项目研究发现

## 需求来源

`question.txt` 要求：面向无人机集群多任务节点行为决策，融合多智能体强化学习、分布式协同优化和群体智能；支持动态/静态障碍、30–60 架规模仿真、自主任务切换和协同优化；交付算法设计、仿真平台/源码/实验数据与报告、演示系统和技术研究报告；目标是相对传统规则调度效率提升 15% 以上。

## 论文基线

- 受限 FoV 的多机器人主动探索；目标是最小化完成探索的最大轨迹代价。
- 观测：动态图、机器人位置、每个节点的 frontier 分布；节点属性包含相对位置、utility、occupancy、guidepost、最佳朝向。
- 策略：6 层 masked graph attention encoder + decoder + 方向/访问历史融合 + pointer 输出联合动作。
- 动作：邻居 waypoint 与 3 个信息增益最高 heading 的组合。
- 训练：CTDE；策略使用局部观测，critic 可使用完整图和其他机器人动作；离散 SAC 双 Q。
- 结果：90×90m、最多 8 机器人及不同 FoV/传感范围；Crazyflie 2.1 硬件验证。

## 代码调用关系（CodeGraph）

- `driver.py` 初始化 `PolicyNet`、两个 `QNet` 和 Ray `RLRunner`，集中采样与 SAC 更新。
- `RLRunner` 调用 `MultiAgentWorker.run_episode()`。
- `MultiAgentWorker` 创建 `Env`、`NodeManager`、`GroundTruthNodeManager` 和多个 `Agent`，循环执行观测→动作→冲突解决→运动模拟→奖励→transition。
- `Agent.get_observation()` 将局部图规范化、padding 到 `NODE_PADDING_SIZE=360`；`select_next_waypoint()` 从 policy log-prob 选择动作。
- `NodeManager` 负责局部图节点、frontier utility、邻接关系、Dijkstra/A* guidepost。
- `GroundTruthNodeManager` 为特权 critic 构造完整图观测。
- `Env` 使用射线传感更新受限 FoV belief，并计算团队 frontier 奖励和探索率。

## 关键基线参数

`N_AGENTS=4`、`FOV=120°`、`SENSOR_RANGE=10m`、`MAX_EPISODE_STEP=128`、`BATCH_SIZE=256`、`REPLAY_SIZE=10000`、`TRAIN_ALGO=3`；测试默认 6 机、100 张地图、greedy policy。

## 架构判断

揭榜需求应采用“高层多任务调度/分配 + 低层 MARVEL 图注意力导航探索 + 安全约束与在线重规划”的分层架构。扩展重点不是替换 MARVEL policy，而是增加任务上下文、动态障碍预测、通信/故障建模、稀疏邻域协同和规模化训练评测。
