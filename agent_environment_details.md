# MARVEL 智能体与环境代码细节

## 1. 三层对象

### Env：全局环境

文件：`MARVEL-main (1)/MARVEL-main/utils/env.py`

- 读取 `maps_medium` 中的栅格地图，并缩小为较低分辨率。
- 地图像素被归一为：自由区 `255`、障碍 `1`、未知 `127`。
- `robot_belief` 是所有机器人共享的全局未知地图，初始为 `127`，再由初始 360°观测填充。
- 随机生成 `N_AGENTS` 个可行起点和随机初始航向。
- 使用 `sensor_work_heading()` 按受限 FoV 射线更新 belief map。
- 维护 `global_frontiers`，即已知自由区和未知区之间的边界。
- `evaluate_exploration_rate()` 用已知自由栅格数/真实自由栅格数计算探索率。
- `calculate_team_reward()` 根据本轮消失的 frontier 数计算团队奖励。

该类不是标准 Gymnasium 环境：虽然有 `step()`，但主流程实际由 `MultiAgentWorker.run_episode()` 驱动，环境没有统一的 `reset()/step()->(obs,reward,terminated,truncated,info)` 接口。

### Agent：单个机器人

文件：`utils/agent.py`

每个 Agent 保存：位置、航向、速度、最大 yaw rate、局部地图窗口、frontier 集合、图节点状态、旅行距离和 episode transition 缓存。所有 Agent 共享同一个 policy network、NodeManager 和 Env belief map，但每个 Agent 以自己的当前位置构造观测。

### MultiAgentWorker：多机器人 episode 调度器

文件：`utils/multi_agent_worker.py`

它创建一个 Env、一个局部 NodeManager、一个 GroundTruthNodeManager 和多个 Agent，然后在每个决策步中统一完成：观测、动作选择、目标冲突处理、连续运动模拟、传感更新、奖励计算、图更新和 transition 保存。

## 2. 地图与图状态

`NodeManager.update_graph()` 在机器人附近生成/更新固定分辨率节点。节点间通过碰撞检测建立邻接关系。每个 Node 记录：

- 坐标；
- 可观测 frontier 数量 utility；
- 36-bin frontier 方向分布；
- frontier 集合中的最佳观察方向；
- 36-bin 已访问航向；
- 邻居列表和访问状态。

`get_all_node_graph()` 进一步计算：

- 邻接矩阵；
- 当前节点索引；
- 当前节点的邻居索引；
- 最近有 utility 节点的 A* guidepost；
- 其他机器人占据标志。

图搜索使用 Dijkstra 找距离，A* 生成到最近有 frontier utility 节点的路径。guidepost 不是最终规划器，而是提供给网络的导航提示。

## 3. Agent 观测构造

`Agent.get_observation()` 将每个图节点转换成网络输入：

```text
节点输入 6 维 = 相对 x、相对 y、utility、guidepost、occupancy、最佳效用角度
```

另外为每个节点提供：

- 36 维 frontier 分布；
- 36 维已访问航向分布；
- 当前节点索引；
- 邻接矩阵；
- 当前节点邻居索引；
- 邻居 padding mask；
- 每个邻居的候选最佳 heading 特征。

相对坐标、utility、角度和 frontier 分布会归一化。节点数量 padding 到 `NODE_PADDING_SIZE=360`，邻居动作空间 padding 到 `K_SIZE=25`。

## 4. 动作空间与剪枝

当前 Agent 不是直接输出连续速度，而是从当前节点邻居中选择：

```text
动作 = 下一个邻居节点 + 该节点的一个候选 heading
```

每个邻居保留 `NUM_HEADING_CANDIDATES=3` 个 heading，因此最多 25×3 个联合动作。候选 heading 的来源是：

- 节点有 utility：根据 frontier 方向分布选信息量最高的 3 个 FoV 中心；
- 节点无 utility：沿 A* guidepost 方向，或沿邻居方向生成回退 heading。

策略网络输出联合动作的 log-probability。训练时按分布采样，测试时 `GREEDY=True` 使用最大概率动作。

## 5. 运动与传感器更新

选定 waypoint 后，worker 先按到达距离处理同目标冲突：如果多个 Agent 选择同一节点，后到者尝试改派到附近未占用节点。

随后：

1. `compute_allowable_heading()` 根据当前位置、目标位置、当前航向、期望航向、速度和最大 yaw rate，计算可实现的最终航向。
2. 在 `NUM_SIM_STEPS=6` 个中间步上对位置线性插值、对航向平滑变化。
3. 每个中间步调用 `Env.update_robot_belief()`。
4. `sensor_work_heading()` 从机器人位置按 0.5° 角度间隔发射射线。
5. 射线遇到障碍后停止继续观测，并把可见栅格写入共享 belief map。

当前是二维、确定性、共享地图传感模型；没有传感噪声、定位误差、通信延迟或动态障碍。

## 6. 奖励与终止

单机器人奖励由 worker 计算：

- utility reward：最终节点在当前 heading 下看到的 frontier 比例；
- trajectory reward：航向与移动方向的一致性；
- team reward：团队本轮减少的 frontier 数，另有固定惩罚；
- 完成奖励：任务完成时团队奖励增加 10。

代码还计算了 `angle_reward = cos(robot.heading - preferred_angle)`，但最终 `reward_list.append()` 使用的是 `utility_reward + trajectory_reward`，因此该朝向奖励当前没有真正加入训练奖励。

终止时主要依据 worker 中 `robot_list[0].utility.sum() == 0`。`Env.check_done()` 虽然存在，但主 episode 流程没有使用它作为主要终止判断。成功指标是 episode 在最大步数内触发 done。

## 7. 训练数据

每个 Agent 将 transition 拆成多个列表保存：当前/下一时刻局部观测、动作、奖励、done、ground-truth 观测、所有 Agent 节点索引和邻居 heading。episode 结束后，worker 把所有 Agent 的缓存拼接到统一 replay buffer。

`TRAIN_ALGO=3` 时：

- policy 只使用局部 observation；
- critic 使用 ground-truth graph；
- critic 还接收所有 Agent 当前节点和下一节点索引。

这实现了 CTDE，但 ground-truth 只能用于训练，不能进入部署策略输入。

## 8. 一次决策步的完整时序

```text
共享 belief map
  -> Agent 更新局部图
  -> 生成局部观测
  -> PolicyNet 选 waypoint-heading
  -> 解决同目标冲突
  -> 受 yaw rate 约束的连续运动
  -> 受限 FoV 射线观测
  -> 更新 belief/frontier/NodeManager
  -> 计算个人奖励和团队奖励
  -> 保存局部与特权 transition
```
