# Joint TaskGraph 成熟强化学习选型

在集中协调器实际提交一个合法任务和 UAV 时定义一次决策。合法图随任务、观测、资源和时钟变化，两次有动作的决策之间可能隔多个控制边界。事件时长取真实物理秒，空图沿原控制系统推进。真实模拟状态可以按事件嵌入一个 SMDP。控制器获得延迟公共信息和当前候选图，存在历史依赖，因此实验使用部分可观测 SMDP 的无记忆逼近。

| 方法 | 适配与本轮取舍 |
|---|---|
| Double DQN | 在线共享候选网络从下一事件合法集中选动作，目标网络评估。目标同步、均衡种子组回放、Huber 和掩码均可检验。 |
| Rainbow | 组合 Double Q、优先回放、dueling、multi-step、分布回报和 noisy net。稀疏真实样本难以识别各组件，本轮采用可验证的一步 Double DQN，另有多步实验开关。 |
| Discrete SAC | 两个独立 Q 和掩码 actor 对合法动作精确求和，温度目标随候选集大小变化。离线动作分布偏移通过真实配对检验。 |
| Fitted Q Iteration | 固定数据上重复拟合 Bellman 靶可诊断小样本估值。树模型可用于后续估值对照。 |
| IQL | 在观测动作上做 expectile value，再加权行为克隆，适合针对离线动作支持不足。 |
| AWAC | 可按优势加权实执行动作并继续在线更新，依赖真实下一状态与有效分支。 |

Double Q 的目标分离依据原论文，与 Garage 的 DDQN 目标计算交叉核对。CleanRL DQN 用作回放、目标网络和训练指标的实现对照。[Double DQN](https://arxiv.org/abs/1509.06461) · [Garage DQN](https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/dqn.py) · [CleanRL DQN](https://docs.cleanrl.dev/rl-algorithms/dqn/)

Rainbow 的组件定义与作者论文及 Dopamine 核对。Discrete SAC 使用原论文的可枚举动作概率、熵项和目标价值形式。[Rainbow](https://arxiv.org/abs/1710.02298) · [Dopamine](https://github.com/google/dopamine) · [Discrete SAC](https://arxiv.org/abs/1910.07207)

离线方法分别采用批量 Q 迭代、隐式 Q 学习和优势加权在线微调目标。[FQI](https://www.jmlr.org/papers/v6/ernst05a.html) · [IQL](https://arxiv.org/abs/2110.06169) · [AWAC](https://arxiv.org/abs/2006.09359)
