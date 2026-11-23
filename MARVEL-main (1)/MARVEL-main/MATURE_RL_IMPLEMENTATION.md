# Joint TaskGraph 强化学习实现

## 同一环境执行契约

`scripts/run_mature_rl.py` 使用已验证的 Joint GPPO runner。它在调度器的合法选择回调选择一枚合法候选，从已有原子提交读取实际回执。原 Rule、动态 Safety、MARVEL 权重、任务终止条件、物理和通信评分采用原版本。未发生分配的控制边界照常执行。新记录包含行为分布、实际动作概率和决策推理开销。

`integrations/gppo/event_rl_replay.py` 从合法图提取候选角色、公开通信分量、承诺数、候选公开代价和首步路径证书、观察位置、归一化规模、冻结 actor 相对分数，以及 Relay 供体的 Search 机会成本。DDQN 与 SAC 均从随机权重训练，冻结 actor 只提供公开图排序特征。稳定 ID 用于审计，矩形批次的填充行另行掩码。

## Double DQN

```text
candidate = argmax_(b in A(s')) Q_online(s', b)
y = r + 1[nonterminal] * gamma^delta_time * Q_target(s', candidate)
loss = Huber(Q_online(s, a), y)
```

每 50 次 Adam 更新同步目标网络。正式比较采用每物理秒折扣 `gamma=1` 的一步靶，批量 32，学习率 0.0007，64 维两层 SiLU。n-step 开关按实际累积物理时间折扣。正式基线采用一步 Double DQN。

## Discrete SAC

使用两套独立候选 Q 网络及一套 categorical actor。

```text
V(s') = sum_(b in A(s')) pi(b|s') * [min(Q1_target, Q2_target)(s',b) - alpha * log pi(b|s')]
loss_actor = E_(s from replay) sum_(b legal) pi(b|s) * [alpha * log pi(b|s) - min(Q1,Q2)(s,b)]
H_target(s) = 0.65 * log(number_of_legal_actions)
```

温度根据实际合法熵与目标差更新，范围 `[1e-4, .3]`。两个算法使用相同回放、seed 组抽样序列、批量与优化次数。SAC 熵对应单位事件的训练正则，部署与评测采用合法贪心和原完整任务计分。checkpoint 包含目标网络、优化器、模型和回放 SHA。

## 有界合法探索

混合图以 0.85 混入探索，按可用角色等权分流，再在该角色内高冻结分数候选及一个最近成本候选中均匀采样。非混合探索率为 0.18，其余概率保留贪心。第一组共享种子只在首个真实混合图执行一次探索，再回到冻结 J0。第二组新种子持续探索。行为 RNG 独立于模拟场景随机种子，实际抽样概率随原子回执记录。
