# Joint GPPO 事件级训练

分支以 Phase 2 的 `7b4144349a0b7849257e58e870e1b7e95faa3791` 为基底。训练代码接入同一个 JointPlanner、原子提交和 MARVEL 物理路径。生产模型、冻结资产、Legacy、Rule、Safety 和项目任务计分保持原接口。模型遵循 `extended-gppo-observed-heading-v2` 的 13D UAV、12D task 和 6D edge 协议。

## 采集与回放

- `scripts/joint_event_ppo.py:48` 在真实 `get_actions` 路径观察候选图，并通过独立 seeded Torch generator 从一个联合 categorical 采样。先物理索引映射到合法向量，后将动作映射回原始二维图中的稳定 slot/UAV。训练行为温度为 3，同一事件的 Search 和 Relay 共享归一化。
- 公共观测压缩保存在 `public_observations.zip`，每份观测以同一版本摘要绑定到事件。用于 PPO 的最终公开特征、六维边、任务角色、稳定 ID、逐元素 mask、完整合法 log probability 向量、所选 logp/value 保存在 `rollout.pt`。整个图动态形状逐事件保存。
- 在 `get_actions` 完成后才检查原子 receipt，校验决策步、版本、角色、flat index 和稳定 ID。未获得对应 receipt 的提案留作拒绝诊断。没有候选的控制边界和 Safety 强制运动不成为 PPO 动作，物理时间仍进入区间计费。若出现执行中断或意外截断，整条轨迹没有 H300 分数，且不进入完整回合批。
- 更新前从物理日志重新计分、重算 return、验证快照 SHA，并对同一个精确行为 checkpoint 回放存储 logits、value 和已选动作。每个事件训练时直接复用原 mask。独立审计器从每个采样事件的公开快照重建准入图，核查稳定 ID、排序和 mask。隐藏真值仅供运行环境与配对身份的离线审核，actor 只收到公开 TaskGraph。

## 目标与更新

使用有限物理时域、`gamma=1`、`lambda=1`。真实 H300 每个物理 tick 都执行正常 runtime。令 `C` 为原项目计分的完整任务耗时，完整失败由原评分器设为 300 秒。对相邻决策事件 `t_k, t_next`，令

```text
r_k = -(min(t_next, C) - min(t_k, C)) / 300
V_target_k = sum_{j>=k}(r_j)
A_k = V_target_k - old_V_k
```

最后事件的下一时刻取实际 H300 边界。首事件前的环境时间单独入账。全轨迹训练回报恰为 `-C/300`。真实通信和安全参与原任务评分，采用已观测的最终得分对闭环日志做事后区间计费。任何训练时改变完工标准、提前结束或截断当作成功，都会破坏这项等价关系。

另有显式的 D1 技术诊断在下一真实合法事件处截取片段，结算已观察的 running cost，加上该事件的冻结 critic value。它只能经 `--allow-segment` 明确加入更新，单独标记截断，且不产生 C。用户侧的主训练使用完整 H300。

PPO 重用旧 `clipped_policy_loss` 和 `FrozenBehaviorBatch`，仅训练高层 Joint GPPO。每批绑定到模型张量指纹和 checkpoint 文件 SHA。逐事件重放相同 mask，计算 `exp(new_logp-old_logp)` 及 0.2 ratio clipping，value loss 与回报对齐。每轮使用原模型的确定性 eval 状态完成梯度计算，对合法动作分布加熵正则，Adam 从新状态开始。每个 epoch 前保存模型和 Adam 的完整状态，更新后验证有限梯度和张量、sampled KL、mean exact KL、max-state exact KL。越界恢复两套状态并结束本批。限制见冻结协议。只有有效更新且张量发生变化才生成可加载 checkpoint。

```text
log πθ(a|s) = log_softmax(logitsθ(s)[legal] / temperature)[a]
d_k = log πθ(a_k|s_k) - old_logp_k
ρ_k = exp(d_k)
L_k = -min(ρ_k A_k, clip(ρ_k, 0.8, 1.2) A_k)
      + 0.25 (Vθ(s_k) - V_target_k)^2 - 0.002 H(πθ(.|s_k))
sampled_KL = mean(exp(d_k) - 1 - d_k)
exact_KL(s) = Σ_{a legal} πold(a|s) (log πold(a|s) - log πθ(a|s))
```

熵项提供合法动作分布的优化正则。按事件保存和累计的奖励仍然只有原计分对应的物理时间代价。

短批次的事件数量小，PPO 用同一批原行为数据计算固定 old logp 的有限 epoch。采集后换成任何其他起始 checkpoint 都会使该批数据失效。阶段实验只保留一个事先指定的四域批结果，D1 checkpoint 留作技术证据。完整任务收益按独立开发 seeds 的 H300 配对报告。
