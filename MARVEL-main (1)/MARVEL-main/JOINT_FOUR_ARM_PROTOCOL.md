# 联合 GPPO 四臂协议与 PPO 训练准备

## 四个配对实验臂

| 臂 | 候选和提交接口 | 选择器及模型 |
| --- | --- | --- |
| R0 | 原 Search → Relay 顺序 | 原 Rule 配对排序 |
| P0 | 原 Search → Relay 顺序 | 已冻结的 November 11 guarded 6D GPPO |
| R1 | 混合 TaskGraph，每事件一次原子分配 | 同一原 Rule 排序 |
| P1 | 与 R1 完全相同 | 联合接口训练后的冻结 6D GPPO。Phase 2 先用旧 6D 模型验证兼容 |

同一公开状态下 R1/P1 使用同一候选、同一 mask 和同一提交器，策略轨迹分歧后图允许随状态变化。R1/R0 衡量接口与分配节奏，P1/R1 衡量同接口下的学习收益，P1/P0 衡量对既有 GPPO 的提升，P1/R0 检验端到端目标。报告里可额外保留 P0/R0 作为旧参照。不要把 R1/R0 的收益归因于学习。

同一 case 四臂固定地图、任务生成、障碍、真实起点、隐藏布局、MARVEL checkpoint、物理步长、Safety 与通信配置。启动前锁定栈版本、代码与权重哈希、轨迹预算和报告协议。`scripts/joint_four_arm_eval.py:18` 提供单 case 四臂 H300 入口，记录配对初始摘要、每个物理 tick 和原项目分数。P1 训练权重需要给定 checkpoint 及 SHA-256，仍须匹配 6D 协议。模型语义变更须新增协议版本后再评估。该入口在 Phase 2 检查 CLI，并用模拟 runtime 验证全 3000 tick 及截断分支。物理 H300 尚待正式试验。

## 验收

训练、开发 smoke、模型选择及正式测试 seed 各自冻结成不交集合。Phase 1/2 已见的样本进入开发名单。正式验收种子分层覆盖 30/60 × 静态/动态。首次冻结前按历史索引核对种子和资产，记录可核查范围。

每臂完整运行 300 秒、3000 个 .1 秒物理 tick。完成需要所有非 Relay 任务完成、真实通信比例至少 .9、Safety 恢复且稳定至少 3 秒。使用 `scripts/gppo_joint_completion_metric.py` 原评分。全窗口另外检查物理碰撞、禁飞和最终 Safety 状态。完整窗口失败由现有函数计为 300 秒。截断单列，完整性门不通过，配对套件在补齐前只报告不确定。保持 per-domain、pooled 以及失败类别分母。

主要端到端指标为 `100 * (1 - sum(C_P1) / sum(C_R0))`。P1/R0 的整体点估计与预注册配对 95% 区间下界应达到 15%，各域点估计须非负。项目定义的任务和安全门逐域通过，且 P1/R1 的整体配对区间下界须大于零，方可把额外收益归给学习。使用环境 seed 为簇的配对 bootstrap，汇报点估计及区间。样本数与功效在看最终数据前冻结。

## PPO 技术合同

- **状态与动作**。以高层控制事件建立部分可观测 SMDP。用同一版本快照的公共观测、已知承诺、容量和时间年龄组成 actor/critic 输入。动作是混合图中的一个合法 Search/Relay 配对。空 mask 发生的继续推进是环境转移，无额外可选择动作。本轮冻结 13D UAV、12D task 和 6D edge 输入，基于同权重初始化兼容性已验证。critic 目前缺少单独的无线拓扑、承诺容量和稳定窗口特征，先重估 value，再做公开特征消融。
- **时间与奖励**。记录每个真实决策间的 `Δt`，包含中间无候选窗口。默认有限时域取 `γ=1`，每个物理 tick 的时间代价为 `-dt/300`，到项目定义的稳定完成点停止计时，H300 末按完整安全/成功结果校正，使 episode 回报为 `-C/300`。安全门仍独立判定。若比较持续折扣，则用 `Γ_n=γ^(Δt_n/τ)` 及包含相同折扣的区间回报，作为不同实验合同。
- **Bootstrap 与优势**。保存真实 terminal/truncated 标志，`δ_n=R_n+Γ_n(1-d_n)V(s_{n+1})-V(s_n)`。按 `Γ_n λ^(Δt_n/τ)` 递推 GAE。完整实际 episode 的终端 value 为零。固定 PPO 更新预算时，基础设施截断与合法分段 rollout 分开，前者标为不完整且不混入完整回报。
- **在策略采样**。计划接入 `scripts/gppo_rollout_batch.py` 冻结 behavior model 的批处理与 `scripts/train_extended_gppo_batched.py` 的 PPO 更新入口。新增 collector 应消费同一提交器的真转移，保存稳定 ID 映射、mask、behavior logits/logp、value、duration、实际 receipt 与不成功原因。selector 通过 `JointTaskScheduler._choose_joint` 接入。重放旧 logp 时使用原 batch 与 mask，禁止随新策略重算采集时的 mask。四域训练数据量按冻结计划报告。
- **更新事务**。复用 Phase 1 指向的 guarded epoch 结构，更新前备份模型参数与 buffer、Adam 状态和步数，记录行为模型对应的 mask 上的采样 KL / 平均精确 KL / 最大单状态 KL。超过预先标定阈值或状态含非有限值则同步恢复 model 和 optimizer，只保存通过事务的新版本。原基础训练的 KL early-break 需要该事务包装。短窗口记录既不做 PPO 更新，也不做新权重选择。
