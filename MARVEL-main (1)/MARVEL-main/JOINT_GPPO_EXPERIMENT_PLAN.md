# 后续可证伪实验合同（仅设计，未执行）

状态：DRAFT_FOR_NEXT_STAGE。本轮只读枚举、合成合同测试和短 CPU 采样不是该合同的训练或 H300 验收。主研究判断 `GO_CONDITIONAL`，不能据此宣布 P1 已可部署或已经优于 Rule。

## 1. 四臂与因果比较

| 臂 | 选择器 | 候选/承诺执行架构 | 权重 |
|---|---|---|---|
| R0 | 当前正式 `choose_rule_assignment` | 原顺序 Search → Relay | 同样冻结 MARVEL；高层模型只用于共同构建配置，不参与 Rule 选择 |
| P0 | 当前 production extended GPPO | 原顺序架构 | 冻结 November11 guarded 6D 推理导出 |
| R1 | Rule 在合法边上按原词典序排序，逐步填充 matching | 新 joint_v1 provider + atomic commit | 同冻结 MARVEL；不训练 |
| P1 | Event-Level Joint Search–Relay GPPO | 与 R1 **完全相同**的 provider、mask、commit、运动 | 新协议下冻结的选定高层模型 |

四个预注册 contrast：P0/R0 保留历史可比性；P1/R1 隔离相同新接口上的学习增益；R1/R0 测量接口改变本身；P1/R0 对应最终端到端目标。不要把共享执行器收益归因于 PPO。额外可报告 `P0 weights + joint interface` 零训练诊断臂，但不能用它代替 P1 或四个 contrast。

同一接口给定同一公开状态/所有权版本时，Rule 与 GPPO 必须拥有完全一致的 slot IDs、UAV IDs、合法 mask、Defer/Continue 开放条件。闭环轨迹分叉之后候选不同是正常结果；不能要求两个策略沿不同轨迹仍看到相同图。

R1 保留严格排序 `(-priority, raw total_time, taskID, UAVID, ti, ui)`，有候选时填满合法容量后 STOP。不为了给 P1 制造优势临时减弱 R1。若首版不开 Defer，则 R1/P1 都不开；以后开放须新实验版本，Rule 的 Defer 策略也在训练前冻结。

## 2. 必须固定的共同条件

四臂共用场景/地图文件、初始位置与 heading、隐藏布局生成 seed、task priorities/activation、sensor/FoV、通信半径/延迟、恒速运动、dt=.1、control period=1s、.2m五探针、.5m Shield、动态 warning 物理速度 .2m/s、Safety margin/优先级、任务 Find 完成与3s稳定评分、相同停止和失败处理。

R0/P0 继续原 scheduler；R1/P1 唯一设计差异是候选/新承诺的联合顺序与明确配置的 STOP。以下机制在四臂不变：旧任务不可任意抢占、Relay 初始/到达补充预算、arrival-before-retarget 修复、稳定释放、anchor 构造器、非 GPPO probe、Search support/deadline/reachability、物理路由与恢复。若新接口确需改变任何一项，写入版本差异表，R1/P1 同步改，不能夹带 P1 专用执行优势。

隐藏布局/真实 occupancy 只能用于离线配对核验与物理安全评分，不能进入 candidate provider 或在线 actor/critic。评估工具允许 oracle 计分，但必须与公开输入分开归档。

冻结清单必须包含 Git SHA、所有源码与配置 SHA256、地图、MARVEL、native GPPO、P0 exported model、P1 selected model、依赖锁、dtype/device、normalization、温度、mask/schema、奖励/折扣、优化器、early-stop/KL回退、预注册统计及 seed roster。生产原文件与权重在所有作业前后核对。

## 3. 阶段和计算上限

以下为下一阶段建议的**预先冻结预算**，本轮不执行。

1. **接口 gate**：CPU 合同测试、纯角色退化、空事件、并发冲突/原子回退、Safety 抢占、公开字段读取陷阱。针对不少于40个保存公开事件，对 R1/P1 provider 的 mask 做逐位 parity。只读样本可来自本轮，但不计验收样本。
2. **技术 smoke**：每域1个已知开发 seed，四臂各≤10控制步（最多16短运行），只检查承诺/物理路径/日志。无 PPO、无收益结论。任意状态污染、非法动作、模型差异或原子性失败立即停止。
3. **有界训练可学性 probe**：完成独立授权和预算冻结后，最多8个新训练 environment seeds（每域2），每 seed4个独立 policy RNG rollout，共≤32完整H300；最多4次批更新，每批行为模型完全冻结。只训练高层；复用采样/PPO/事务工具。若宏事件 transition 量过少、全组 outcome 零方差、或多次更新无合格事务，则不追加算力来凑“训练很多”。
4. **开发/选择**：固定每域4个开发 seeds（16），只允许预注册最多 initial + 4个合格更新模型进入选择。按全域任务成功/安全 gate、再 pooled C、再较早 checkpoint 平局规则选择。开发计算上限 `16 seeds × 4 arms × 5 candidate slots` 的实际简化：R0/P0/R1各运行一次复用；P1各候选各16次（最多128 H300运行）。任何超预算需新合同，不能无截止滚动试验。
5. **最终测试**：每域12个从未用于训练/开发/选择的 environment seeds（48），四臂共192个H300；单独冻结、只运行一次已选模型。建议至少2种公开地图布局/几何来源、每域每地图6例；若只使用 connected80，则结果声明严格限定该地图分布，不能称跨地图通过。新地图先做独立可行性规格/时域审查，不能在最终测试中按结果剔除困难实例。

阶段3–5仅为规模建议；开始前按实测 CPU/GPU 吞吐写 wall budget、并发数、恢复规则和截止时间，不能把本轮 CPU 用时外推为已批准预算。

## 4. Seed 隔离与样本单位

环境 seeds 分为 `TRAIN`, `DEV_SMOKE`, `DEV_SELECT`, `FINAL_TEST`，集合严格不交。policy sampling seed 独立于环境 seed；每个环境 seed 的重复 rollout 是同一个 cluster，不能当四个独立场景。地图生成/隐藏布局/初始状态的 seed 流和算法 RNG 流分别派生并记录。

在任何训练前生成加盐、可重放的 roster（固定整数→hash派生，碰撞检查），保留 roster 文件哈希；FINAL_TEST roster 由独立阶段保持不可用于模型选择。扫描历史 `plan/manifest/seed` 元数据和本轮 known seeds，记录扫描范围及排除表，不能声称查遍不存在的私有历史。历史提到的所有验收/开发 seeds（含本轮1280021/1280022/9164/1981204）均从最终池排除。

每个 case 四臂使用相同环境 seed，保存 `initial.json`、config（仅模型路径/hash差异白名单）、隐藏布局离线 digest，逐项配对。任务失败样本仍保留；结果缺失不自动从四臂 aggregate 删除。

## 5. 完整 H300 与指标定义

每臂运行完整300s（3000个.1s物理样本），即使 Search 或整个非 Relay 任务提前完成也继续，观察后续安全与通信。每 tick 保存实际 radio、所有 nonRelay 完成标志、Safety records/escapes/active、碰撞与禁飞。复用 `scripts/gppo_joint_completion_metric.py:4` 的有序物理采样检查。

完整任务成功：所有 nonRelay 任务完成，实际 connectivity≥.9，Safety已处理/释放，并连续稳定3s。C 为该完整条件首次稳定时刻；还必须满足整个H300安全及最终Safety完成。完整但失败/不安全时 `Ctilde=300`。任务未完成、通信失败、安全失败分别计数，不用一个 cap 掩盖类别。

物理截断（wall timeout、进程错误、缺帧）记为 TRUNCATED，与完整任务失败分列。最终 gate 要求零缺失完整窗口；在补齐前只能 INCONCLUSIVE。基础设施故障仅按预注册的同配置/同权重/同seed原样重跑一次，保留旧日志；若再次截断，整套不通过，不挑替代seed。不得把不安全结果当基础设施失败重跑。

每个domain和pooled分别报告：样本数/完整数/截断数，成功率、全窗安全率、Ctilde均值/中位数/配对差、Search Find均值/最后Find、非Relay完成、通信达标起点、Safety恢复、分配/空事件数、跨角色/共享UAV机会、实际选角色、Defer数、motion veto/hold比率、计算wall time。wall time是成本指标，不代替任务时钟。

## 6. 预先确定的通过门与统计

主时间降幅：

\[
I(P,R)=100\left(1-\frac{\sum_i\widetilde C_{P,i}}{\sum_i\widetilde C_{R,i}}\right).
\]

四域等样本，pooled仍使用上述总时间比，**不平均四个百分比**。报告每域和pooled全部四个 contrast；不只报告有利的 P1/R0。

最终 `PASS` 同时要求：

- 所有192臂运行有完整H300，所有臂全窗安全、可达性/任务约束按冻结规格检验；P1每case成功，不能用cap均值掩盖任务失败。R0若有完整失败仍保留cap，不删除。
- `I(P1,R0) ≥ 15%`，且按环境seed（跨域/地图若共享根seed则整cluster）配对 bootstrap 10,000次的95% CI下界≥15%。point estimate达标而CI未达为INCONCLUSIVE，不能标PASS。
- 各域 `I_d(P1,R0) ≥ 0%`；并报告同时置信区间（Holm或预注册max-stat bootstrap）。任一域点估计退化即FAIL，不用其他域补偿。CI很宽单独标不确定。
- `I(P1,R1)>0` 且95%配对CI下界>0，支持“新接口中学习有额外收益”。若 P1/R0达15%而 P1/R1不胜，结论只能是接口/共有机制贡献，不能称 GPPO 学习架构已证实优越。
- R1/R0 与 P0/R0不设“必须改善”的先验门，但必须完整报告；它们分别揭示接口效应和历史参考模型稳定性。P1/R1若任一域任务安全退化，停止部署。

上述统计是保守的拟定验收合同；正式执行前冻结，之后不得看到结果再放宽。若功效不足，在测试前依据开发方差做一次独立功效分析并提高预注册样本量；看过FINAL_TEST再扩样须另立确认性套件，旧套件如实保留。

## 7. 证伪与停止条件

- 任意 hidden truth 进入公开候选/策略、同接口Rule/GPPO动作集不一致、非法重复占用、Safety被defer、物理半径/时钟改变：技术 gate FAIL，停止训练。
- 任意 epoch KL 或非有限值越界：model+optimizer事务回退；无合格epoch不产生候选，不用开发性能覆盖gate。
- 接口 smoke中出现新安全故障，先修共享机制并重冻结；不能只给P1补特殊mask。
- 32训练rollout预算内，真实跨角色共享资源事件极少或行为回报无可识别方差：`NO_GO_CURRENT_DESIGN` 或缩小问题重新立合同，不直接扩大PPO。
- P1/R1无收益但R1/R0显著改善：主要贡献在接口，撤回“学习收益”主张；保留负结果。
- P1/R0整体未达15%、域退化或任务失败：按预定FAIL/INCONCLUSIVE报告。开发样本、已见地图小子集和本轮short smoke均不能代替最终通过。

## 8. 交付证据

保存冻结合同/roster/哈希，四臂每个物理tick记录、公开事件图、实际logp与模型指纹、动作/commit receipt与拒绝原因、按域完整分母、事务回退proof、独立指标重算脚本与最终结果。逐臂区分“原模型作用”“接口作用”“新学习增量”，不混算不同warning速度、reward协议或执行器版本。
