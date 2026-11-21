# MARVEL + GPPO 完整对话上下文与接手总结

整理日期：2026-11-21（Asia/Shanghai）  
最后一轮实验结束：2026-11-11。  
本文件是当前对话与本地证据的上下文总结，不是逐字聊天导出。原始交接、历史台账和实验文件仍保留；早期回复中无法重新确认的细节不补造。2026-11-21 本次只核对与整理，未训练、未运行仿真、未提交或推送。

## 1. 先读这一页：目前到底做到哪了

1. 项目是 **MARVEL 低层探索控制 + GPPO 高层任务分配 + 本地扩展运行时**，需要支持 30–60 架、动态/静态障碍、自主任务切换与协同。
2. 技术目标包含：相对同条件传统规则调度，完整任务执行效率提升至少 15%。用户后来明确：只要技术成果，不需要 PPT、演示系统或正式技术报告；本次上下文总结是另行明确要求的交接文档。
3. 不能说项目已全面通过验收。某些已知场景和测试子集超过 15%，但结果会随模型、场景、指标、执行器版本变化；不能挑子集代替整体结论。
4. 当前最近实验：冻结同一 GPPO 模型，验证一个公开通信拓扑 donor 约束，4 个新 seed × 3 策略 = 12 次完整 H300。全部安全、配对一致，**约束增量收益为 0%**。不建议据此推广该约束。
5. 这 4 组上固定 GPPO 比规则总时间短 20.39%，但原 GPPO 与加约束 GPPO 完全持平；不是新约束的收益，也不是新一次训练的收益。
6. 同权重 GPPO 在此前另外 3 组新 seed 上反而比规则慢 4.17%。把这 3+4 组只作描述性合计，规则 185.1 秒、GPPO 167.4 秒，降幅约 9.56%。这不是预注册的原八 seed 正式验收。
7. 已实证并修复一项**诊断训练器**稳定性缺口：逐 epoch 保存模型与 Adam，KL 超限后恢复上一步，而非仅 break 后保留越界权重。没有据此证明所有历史退步都来自该问题。
8. 当前没有保留自动计算任务。最近实验计算进程和临时防睡眠 helper 均已结束；未改永久电源设置。以后开始新工作仍须重新检查现场状态。

## 2. GitHub 仓库与本地工作区

### 2.1 两个 GitHub 仓库的角色

| 角色 | GitHub | 本地 remote | 注意 |
|---|---|---|---|
| 项目团队仓库 | https://github.com/Azucat111/2026PD_MARVEL | `team` | 用户自己的项目协作仓库 |
| MARVEL 上游官方仓库 | https://github.com/marmotlab/MARVEL | `origin` | 原始探索方法；不要擅自向上游推送 |

2026-11-21 实查 `git remote -v`：

```text
origin https://github.com/marmotlab/MARVEL.git (fetch/push)
team   https://github.com/Azucat111/2026PD_MARVEL.git (fetch/push)
```

这是本地 remote 配置核验，**不是**远端分支最新提交、私有性或工作树已上传的证明。本次没有 fetch/push。原交接记录说用户曾在 2026-10-09 推送 `team extended-scalability`，但不能据此认为后来的本地修改和 artifacts 已在 GitHub。

### 2.2 路径、分支、HEAD

- 实际 Git root：`<local-home>/MARVEL_INTEGRATION`。
- 实际项目目录：`<local-home>/MARVEL_INTEGRATION/MARVEL-main (1)/MARVEL-main`。
- Windows 路径：`\\<local-home>\MARVEL_INTEGRATION\MARVEL-main (1)\MARVEL-main`。
- 分支：`extended-scalability`。
- 2026-11-21 实查 HEAD：`5143fbdc4bdadf907f59c433e81ab11b2e98c6dc`。
- 原 Claude 交接的历史 HEAD 为 `e7246a8`，更早集成基线是 `gppo-integration @ b7a167e`。不要将这些旧值当作当前 HEAD。
- 当前工作区不是 clean。2026-11-21 `git status --porcelain=v1 --untracked-files=normal` 统计为 15 条 tracked modified、425 条 untracked 项；这是状态条目数，**不是实际文件总数**，目录会被合并显示。
- 本次重新比对 2026-11-10 修正后冻结清单中的 478 个源码/评估文件，差异为 0。脏工作树包含此前积累的真实工作，不能清理或 restore。

### 2.3 环境与运行方式

- WSL 发行版：`Ubuntu-22.04`，原地本地工作，别重新 clone 或搬到 Windows 文件系统。
- Python：`<local-home>/miniconda3/envs/marvel/bin/python`。
- 2026-11-21 轻量核验：PyTorch `2.5.1`，CUDA 可用，GPU 为 `NVIDIA GeForce RTX 4060 Laptop GPU`。
- Windows 工具 shell 是 PowerShell，但仿真/Python 应通过 WSL 的上述环境执行。
- 推荐模式（路径有空格与括号，必须正确引用）：

```powershell
wsl.exe -d Ubuntu-22.04 --cd '<local-home>/MARVEL_INTEGRATION/MARVEL-main (1)/MARVEL-main' -- <local-home>/miniconda3/envs/marvel/bin/python -B <script.py>
```

- `-B` 用于避免顺手产生新的 pycache；已经存在或 tracked 的 pyc 也不能擅自清理。
- 内部日志多用 UTC；用户时间应换算 Asia/Shanghai（UTC+8）。本地 Conda Python 与 Windows Python 不能混用。

## 3. 用户目标、授权演变和偏好

### 3.1 原始技术目标

原始需求位于 Git root 的 `question.txt`，开发计划位于同目录 `task_plan.md`。原文目标包括：

1. 相比传统规则调度，集群任务执行效率提升 15% 以上。
2. 支持动态、静态障碍物场景。
3. 支持 30–60 架无人机仿真。
4. 实现自主任务切换和协同行为优化。

原始文档还列了演示、技术报告等交付物，但用户后来明确移除 PPT/演示/正式报告要求，只达成技术目标。用户曾临时要求可视化给老师看、参照 IROS 常见画法，并因链接打不开要求放桌面；这不等于永久恢复 PPT/正式报告任务。旧图位置和具体文件本次没有逐个核验。

### 3.2 授权如何变化

- 最初：只读接手、只读诊断；禁止训练、生产代码修改、重复长仿真、commit/push。
- 随后：允许独立离线诊断脚本和报告；仍不改生产链。
- 再后来：用户明确允许自主修复、训练和独立验收，不希望中途反复询问；先固定指标、检查训练问题，再花预算训练。
- 多轮给予 30/40/60 分钟、1 小时、3 小时、2 小时等预算；每个预算属于对应轮次，不应跨轮累加或假定过期预算仍有效。
- 在模型公平比较阶段用户又明确“先别训练”；应遵守具体阶段约束。
- 最近的指令是**实际验证现有 donor 拓扑约束**。该轮已完成，没有新训练或生产集成。

### 3.3 持续有效的要求

- 优先复用已有论文、开源代码、检查点、实验 CSV、JSON、轨迹、失败记录。
- 不要重复造规划器、PPO、GAE、collation 等已有轮子。
- 不要把 wall time 很长等同于有效训练样本很多。
- 已完成的同条件仿真不要无理由重跑；修改实验条件的因果探测须单独命名、保留旧数据。
- 有结果立即汇报，不等全部结束再统一展示。
- 不要只停在计划与汇报；完成当前授权工作。但不能为了达标篡改指标、挑 seed、泄露真值或隐瞒失败。
- 不擅自 commit、push、merge、rebase、reset/clean、删除旧 artifacts 或批量 `git add .`。
- `marvel_native`、冻结模型、安全半径与 SafetyShield 不能为了 extended 成绩而改坏。不要通过穿墙、缩小半径、关闭安全机制“解决”。
- 内部多代理用过：`reuse_review`、`review_fix_training`、`audit_policy_information`。它们的审查 JSON 是可追踪证据，不能替代原始实验验证。

## 4. 论文、上游代码与本地实现的关系

### 4.1 已确定来源

- MARVEL 上游：https://github.com/marmotlab/MARVEL。
- 官方训练入口 `driver.py`，官方低层算法为离散 SAC，不是本地的高层 GPPO。
- 用户提供的 GPPO 论文：**Multi-UAV Dynamic Task Assignment Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication**。
- DOI：https://doi.org/10.1109/TASE.2025.3593125。
- 作者：Jin Yu、Ya Zhang、Changyin Sun。
- 用户 PDF 路径：`<windows-local>`。
- 已提取文本：`artifacts/original_goal_paper_audit_20261110/gppo_paper.txt`。

**没有核实到 GPPO 作者官方仓库**，这不等于证明它不存在。不能把 `<local-home>/MARVEL/utils/gppo_ppo.py` 或当前 `integrations/gppo/` 的本地实现说成作者官方开源代码。代码注释中的 `paper-faithful` 也不能证明完整复现。

### 4.2 本地学习问题已经不同

论文的主要训练环境按处理时长、UAV 可用时间、任务先后约束更新抽象调度状态；奖励为估计 makespan 的差值。当前系统则包含真实 MARVEL 感知、受限 FoV、隐藏目标发现、连续运动、避障、动态 Safety 和移动 Relay 的物理闭环。

本地完整任务指标还含真实通信比例与 3 秒稳定窗口；论文的“弱通信”不能直接等同于此处“移动 helper 恢复最大连通分量到 90%”的目标。网络和优化器原语可复用，不代表插上后学习任务已经相同。

论文印刷页 19116 的表 II / 图 8 曾直接渲染核对：batch size 512、max iterations 2000、学习率 2e-4、discount .99、clip .2、entropy .01；四种规模各 100 随机实例。100 实例不能说成仅 100 个训练回合，iterations/episodes/updates 也不能直接与本地准确换倍数。

### 4.3 其他已参考且不要重新移植的资产

- MAPPO 数学/available-actions/returns 参考：https://github.com/marlbenchmark/on-policy。
- RLOO 组内 leave-one-out 基线参考：https://github.com/huggingface/trl/blob/main/docs/source/rloo_trainer.md。
- SimPy 同时刻事件顺序、CBBA-PR/Relay 分配只作设计参考，没有安装或整体移植。
- 已有本地 `FrozenBehaviorBatch`、`existing_update_function`、并行采样、GAE 和 collation 校验；不要为同一问题再造训练框架。

证据：`artifacts/gppo_training_reaudit_20261110/paper_task_review.json`、`artifacts/gppo_three_hour_20261110/reuse_review/`。

## 5. 架构、控制权限与关键文件

高层 GPPO 选择可行的 `(subtask, UAV)` 配对；低层 MARVEL/既有运动控制执行路径。Search、Relay、Safety 的触发、容量、预占、释放和很多 anchor 逻辑由共同调度器决定。

关键位置：

| 文件/模块 | 作用 |
|---|---|
| `integrations/gppo/scheduler.py` | Search/Relay/Safety 调度、override、任务生命周期 |
| `integrations/gppo/event_allocator.py` | 生成当前事件任务图、可行候选、预占 |
| `integrations/gppo/gppo_model.py` | AHGNN、pair actor、graph critic |
| `integrations/gppo/task_graph.py` | 原始/归一化特征、mask、ID 映射 |
| `integrations/gppo/heading_features.py`、`heading_adapter.py` | 公开延迟航向特征、实际推理输入 |
| `integrations/gppo/opportunity_features.py` | 9D Search 机会成本扩展 |
| `integrations/gppo/mesh_relay.py` | 公开通信分量、anchor、LCC 增益和 donor 连通检查 |
| `integrations/gppo/relay_release.py` | 反事实释放和稳定计数 |
| `integrations/gppo/relay_frontier.py` | 已知路径/前沿/forecast LCC 等原语 |
| `integrations/gppo/search_recovery.py`、`extended_travel.py` | 观测约束的恢复前缀与旅行估计 |
| `utils/policy_adapter.py`、`simulation_runtime.py` | 实际控制入口与物理 tick |
| `utils/dynamics_models.py`、`geometry.py`、`safety_shield.py` | 运动、真实碰撞判断、安全屏蔽 |
| `scripts/benchmark_fair_search.py` | 共执行条件规则/GPPO 比较 |
| `scripts/gppo_joint_completion_metric.py` | 完整 H300 物理指标 |
| `scripts/train_extended_gppo_batched.py` | 既有 PPO 更新实现 |
| `scripts/gppo_parallel_rollouts.py`、`gppo_rollout_batch.py` | 采样隔离、冻结行为批、更新复用 |

最近实际模型输入为 13D UAV、12D task、6D edge。保存文件名 `actor_inputs_9.json` 不代表实际模型吃了 9D：评估中逐项断言真实模型输入等于保存数据前 6 维切片。

UAV/task 中有相对几何位置与部分执行状态；没有显式 radio 邻接、完整 UAV→anchor 承诺或释放计数。**不能因此断言模型完全看不到通信信息或数学上不可学习**，因为位置、任务位置和 mask 可间接携带信息。

最近 80 条训练记录中 precedence 全零、完成标记与若干字段恒定；实际图全为纯 Search 或纯 Relay，不包含可一起竞争的跨角色任务链。模型能选任务和 UAV，但不能自主选择 wait、修改 anchor、主动 release，或控制 Search/Relay 的整体触发顺序。这是任务接口限制，尚非失败唯一根因的证明。

9D 第 19 轮新增的 3 维分别是 Search 名义接近 ETA、移走候选 UAV 对其余 Search 的平均 ETA 后悔值、context flag；**不是通信拓扑或割点特征**。已有 radio 原语主要在 controller/mask 内，别误称整个系统没有这些能力。

## 6. 最初的 60-UAV 停滞及恢复结论

原始证据：`artifacts/extended_postfix_live_gppo_20261002T151131Z/`，优先 `search_progress.csv`、`stall_trace/stall_trace.csv` 与关联材料。

- Search UAV43 从 step43 持续停车，距 heat0 约 117.21m。
- Search UAV0 从 step56 持续停车，距 heat1 约 370–410m。
- route_kind=override，moved_m=0，grid_blocks=10，SafetyShield 未拦截。
- 首轮分类为 `RESIDUAL_PARKING`。

离线重建见 `artifacts/offline_search_recovery_20261009/OFFLINE_SEARCH_RECOVERY_REPORT.md`。

### 6.1 已证明的真实原因

Search override 直接执行非空 BFS hop；它没有保证**真实当前位置到图接入节点/next-hop 的段**合法。图节点到下一跳合法，不代表真实 UAV 到该跳合法。失败后重复同一命令，runtime 拒绝运动却不提交新 heading/position。

这不是再次证实 360-node 窗口 decode 错配：该旧 bug 已在早期修复，本案 Search override 直接读图坐标。也没有 current heading 导致错误平移的证据；本运动模型平移朝目标，desired heading 可独立控制。

实际模型：速度 1m/s、physics dt=.1、每 tick 最大平移 .1m、yaw rate35；radius=.2m 的栅格碰撞是中心＋4个轴向偏移五探针、floor 格坐标，不能误说成精确圆盘 Minkowski 检测。

| UAV | 停滞真实位置 | 原 hop | 首拒绝命中格 |
|---|---|---|---|
| 43 | (149.6404,74.7832) | (156,80) | (149,75)，+y探针 |
| 0 | (124.7546,75.2119) | (128,72) | (125,75)，+x探针 |

### 6.2 真实规则下有局部合法路径

UAV43 重接 hop：
`(149.6404,74.7832) → (150.1404,74.7832) → (155.1404,79.7832) → (155.6404,79.7832) → (156,80)`，8.491m、110 ticks。

UAV0 重接 hop：
`(124.7546,75.2119) → (124.7546,74.7119) → (127.2546,72.2119) → (128,72)`，4.810m、60 ticks；该绕行暂时远离原 heat。

两者另有局部朝 heat 接近路径：
- UAV43：起点 → (151.1404,74.7832) → (151.6404,74.2832)。
- UAV0：起点 → (124.7546,76.7119) → (124.2546,77.2119)。

有界搜索半径12m、分辨率.5m、8邻接、最多6000扩展；连续段按真实五探针边界规则核验，不是只看落点。原报告包含命令/heading/保持mission数与舍入敏感性复核。

### 6.3 不能越过的证据边界

历史 full heading、belief、完整图、其他59机状态等缺失。真值地图只作离线诊断 oracle，证明物理路径存在，不证明当时 UAV 观测足以选择它。最终早期报告状态为：

```text
物理恢复可行性 COMPLETE
可观测恢复可部署性 INCONCLUSIVE
OFFLINE_SEARCH_RECOVERY_FEASIBILITY = INCONCLUSIVE
```

随后才实现观测型占据语义、真实姿态接入、拒绝反馈、合法感知/运动前缀和回归验证；不要把早期 oracle 路径直接部署。原位转向、瞬移、穿墙不是允许的替代动作。

## 7. 历史实现成果与不要重复的路线

以下为历史台账记录，不是 2026-11-21 重新执行的测试：

- `marvel_native` 冻结 parity 曾通过；保留 native 的感知/控制/任务语义。
- 360节点局部窗口的索引空间解码错配已经修复；新节点使用 .2m clearance。
- 扩展采用 shared planning graph 缓存、heap Dijkstra、索引优化及 CUDA batch 等已验证资产。
- 扩展观测型 FREE/occupied 语义、当前 own footprint 证书、真实位置到图节点接入与 rejection-aware recovery。
- 真实同时 tick 的 swept-relative UAV 近碰 veto、合法悬停，保持实际 Shield 最终权威。
- extended 旅行估计不再用 `Dj(start,node)+Euclidean(node,goal)` 对全节点取 min 导致退化成直线距离，而重用已知路由与明确 unknown residual。
- Relay peer-aware 观测路径、known-grid/observed-frontier fallback、bridge/retarget/release 控制样例通过。
- 动态障碍速度单位曾标 .2m/s 实际 .02m/s；修正前后不能混池，检查真实 config 不只看文件名。
- coverage 的 ALL-map 分母与 observed-free 分母不能混用；人工 activation-state 不是 natural reset。
- 已有 H80 短烟测出现 GPPO 比规则差很多，且旧碰撞后来修过；不要拿旧 smoke 当最终 H300 无碰撞验收。
- full-edge swept clearance 曾慢约25倍、某些 raycast vectorization/CPU batching 无稳定收益，已有失败证据，别无新原因反复重试。

历史测试数字分属不同版本：166pass/1旧fail、148/151/172 focused/native、516pass/1skip 等，不可合并或宣称全是当前通过。本次总结未重新运行测试。

详细演变见 `handoff/codex/PROJECT_ACCEPTANCE_LEDGER.md`、`TECHNICAL_BASELINE_PLAN.md`、`EXTENDED_SEARCH_RECOVERY_PLAN.md`。

## 8. 第8、17、18、19轮及早期验收应如何理解

### 8.1 这些“轮”不是19次相同条件从零训练

| 标签 | 该目录新增轨迹/批次 | horizon | 初始化关系 |
|---|---:|---:|---|
| attempt8 | 16条 / 2批 | H80 | 从attempt5/checkpoint001续接，reset critic |
| attempt17 | 128条 /16批 | H80 | 从attempt15/checkpoint007续接 |
| attempt18 | 256条 /32批 | H150 | 从attempt17/checkpoint015续接，reset critic |
| attempt19 | 256条 /32批 | H150 | 从initial_opportunity.pth初始化的9D扩展，保留旧网络能力，不是全网从零 |

轮次之间有继承、critic/Adam重置、输入6→9维扩展、恢复中断等不同操作，不能简单说“一直到19每一轮都是完全相同续训”。也不能把文件夹编号当总训练次数，或把复制来的history重复计入新增训练。

用户多次追问第8轮是否训练少、继承模型什么意思、为何没及时停、18/19是否从零训练。明确答案是：第8轮本身新增量确实远小于17/18/19，但它继承了此前模型；训练更久不保证在新指标/执行器/场景上更好。具体累计时间需按 lineage 去重、区分采样/梯度/中断，不可凭当前mtime猜。

### 8.2 第8/17/19轮同条件比较

权威目录：`artifacts/extended_checkpoint_common_comparison_20261015_attempt33/`。

| 模型 | 该历史suite的Search Find时间降幅 | 同suite完整任务时间降幅 |
|---|---:|---:|
| round8 | -6.00% | +16.34% |
| round17 | +14.04% | +10.27% |
| round19 | -0.21% | +9.77% |

这说明“Search指标最好”与“完整任务最好”可能不是同一模型。对话曾提及14.13%、-18%、第8轮约10%等阶段数字；它们不能直接合并。此次核对的这份最终同条件文件里 round17 Search 点估计是14.0376%，不是精确14.13%。早期精确14.13来源未在本次定位，不能重新杜撰算法或分母。

### 8.3 第8轮独立验收：总均值超过15仍不等于PASS

目录：`artifacts/extended_round8_independent_mission_acceptance_20261015_attempt34/`。

- 保留的测试：seed101101/202202/303303，每个覆盖30/60 × static/dynamic，共12对、24次H300、72000物理采样。
- joint平均时间：规则29.4417s、round8 20.9083s；时间降幅28.9839%。
- inverse-time效率增幅40.8131%，与“时间降幅28.98%”不是同一个百分比。
- 30static +36.41%，30dynamic +41.77%；60static -11.66%，60dynamic -3.04%。
- 两个seed总体持平，主要收益来自303303；seed-cluster描述性95% bootstrap区间约[0%,46.26%]。
- **文件verdict=FAIL**。预先规定除pooled≥15外，四个domain子组都不能回归，60两组没有满足。
- 范围仍是同一connected80m场景的新seed，不证明500m部署或跨地图泛化。

因此不能把早期“第8轮看起来好”或某个正均值改写成项目全面完成。

## 9. 2026-11-10三小时：Relay顺序修复和两批新训练

目录：`artifacts/gppo_three_hour_20261110/`。

### 9.1 有实际因果证据的生产修复

`integrations/gppo/scheduler.py` 的 ObservedMeshRelayDemandBuilder 分支，先消费已到达事件/增援机会，再刷新anchor和probe：

```python
self._augment_arrived_relay(task, stale_positions)
self.relay_builder.refresh_assignments(self.relay_assignments, stale_positions)
self._refresh_observed_relay_probe(stale_positions)
```

旧顺序先改目标，可能让已经到达的helper重新被当作in-flight，错过增援。已知seed1480021/30static，共执行器修正前两臂均120.4s，修正后均44.4s，H300安全。

**这是共享执行器收益，不是GPPO学习收益。** 没有关闭Shield、缩小radius或引入新planner。新回归先RED后GREEN；当轮91项相关测试通过，这是当轮历史记录。

### 9.2 两批训练

从round17 actor/encoder warm start；新完整joint目标下critic输出与Adam先重置一次。每批8条、温度1、lr2e-5、最多4epochs；两批总16轨迹、80决策、8次Adam更新。

- batch00采样约46.24分钟；batch01约24.29分钟。包含物理仿真、并发争用，不是纯GPU训练时间。
- batch01保留batch00的Adam状态，实际step从4到8，确为续接。
- 固定两个已知开发案例：parent合计45.6s→batch00 37.5s（+17.76%）；batch01仍37.5s、轨迹相同，按平局保留更早模型规则选batch00。
- batch00 SHA：`92415b8ff6f1a645045612b53f2716c130ccb6c0905c2874d8aa7ac4f6e1642d`。
- 开发集好转不是独立泛化证明。

### 9.3 8新seed独立测试只有15/16完成

seed1680021..1680028，每domain两例，rule/candidate各H300。最后1680026/30dynamic的rule触及预设截止，没有完整结果。

| seed | 规模/场景 | rule秒 | candidate秒 |
|---|---|---:|---:|
| 1680021 | 30static |43.6|65.7|
| 1680022 | 30dynamic |56.6|136.6|
| 1680023 | 60static |8.9|8.9|
| 1680024 | 60dynamic |29.0|13.2|
| 1680025 | 30static |58.7|93.5|
| 1680026 | 30dynamic |未完成|74.4|
| 1680027 | 60static |7.1|7.1|
| 1680028 | 60dynamic |14.0|14.0|

完成的15次均安全。60四对合计59→43.2s，+26.78%；30只有3完整配对，158.9→295.8s，慢86.15%。这些子集不能替代完整8对验收。

`final_result.json` 为 `INCONCLUSIVE`，goal15=null。超时不是已测物理失败，也不能当作安全通过或给截断数据算H300成绩。

## 10. 重新审视训练：已排除什么、还存在哪些问题

证据目录：`artifacts/gppo_training_reaudit_20261110/`。

### 10.1 没有发现的直接实现错误

实际saved transitions重放支持：合法mask、task/UAV flatten映射、old logp/temperature、reward/PPO符号、optimizer实际更新没有已证实错误。采样确用随机Categorical；推理是greedy。`F.rrelu`确实存在，但采样和更新都在eval模式，仍允许梯度，不是“没训练”。

“优化器执行一致”只证明实现契约，不证明标签能识别好动作、模型能泛化或指标对齐。

### 10.2 已量化的训练信号问题

- 最近两批16独立trajectory只有80个调度样本，Search61、Relay19，每个seed只一条结果。重复4个epoch不会创造新的反事实结果。
- batch00价值预测全0，batch01约[-.00767,-.00555]，相对实际return仍接近常数。
- Search advantage方差约96.94%/99.20%来自episode间差异，advantage与C相关约-.983/-.996。
- 每个回合内Search样本往往同号，存在场景难度/决策时刻/动作质量混杂。
- 暖启动策略集中，80次采样仅17次不同于argmax；4epoch可能只是反复利用少量动作信息。
- 在保存的42个独立测试candidate状态上，batch00相对parent17仅改变1次贪心选择，batch01又改变0次。不能说“新训练全面把模型训坏”。也不能仅凭shadow给parent完整闭环分数。
- 团队共享return及按role中心化不是自动的数学错误；高相关性也不能单独证明梯度有偏或符号错。

### 10.3 critic可学性probe

冻结actor与encoder，batch00八episode训练，batch01八episode验证；只离线拟合现有critic500步，不推广模型。

- train MSE由zero .007056降到fit .000209。
- validation常数预测MSE .005862，500步fit .009109，差约55.4%；原captured critic .014083。
- 早期step10曾略优于常数（.005130），之后过拟合。因此结论是欠校准与小样本过拟合，不是critic绝对不可学。

## 11. 一小时小场景实验为什么没完成

目录：`artifacts/gppo_learnability_hour_20261110/`。

计划两个固定训练seed各4次独立策略采样，环境seed相同、policy使用独立CUDA Generator。30static seed1880021四条完整C：`[24.1,32.5,24.1,24.1]`，都安全。60dynamic四条未完整，不能用于更新。

Windows事件审查发现合盖/电源键Modern Standby；最长一次约20分钟。父墙时与末日志仿真计时差异不能全归因于慢仿真，且旧awake已在更早的12:38UTC结束，未覆盖这一小时。时钟有过对齐疑点，不能把每个差额说成精确睡眠时间。

该轮policy updates=0，没有新模型。临时helper只防闲置睡眠，不能保证阻止主动合盖；之后各轮启用有截止的helper并验证释放，未改永久电源设置。

## 12. 2026-11-11两小时：真正完成组内可学性与更新回退

目录：`artifacts/gppo_learnability_two_hours_20261111/`。

### 12.1 完整数据与组内基线

复用前述4条完整30static，不重跑；新采30dynamic seed1280022四条和60dynamic seed1880024四条，共12trajectory、63decisions，所有数据同一行为模型、完整H300，实际logp/value/指纹与指标重放通过。

| training seed | C秒 |
|---|---|
|1880021/30static|24.1,32.5,24.1,24.1|
|1280022/30dynamic|43.7,38.0,29.9,38.0|
|1880024/60dynamic|25.0,11.0,25.0,25.0|

基线为同一seed中其他3条的平均C；优势 `(mean(other C)-own C)/15`。对应原后缀目标 `G(t)=-(C-t)/300` 与时间基线相减后t抵消，再除固定.05。不是在不同seed间凭绝对快慢给好坏标签。

零方差组保留为A=0；完整失败C=300也不排除。复用既有PPO clip/更新函数，不新造优化器；critic loss=0、lr2e-5、最多4epoch，保留原transition-mean权重。

### 12.2 真实触发的KL超限及修复

原一次更新跑2步：第一步sample KL .0033046，第二步 .0107717；最终mean exact KL .0090423，但max-state exact KL .1222405，超过预设.05；sample也超过.01。原函数只更新后break，仍保存最后越界weights。整个候选被gate拒绝，**从未进入物理评估**。

注意：这证明原停止机制不足以满足本实验硬上限，不等于证明所有PPO实现都数学错误，也不解释所有历史退步。

随后使用**同一批63条数据作可复现受保护重放**，不采新数据、不放宽cap、不按物理结果挑模型：

- 每epoch前snapshot model+Adam。
- 每epoch后相对原行为模型算sample/mean exact/max-state KL。
- 仍用原caps sample≤.01、mean≤.01、max-state≤.05；target .005保留。
- 第二步完全重现原被拒绝模型指纹，随后双回滚。
- 最终保留第一步：mean exact .002984388、max-state .0406004、sample .0033046。
- 实读保存checkpoint确认Adam step=1；拒绝旧checkpoint为step=2，模型指纹与快照一致。
- 原拒绝proof/模型、原合同均保留，没有覆盖。

合格模型路径：
`artifacts/gppo_learnability_two_hours_20261111/guarded/update/checkpoint_batch_000.pth`

SHA256：`440d13d0ce33e1a68a87b38f57781280b58855bc75524f132c3f2ad6326fd7ae`。

实现与测试：`guarded/epoch_transaction.py`、`guarded/guarded_update.py`、`guarded/test_epoch_transaction.py`、`guarded/independent_review.json`、`guarded/actual_saved_state_verification.json`。这是诊断训练wrapper修复，未自动改正式生产训练入口或默认推理模型。

训练计数要诚实：同一批数据原拒绝执行2步，重放又尝试2步，但最终候选净保留1步；不是两批独立数据训练，也不能说总共只计算了1次梯度。

### 12.3 三个训练场景与三个新seed

| training case | rule | parent17行为模型 | 新合格模型 |
|---|---:|---:|---:|
|1880021/30static|32.5|24.1|24.1|
|1280022/30dynamic|30.7|38.0|29.9|
|1880024/60dynamic|25.0|11.0|11.0|
|合计|88.2|73.1|65.0|

新模型比parent快11.08%，全部新增8.1秒来自已知30dynamic场景，单组快21.3%；比rule快26.30%包含历史parent已有优势。29.9秒上一轮selected已经达到，不能当首次性能突破。

| 新seed | rule | parent | 新模型 |
|---|---:|---:|---:|
|1981101/30static|44.5|47.9|47.9|
|1981102/30dynamic|27.0|28.0|28.0|
|1981103/60static|10.1|9.1|9.1|
|合计|81.6|85.0|85.0|

18次对照全H300、安全、同初始/隐藏布局/config。新seed上没有相对parent增量，且比rule慢4.17%。缺60dynamic的新seed，不是原八seed正式验收。

### 12.4 学习方向不是“每个首动作都必须涨概率”

离线CPU/autograd审查：30dynamic较快整轨迹相对较慢轨迹logodds提高约.988；60dynamic约+.030；30static约-.015，有局部代价。60dynamic首动作概率下降不能单独推出整条轨迹没学到。

存在局部梯度冲突（30S/30D cosine约-.241），但30D/60D为正约+.287；不能将60首动作下降全部归因于跨组相消。以上是保存轨迹的目标/梯度证据，不替代实际执行分数。

对上一轮selected模型924...的shadow：三个训练case共16保存状态当前与旧selected选择全同；新三case18状态仅1处不同。不能据此给旧selected闭环成绩，也没证明当前模型胜过上一selected。

## 13. donor拓扑探测与跨场景验证：最近的完整结论

### 13.1 为什么提出这个探测

fresh seed1981101：rule Search/非Relay完成20.8s，通信≥.9始于41.5s，joint44.5s；GPPO Search更早16.8s，但通信到44.9s才恢复，joint47.9s。当前与parent的7次动作、模型输入和3000帧joint采样一致，没发生训练迁移。

公开延迟位置+已知radio20m已足够计算当下通信分量与移除UAV后的分裂，不需访问真实障碍图。该case中GPPO选UAV1会分裂原小分量，rule选UAV2不会。但静态当前图不能直接预测未来运动、anchor承诺和到达时间；不能由此断言唯一根因或模型不可学。

### 13.2 固定的实验约束

函数：`artifacts/gppo_learnability_two_hours_20261111/donor_probe/donor_filter.py`。

- 只作用于纯Search图。
- 只使用graph中公开延迟position和既有`mesh_components`，range20m。
- 若移除某可选UAV会将其原donor分量剩余节点分裂，则屏蔽该UAV对应Search动作。
- copy graph和mask，不修改生产原图。
- 有公开位置缺失就跳过，不猜测；过滤后无合法动作则回退原mask。
- Relay不改；没有endpoint-LCC-gain条件、新planner、真实障碍地图、瞬移、碰撞半径改动。

### 13.3 单个已观察case的开发性探测

同权重440d、同物理/初始/隐藏布局，1981101从47.9→44.5s，约7.10%，安全，追平rule。

首个实际变化同时改变了任务与UAV：UAV1→task100003变为UAV2→task100001。故只证整策略变体的局部效果，不能说单独识别了某一骨干因素的全部因果作用。该case已经观察过，不算独立验收。

之后拟补同variant的30dynamic开发探测，但在06:37启动门槛时取消，仅有`donor_probe_dynamic/cancelled.json`，**没有运行和结果**。不要把它当失败物理实验或已验证泛化。

### 13.4 真正跨场景：四新seed、三臂、12次H300

用户明确“继续呀，你做，验证”后，于2026-11-11启动冻结新suite。目录：`artifacts/gppo_donor_validation_20261111/`。

四新seed在旧940份plan/manifest的种子元数据中未出现（此范围不证明所有未记录历史都没用过）。同一80m地图，30/60×static/dynamic各1，rule/unfiltered/donor三臂。所有模型alias使用同一440d；只有donor调用过滤，不训练、不调阈值、不选择模型。

| seed | 场景 | rule秒 | 原GPPO秒 | 加约束GPPO秒 |
|---|---|---:|---:|---:|
|1981201|30static|10.3|11.4|11.4|
|1981202|30dynamic|60.1|37.9|37.9|
|1981203|60static|5.1|5.1|5.1|
|1981204|60dynamic|28.0|28.0|28.0|
|合计||103.5|82.4|82.4|

- 原始12×3000物理采样重算，全部成功安全；同case三臂initial、hidden target layout、去掉checkpoint路径/hash两字段后的config一致。
- 505项source/asset guard、模型/proof/plan hash、旧filter逐字复制均核验。
- donor调用26次，22次applied；3个case共5次实际mask改变、11个新增禁用项；6次fallback、0缺公开位置。
- 并非代码没执行。但四case的**保存动作整序列逐项一致、解析后的joint metric trace相同**，不应误说actor_inputs文件完全相同：mask和概率可能不同。
- 原GPPO vsrule总时间短20.3865%；**donor vs原GPPO增量0%**。20.39%是既有模型收益，不能归因此约束。
- 30两例合计rule70.4、GPPO49.3，约29.97%；60两例均33.1，持平。
- 结论：单点开发改善未在四组新场景复现。当前不建议推广这个hard mask，不据此围绕它扩大训练。

这是最近完成的工作，不是待运行计划。进程、临时防睡眠helper已停止，没有生产修改或commit/push。

## 14. 指标与比较纪律：后续一定要守住

最近固定完整任务指标由 `scripts/gppo_joint_completion_metric.py` 计算：

1. 所有nonRelay任务完成；实际radio/connectivity≥.9；当前Safety已全部处理/释放。
2. 上述状态持续3秒，取首次满足的时间C。
3. 同时必须跑满H300（300s、dt=.1、3000物理tick），全窗口无碰撞/禁飞违规，最终Safety处理完成。
4. 失败/不完整不得删去；完整但不成功/不安全按预定义cap300处理，物理窗口截断则不能伪装完整成绩。
5. 主时间降幅：`100*(1-sum(candidate C)/sum(rule C))`。
6. `sum(rule)/sum(candidate)-1` 是inverse-time效率增幅，不能与上述百分比混用。
7. Search mean Find、all-found时间、coverage、wall time、joint C分别报告。Search更快不一定通信恢复更快。
8. 开发集、挑模型的validation、独立seed、已看过的debug case严格分开；已用于设计/选择的seed不能再称全新holdout。
9. 同地图新seed不等于新地图，80m结果不能外推500m。早期500m UAV0初始距heat约373.87m、速度1m/s，300s连直线到点都不够，需考虑任务/时域可行性，不是简单把80m结果放大。
10. Oracle真值地图可作离线物理诊断；生产决策只能读实际可观测信息。公开counterfactual endpoint几何不是UAV瞬移或真实路径可执行性的证明。
11. 执行器修复的共享收益、GPPO历史权重收益、最近一次训练增量、新增mask增量分开归因。
12. 老CSV、失败日志、被拒checkpoint/中间contract与source snapshot保留，别改写以使结果好看。

**已见测试seed**至少包括101101/202202/303303、9164/9168/9170、1680021..1680028、1981101..1981103、1981201..1981204等。完整训练/开发范围以各plan/manifest为准，不能只拿这份不穷尽清单作freshness审计。

## 15. 重要模型和SHA索引

| 用途 | 路径 | SHA256 |
|---|---|---|
| Frozen native GPPO | `<local-home>/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth` | `57abf685f7a8226dcf71e1441333bf74da6e228e423f1bb0034e110975e3fcd9` |
| round8 /6D | `artifacts/extended_gppo_joint_batch_20261010_attempt8/checkpoint_batch_001.pth` | `869e2ac49e54d24143c497197c27c39800b882fa1e4100b2f021815f73f30298` |
| round17 /6D | `artifacts/extended_gppo_joint_batch_20261012_attempt17/checkpoint_batch_015.pth` | `1cda34ecb47ec1eb411e3bf47f30351bbec1a7d48b96ff6297f70ac0739cf22a` |
| round19 /9D | `artifacts/extended_gppo_joint_batch_20261013_attempt19/checkpoint_batch_031.pth` | `7942984adcf3f55e77bac53b02291d0472fbc22f9b259b322395d4bbe9f20fb0` |
| Nov10已选batch00 /6D | `artifacts/gppo_three_hour_20261110/training/batch_00/update/checkpoint_batch_000.pth` | `92415b8ff6f1a645045612b53f2716c130ccb6c0905c2874d8aa7ac4f6e1642d` |
| Nov11合格guarded /6D | `artifacts/gppo_learnability_two_hours_20261111/guarded/update/checkpoint_batch_000.pth` | `440d13d0ce33e1a68a87b38f57781280b58855bc75524f132c3f2ad6326fd7ae` |

低层MARVEL模型另为 `load_model/MARVEL/checkpoint.pth`，不要与上面的frozen高层GPPO混淆。当前实验的parent文件为 `artifacts/gppo_three_hour_20261110/training/batch_00/behavior.pth`，SHA `e55eba9ca97a1dbfd9954a6360138a3768e3f997f74f92ffc441932c59db430a`；其actor/encoder继承round17，critic输出已重置，不能把这个文件直接称round17原checkpoint。

模型存在不等于已设为生产默认。本轮没有把440d或donor变体推广到默认执行链。

## 16. 接手时优先读的材料

### 原始需求与早期交接

- Git root `question.txt`、`task_plan.md`：原始需求，包含后来已被用户缩减的交付范围。
- `handoff/codex/MARVEL_CODEX_HANDOFF.md`、`MARVEL_CODEX_FIRST_TASK.md`：Claude→Codex初始交接；历史HEAD/只读禁训限于当时，不能覆盖后续明确授权。
- `handoff/codex/PROJECT_ACCEPTANCE_LEDGER.md`：长篇历史技术台账，保留大量中间阶段；最新文件优先于其中“正在训练”等过期条目。

### 权威实验入口

| 目录 | 先读 | 作用 |
|---|---|---|
| `artifacts/offline_search_recovery_20261009/` | `OFFLINE_SEARCH_RECOVERY_REPORT.md`, `final/results.json` | 两个停车位置的物理恢复与观测边界 |
| `artifacts/extended_checkpoint_common_comparison_20261015_attempt33/` | `final_diagnosis.json` | round8/17/19同条件公平比较及指标区别 |
| `artifacts/extended_round8_independent_mission_acceptance_20261015_attempt34/` | `acceptance_complete.json`, `verification_complete.json`, `plan.json` | round8独立12对，point28.98但FAIL |
| `artifacts/gppo_alignment_hour_20261110/` | `final_result.json`, reward/update审计 | 更早目标对齐/小试验资产 |
| `artifacts/gppo_later_rounds_hour_20261110/` | `final_result.json`, `later_rounds_inventory.json`, `training_data_audit.json`, retarget材料 | 后续轮次复用与Relay因果线索 |
| `artifacts/gppo_three_hour_20261110/` | `final_result.json`, `heldout/comparison.json`, `budget_closure.json` | 共享顺序修复、16条新训练、15/16不完整验收 |
| `artifacts/gppo_training_reaudit_20261110/` | `result.json`, `update_review.json`, `paper_task_review.json`, `information_audit.json` | 训练/论文/可观测性重审 |
| `artifacts/gppo_learnability_hour_20261110/` | `final_result.json`, `critic_probe/results.json` | 4条可复用轨迹、未完60组、critic过拟合 |
| `artifacts/gppo_learnability_two_hours_20261111/` | `final_result.json`, `verified_results.json`, `guarded/update_proof.json`, `final_audit.json` | 当前440d、PPO事务回退、local学习与fresh3结果 |
| 同上 | `gradient_conflict_audit.json`, `existing_radio_asset_audit.json`, `radio_feature_probe.json` | 学习方向和已有radio资产，不是新物理验收 |
| 同上 `donor_probe/` | `result.json`, `root_verification.json`, `donor_filter.py` | 一个已见case开发性探测 |
| `artifacts/gppo_donor_validation_20261111/` | `final_result.json`, `verified_results.json`, `filter_activity.json`, `final_independent_audit.json` | 最近四新case正式完成的跨场景约束验证 |

各目录中的 `continuation_status.json`、`plan.json`、runner identity等有些是启动时或阶段快照；以complete/verified/final_result和原始数据为准。不要看旧process PID就假定它今天仍活着。

## 17. 后续接手的最小行动清单

1. 先确认Git root/project cwd/branch/HEAD/status，保护已有文件；不要重新clone或自动切分支。
2. 先读最近`gppo_donor_validation_20261111/final_result.json`及上一轮`gppo_learnability_two_hours_20261111/final_result.json`，知道哪些已经结束、哪些结论被否定。
3. 重新检查CUDA与当前进程；历史helper/PID/已完成goal不是今天的运行状态。
4. 不重跑已完成baseline，不把被拒模型2ca...或旧protocol当合格候选；需要新实验则新目录、新manifest/stop条件。
5. 不把单点29.9s、一个20.39%或训练集26.3%称为整体15%验收。
6. 当前donor hard mask没有新增跨场景收益，暂不推广；下一次新工作需基于这个否定结论，不要又无新证据重试同一方案。
7. 后续是否再训练、修改决策接口或补齐统一验收，应承接用户当次目标与预算。旧预算已结束，本总结任务不启动新的实验。

可直接用于下一对话的简短指令：

> 请先阅读 `handoff/codex/MARVEL_GPPO_CONVERSATION_CONTEXT_20261121.md`，再核对本地现场。团队仓库是 https://github.com/Azucat111/2026PD_MARVEL，上游是 https://github.com/marmotlab/MARVEL；项目位于 WSL Ubuntu-22.04 `<local-home>/MARVEL_INTEGRATION/MARVEL-main (1)/MARVEL-main`，Git root为其上两级 `<local-home>/MARVEL_INTEGRATION`。分支extended-scalability，核验时HEAD5143fbdc...。保留全部脏工作区和artifacts，不commit/push，不破坏frozen native。当前440d模型局部训练有效但新seed泛化不稳定，donor公开拓扑约束4新场景增量0%，尚未通过项目整体15%验收。先复用现有数据，明确下一项有限实验与停止条件，不要从头研究或默认开启长训练。

## 18. 本次整理的证据等级与未知项

- 当前分支/HEAD/remotes/CUDA与478文件hash一致性为2026-11-21现场核验。
- 实验数值来自对应已保存JSON/报告；本次没有重跑旧实验来“重新证明”。
- 部分早期聊天只保留用户追问，无法逐字恢复原回复；精确14.13%、某个“-18%”及第8轮全lineage精确累计小时，本文件未凭空补齐。
- 早期可视化是否已正确放到桌面，本次未核验；不要承诺文件仍在。
- 历史目录可能还有未逐个展开的训练、修复与失败记录，本文件给出入口和权威优先级，不将其等价为未发生。
- 仓库remote已核对，但未核对GitHub远端最新HEAD，也未证明当前所有本地修改已上传。
- 最近各bounded goal的complete仅表示该轮诊断/验证目标完成，不表示项目整体15%技术目标已完成。
