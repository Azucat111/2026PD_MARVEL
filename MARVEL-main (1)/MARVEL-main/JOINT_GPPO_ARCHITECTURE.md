# Event-Level Joint Search–Relay GPPO：可实施设计

结论为 **GO_CONDITIONAL**：推进统一候选和原子承诺接口的下一阶段；本轮交付的是可运行只读原型，未部署联合执行器、未训练、未证明收益。真实短场景在同一公开边界存在跨角色候选，详见 `JOINT_CANDIDATE_SHADOW_AUDIT.md`。条件是实现并检验下面的有限执行合同，不能仅将两次 allocator 图拼起来送入旧模型。

## 1. 从证据到改造范围

- 当前 Search extended catalog 已覆盖全部 pending 热点（`scheduler.py:801`）；保留它，不另造任务生成器。
- 同一边界存在可用于两种角色的同一 UAV，且能调用原运动接口得到安全首步；历史实际 actor 图仍是分离的纯角色图。这支持把选择权提前到**新承诺之前**，不支持主动释放旧任务或提前生成未触发增援。
- 大量边界没有新的合法候选，往往已有角色容量/持续 Relay 事件占用资源。这是生命周期事实，不能为了训练密度伪造新任务。无候选事件仍计入状态审计和终局指标。
- 端点 radio gain、旅行时间下界和首步证书是三种不同证据；均不能预测整条路径或完整任务成功。物理执行与后续观测仍须闭环验证。
- 只在同一 80m 场景、既有开发 seed 上短采样；尚不能确定机会的长期频率、收益大小或新地图泛化。架构先保留最少自由度，训练预算需满足实验合同。

保留冻结 MARVEL PolicyNet、GPPO AHGNN / pair actor / graph critic、TaskGraph ID 映射与 mask convention、延迟观测、Search Find 完成合同、Relay anchor/到达增援/稳定释放、动态 Safety、真实运动和 Shield。原 Rule 与生产 scheduler 保留为 R0/P0；R1/P1 使用同一个新候选接口。

## 2. 事件状态：部分可观测 SMDP

物理时刻 `t_n` 的事件在强制 Safety、公开完成通知、不可达/yield 释放和资源 reconcile 后产生。公开事件包括控制边界、任务完成/激活、Safety 抢占/释放、Relay 到达/稳定释放、公开拓扑改变。首版仍在现有控制边界消费事件，不新增物理时钟更新频率。

环境真实状态为 `x_n`，策略只见 `o_n = h(history_public≤t_n)`；这是观测型半马尔可夫控制，不能声称单帧图构成完整 Markov 状态。可先沿用公开历史摘要；若依赖持续记忆，再以消融确定是否加 recurrent summary。

```python
@dataclass(frozen=True)
class JointObservation:
    event_id: int
    physical_step: int
    public_version: str              # projection digest + ownership version
    current_time_s: float
    position_age_bound_s: float
    uavs: tuple[PublicUAV, ...]      # delayed peer positions, capabilities, commitments
    tasks: tuple[PublicSlot, ...]    # ONLY triggered, non-complete lifecycle entries
    radio: PublicRadioGraph         # delayed topology, age and confidence
    commitments: tuple[Commitment, ...]
    role_capacity_remaining: dict[Role, int]
    release_and_arrival: PublicGateState
    remaining_horizon_s: float

@dataclass(frozen=True)
class CandidateEdge:
    action_id: tuple[int, int]       # stable (slot_id, uav_id), never row index alone
    role: Role
    anchor_version: str
    raw_eta_s: float                 # known route + labelled unknown lower bound
    legal: bool
    mask_reasons: tuple[str, ...]
    prerequisites: tuple[str, ...]
    prefix: PrefixCertificate | None # own sensor; NOT fed as peer truth

class JointCandidateProvider:
    def observe_and_collect(self, public: JointObservation) -> JointCatalog: ...
    def residual(self, catalog, selected: tuple[CandidateEdge, ...]) -> JointCatalog: ...

class JointExecutor:
    def validate_and_commit(self, version, selected, stop_kind) -> CommitReceipt: ...
    def advance_existing_physics(self, receipt, marvel_actions) -> PublicEvents: ...
```

本轮实现对应 `capture_public_state → enumerate_joint → attach_actual`，不实现 `validate_and_commit`。公共 JSON 中没有隐藏目标、真值 occupancy、训练数据或模型对象。own-sensor 的当前姿态只用于运动证书；actor 的 fleet 空间特征继续采用规定的延迟快照，不借枚举器升级观测权限。

## 3. 候选集合、联合动作与冲突

在一个真实边界构造 `C_n = C_Search(o_n) ∪ C_Relay(o_n)`。每类只接受原触发门已打开的 slot：Search 已激活/active、未服务、未分配；Relay 通信不足且初始或合法到达增援；释放当拍不重触发。新接口不学习 anchor，不取消补充预算/到达条件，不把 probe 当成 Relay slot。

逐边掩码以 `True=invalid`：

\[
M_n(t,u\mid A)=M_{lifecycle}\lor M_{capability}\lor M_{availability}
\lor M_{Safety}\lor M_{capacity}(A)\lor M_{reservation}(A)
\lor M_{deadline}\lor M_{observed\_cut}\lor M_{support}\lor M_{relay}(A).
\]

所有门复用当前实现；`support` 是已标记的支持优先规则，不是不可达定理；首版不新增此前无泛化收益的 Search donor hard mask。Relay guard 必须**仅作用于 Relay 行**，不能把 `mesh_relay.candidate_mask` 原样用于混合图，否则其 radio-gain 约束会错误限制 Search。

联合动作采用自回归有界 matching：

\[
A_n=(a_{n,1},\dots,a_{n,K_n},STOP),\quad
\pi_\theta(A_n\mid o_n)=\prod_{k=1}^{K_n+1}\pi_\theta(a_{n,k}\mid o_n,A_{n,<k},M_{n,k}).
\]

每选一条边，在**临时承诺表**标记 UAV、slot、角色容量，重建 residual mask；不得重用同 UAV/slot。Relay 新 helper 的计划端点用于原 radio forecast；Search 保留当前公开位置，不虚构其未来 waypoint 已到达。旧 Search/Relay/Safety/probe 预占持续保留。

先允许每个边界最多原角色预算之和的选择，随后一次提交。`STOP` 若还有可行边，等价于明确 Defer 剩余新任务到下一次现有控制更新；若无边，只能 Continue 旧承诺/探索。两者不重复为两个相同动作。只有存在真实可执行候选且可以安全维持现有系统一个控制周期时才开放 Defer；Safety 请求不可 defer，临近已认证硬 deadline 的强制任务不得 defer。若安全延迟无法证明，只开放填满可行预算的 STOP。首个执行版本可以完全禁用有候选时的 Defer，先隔离跨角色 matching 收益。

同一 UAV 可 Search/Relay 的证据不意味着所有 matching 都合法；同拍第二个 Relay 的 forecast、slot/容量占用必须在 residual 状态再检查。到达、增援、释放计数在真实事件中只消费一次，不随自回归选择次数重复更新。

## 4. 完整状态转移合同

1. 在 `scheduler._high_level_update:1200` 后冻结公开快照及版本。对真实任务只消费一次本拍强制完成/释放，得到 `B_n`。运行既有 Relay release/arrival 判定必须与 Search collect 独立，不能在候选选择循环中推进计数。
2. `collect(B_n)` 只生成描述和 mask，不改变 UAV、任务、RNG。固定 demand/anchor 版本；若既有 Relay 事件需要合法 retarget，按原“先到达增援机会、后 retarget”顺序生成计划，记录其版本。
3. actor 或 Rule 在完全相同 `C_n` 上选择，生成临时 matching `A_n`。Continue/Defer 若开放，两个选择器拥有相同动作，R1 预注册固定填充规则。
4. `validate_and_commit` 验证版本、有限数值、所有硬门、UAV/slot 唯一性、角色容量、Safety 所有权、依赖及 Relay joint forecast。失败则**全部不写入**，重新观测或保守 Continue；不得残留半个 Search batch。
5. 成功后原子写 `search_assignments`、`relay_assignments`、`robot.available/busy_until/assigned_task_num/current_task`、`active_search_uav_ids`，使用既有 `_apply_uav_state` 的计量语义。busy_until 是估计，不授权越过任务完成条件自动释放。
6. 继续同一 `apply → _search_action/_relay_action → adapter certification → before_physics_tick → Shield → dynamics → physical collision → sensing` 链。不瞬移、不换半径、不取消 warning、不以 endpoint forecast 代替轨迹。
7. 新观测产生 Find、到达、Safety、拓扑与释放事件；下一 `t_{n+1}` 更新。被安全拒绝的 motion 仍是实际 hold，并进入既有 rejection-aware recovery。

安全证明/验收要求分层：资源归纳不变式（每 UAV/slot 唯一、容量、强制 Safety）；信息流不变式（公开投影、无 oracle）；首步证书（已观测五探针、公开 warning、同伴运动包络）；真实 H300 全窗安全。最后一项只能由物理闭环验证，不能由 mask 或 shadow 代替。首步执行可能被保守 veto，必须记录实际动作/拒绝率，不能把“可分配”算“能完成”。

有限改造点集中在 coordinator/provider/commit receipt。原生产类不直接替换；独立开关选择 `original` 或 `joint_v1`，默认 original。Native 路径保持原样。

## 5. 网络最小改造与可观测特征

现有 13D UAV、12D task 与6D heading edge 已包含任务类型及相对运动信息，pair actor 可自然输出混合图每条边 logits。先复用 encoder/actor 参数，保留 ID 与 mask 映射；**不假定新混合图归一化与旧图同分布**。`TaskGraph._build_model_features:326` 的图内缩放会随角色混合改变，需输入重放与纯角色 parity 测试。

显式增加小型公共上下文 side encoder，而非重写 AHGNN：

- radio 边：从延迟位置与固定 comm_range 得出的 adjacency、边距、观测年龄；同时保留“这是预测”的标签。
- UAV：组件大小、LCC membership、移除后非 donor 分裂数、已有 Search/Relay/probe 承诺、公开余 ETA；这些是特征，首版不据此引入新 hard mask。
- slot：角色、触发原因、anchor 对应组件/endpoint、已到达条件、初始/增援 budget、公共依赖、服务/释放窗口状态。
- 全局：两角色剩余容量、remaining horizon、上次事件间隔、任务完成计数、公开连通不足；critic 和 actor 使用同等公开信息权限。

新增特征接入 zero-initialized residual projection，旧维度/权重可 strict-compatible warm start；STOP 采用独立小 head。critic 加图级容量/依赖/时间摘要，训练前检查其标度；不沿用旧 reward critic 的数值校准假设。若无需 STOP 的首版验证，则可不增加 actor head，仅修改候选接口和 residual graph。

## 6. 目标、事件时长、奖励与折扣

主目标仍是完整 H300 的实际完成时刻：

\[
\widetilde C = \begin{cases}C_{joint},&\text{H300完整、全窗安全且完整任务/通信/Safety稳定3s}\300,&\text{完整窗口任务失败或不安全}.\end{cases}
\quad J(\theta)=-\mathbb E[\widetilde C/300].
\]

物理截断不是 C=300 的完整失败样本；单独报告，阻止对应批次用于正式通过或普通 on-policy 更新。

首版有限时域采用 `gamma=1`，直接对齐未折扣 makespan。令 `c_prov` 为首个稳定完成时刻（若无则 H），物理 running cost 在稳定 latch 前每 tick `-dt/H`；H300 结束根据全窗安全加 `-(Ctilde-c_prov)/H` 修正，故回合奖励精确等于 `-Ctilde/H`。安全不能仅作可交易的小罚项，仍是独立通过门。

事件奖励聚合物理 tick：`R_n = sum_{t∈[t_n,t_{n+1})} r_t`，最后事件接收 terminal correction。零时长自回归选择不重复发时钟奖励，避免分配越多惩罚越多的假目标。

可选折扣研究须明示改变优化目标：

\[
\Gamma_n=\exp(-\beta\Delta t_n)=\gamma_\tau^{\Delta t_n/\tau},\quad
\delta_n=R_n+\Gamma_n(1-d_n)V(o_{n+1})-V(o_n),
\]
\[
\hat A_n=\delta_n+\Gamma_n\lambda_\tau^{\Delta t_n/\tau}(1-d_n)\hat A_{n+1}.
\]

`beta=0` 是主合同。Macro event 的 log probability 为整个 matching 的 conditional logp 之和；不能将同一 event reward 原封复制到 K 个零时长决定而不说明权重。首版以 macro-event 样本训练，critic 输出 event value；可辅以固定 seed 的组内重复 rollout baseline，明确每个独立 episode 与学习样本分母，复用已有冻结采样/PPO组件。

## 7. On-policy PPO 与事务回退

每批冻结 behavior checkpoint、观测/候选接口版本、normalization、temperature 和 mask。保存每个 event 的公开输入、每个 residual graph/mask/ID、选中 action、behavior conditional logits/logp、`Δt`、reward protocol、terminal/truncation、模型指纹。物理轨迹只采一次；重放 epoch 不是新样本。

\[
\rho_n(\theta)=\exp[\log\pi_\theta(A_n|o_n)-\log\pi_{old}(A_n|o_n)],
\]
\[
L_{clip}=\mathbb E_n\min(\rho_n\hat A_n,
\operatorname{clip}(\rho_n,1-\epsilon,1+\epsilon)\hat A_n),
\quad L=-L_{clip}+c_v(V-\hat G)^2-c_H\mathcal H.
\]

缓存的旧 residual mask 用于 PPO 概率重放，更新时不重新访问在线环境生成不同 mask。Rule teacher 不得默认加入 policy objective；若有 imitation warmup，单独列协议与公平对照。

复用 `scripts/gppo_rollout_batch.py:15` 的 FrozenBehaviorBatch/指纹思想、`gppo_parallel_rollouts.py:40` 的 update 复用路径和原 PPO loss；actor/critic 输入 schema 变化需版本化。当前生产训练循环 `train_extended_gppo_batched.py:217` 是 update 后测 KL 再 break，**不自动回退**；不得把历史 guarded wrapper 修复说成已上线。

下一版移入已核实的 `artifacts/gppo_learnability_two_hours_20261111/guarded/epoch_transaction.py` 思想：每 epoch 前 snapshot **model + Adam（moment、step、param_groups）**；更新后与批次初始 behavior 比较 sampled、mean exact conditional、max-state exact KL，任一非有限/越界，或异常/非有限梯度/优化器状态，恢复两者并清梯度。checkpoint 只保存最后合格事务。KL target early stop 在保留上次合格状态后执行。

初始建议沿用历史 cap：sample .01 / mean exact .01 / max-state .05、target .005、lr2e-5，**只作待冻结设计参数**；macro matching 的 KL 是 conditional 链之和，旧 per-pair cap 不可直接当同等强度。先 CPU重放确认 chain/macro度量，再在任何训练前冻结最终阈值。回退是停止条件，不用物理开发成绩来“挽救”越界 epoch；新批次必须重新采样，禁止将旧独立测试结果作为优势标签。

## 8. 下一阶段进入条件

先实现 `joint_v1` 原子接口与资源/Safety归纳测试，构造 R1 与 P1 同动作集的逐状态 parity，验证纯角色退化与空事件。再执行实验计划中的小闭环技术验证；若同拍联合选择实际无法落入原运动语义、强制 Safety 退化、或跨角色选择长期稀少到无法在有限预算采样，停止 PPO 扩大，修订接口。GO_CONDITIONAL 不构成训练或大规模验收已经授权/通过的声明。
