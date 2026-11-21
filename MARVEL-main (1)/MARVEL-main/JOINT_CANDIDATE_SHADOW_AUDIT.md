# 联合候选 Shadow 审计

**结论：GO_CONDITIONAL。** 真实公开状态提供跨角色、共享UAV、首步可执行且可产生不同运动/角色承诺的选择；现生产GPPO却分别接收Search和Relay图。值得实现有限的联合候选/原子承诺接口，但尚未实现联合执行器或验证完整任务收益。冻结MARVEL、原Rule与物理执行保持原样。

## 1. 接手、材料与证据边界

准确交接 `5692ca4fe5c56f398707deea87c1efc844297b71`；新分支 `codex/joint-gppo-shadow-20261121`。实际工作目录 `/workspace/2026PD_MARVEL/MARVEL-main (1)/MARVEL-main`，预置上游目录保留。交接428源码/配置哈希全匹配，8个模型SHA/strict CPU load/masked forward通过；未重跑历史训练或H300审计。

已读指定历史材料。`artifacts/gppo_three_hour_20261110/development/decision_space_audit.json` 的104次分配摘要没有完整同步观测、恢复FREE cells、全部图/heading历史、资源生命周期和当时MARVEL fallback，不能复原同拍联合候选，也不能证明所有真实状态都没有联合机会。本轮不补造这些输入。

主证据：`artifacts/joint_shadow_20261121`。冻结模型为云端manifest中的“November 11 guarded candidate”6D推理导出和原MARVEL；原型提交`a98b9a3`，CPU、无训练，四域各3控制步、同一connected80m地图，动态域为原公开expanding warning、物理速度.2m/s。全部为已见开发seed，非独立验收。

初步23边界位于`artifacts/joint_shadow_20261121_preliminary`，捕获原型`ad864dd`、重枚举`5c3d5a6`。这些输入没有真实MARVEL base action；own-pose hold dry-run只能证明条件性prefix，不能用其旧executable字段当最终立即执行证据。已保留原物理/公开记录并明确纠正。两套有状态重叠，**不合并为35个独立样本**。

## 2. 唯一同步边界与实现

`_high_level_update`在`integrations/gppo/scheduler.py:1200`完成reconcile，下一行的`_refresh_search`入口是捕获边界。真实系统随后照常执行Search→Relay。observer对真实assign仅转发一次，记录输入mask、ID与实际选择，最后记录分配、所有权、scheduler命令、adapter最终命令和正常物理返回的own观测。它不替换选择器。

`capture_public_state`只投影白名单：延迟位置、own感知位置/heading、观测FREE/occupied与反馈、MARVEL已观测图、公共任务元数据/热点、角色能力、承诺/计数、公开warning参数和**当时实际MARVEL base actions**。不复制runtime/scheduler/agent整个对象，不取隐藏目标、任务targets、真值occupancy或模型对象。

`enumerate_joint`在显式构造的隔离runtime上复用原 `_refresh_search/_refresh_relay`、SearchSlotBuilder、mesh demand/anchor、allocator、TaskGraph、deadline、closed-cut、support和Relay guard。只读adapter返回None：不选择任何候选，所以两类候选都从相同既有承诺出发。Relay release/arrival gate仅在隔离副本观察一次；已有probe预占保留，probe不伪装成学习动作。

这给出**同边界第一条动作**的候选集合，不穷举整个matching或重排后的长期轨迹。实际原图保持分离；潜在集合取并集是研究输入，不声称生产actor已经看到联合图。

### 四类分开记录

| 分类 | 判据 | 对合法/立即执行统计的贡献 |
|---|---|---|
| `existing_scheduler` | 同边界通过原掩码，且同配对出现在真实本拍某次原顺序调用的合法图中 | 必须另过首步证书才能算立即执行；出现在分离图不等于actor可以选角色 |
| `execution_supported_order_blocked` | 同边界通过原掩码，但实际顺序选择/新预占导致本拍真实图中没有该合法配对 | 不释放旧承诺；保留真实已选UAV排除了哪些Relay边的证据 |
| `requires_execution_contract` | 例如已有Search容量占满，想立即新增pending热点需改抢占/资源合同 | `admission_legal=False`；其他约束尚未评估，不能算已证实机会 |
| `constraint_rejected` | 现有能力/可用性/endpoint/deadline/cut/support/radio门拒绝，或任务/触发尚未成立 | 不进入合法集合；support优先是调度条件而非物理不可达证明 |

Dormant目录是逐slot/UAV的**未获准提案**，附`other_constraints_evaluated=False`。未激活Search、未触发Relay、任务不active均属于当前非法，不通过换标签增加合法候选。

### 首步执行证书

每个合法边在单独的恢复/视点副本中临时承诺，先按实际`apply`语义同步assignment token，再调用现有`_search_action/_relay_action`，使用捕获的同一MARVEL base input。恒速模型预测下一个.1s位置，要求：

1. 存在fresh own sensor pose和实际base action；Search必须有实际移动/服务意义，Relay允许已到anchor的维护hold。
2. 观测FREE五探针连续段通过；当前公开warning buffer和下一tick no-fly+.2m不相交。
3. 对所有同伴延迟位置，候选首步线段到其位置的距离大于`.5 + v_peer*(age+dt)`。这保守覆盖同伴下一tick任意合法恒速动作，无需预测隐藏运动。

缺少实际base动作时只给`conditional_prefix_certified`，`immediately_executable=False`。证书只覆盖首个物理tick；不证明UNKNOWN尾段、完成deadline、通信未来保持或H300全窗安全。

## 3. 数值结果与准确分母

每个控制边界都计入，包括没有allocator调用的边界。主表的分母为**全部12个严格捕获边界**，每域3个；“无/单/同角色多/跨角色”按通过全部现有分配掩码的边计数，四类互斥。“严格跨角色”再要求上述同实际执行输入的首步证书。

| 域 / seed | 边界 | 无合法候选 | 单候选 | 同角色多候选 | 两角色潜在合法 | 严格跨角色可执行 | 同UAV跨角色竞争 | 顺序/新预占排除事件 | 需资源合同的未验证目录事件 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30静态 /1280021 | 3 | 1/3 | 0/3 | 1/3 | 1/3 | 1/3 | 1/3 | 1/3 | 0/3 |
| 30动态 /1280022 | 3 | 2/3 | 0/3 | 0/3 | 1/3 | 1/3 | 1/3 | 1/3 | 1/3 |
| 60静态 /9164 | 3 | 1/3 | 0/3 | 1/3 | 1/3 | 1/3 | 1/3 | 0/3 | 0/3 |
| 60动态 /1981204 | 3 | 1/3 | 0/3 | 1/3 | 1/3 | 1/3 | 1/3 | 1/3 | 0/3 |
| 合计 | 12 | 5/12 | 0/12 | 3/12 | 4/12 | 4/12 | 4/12 | 3/12 | 1/12 |

这是早期事件密度，不是自然H300频率估计；四个case、同一地图、强相关初始状态，不将候选边数当独立样本。严格跨角色事件以不同角色生命周期承诺为基础，并在每域找到**同一UAV可产生不同首个物理位置**的更强见证。见机器可读`witnesses.json`及逐事件JSON，不能仅凭多个Task–UAV配对推出该结论。

真实原allocator有22次建图/选择调用，其中16张含合法动作，6张全masked；mixed-role为0/22（限定非空图也为0/16）。它们与12个同边界潜在集合分别统计。至少两个不同角色/任务承诺且首步有证书的事件为7/12；该更强口径不把同任务不同UAV重复计为不同任务语义。顺序排除数不包括旧承诺的强行抢占；需要执行合同的目录数也不声称其他门已经通过。

主候选记录共1200条逐slot/UAV审查：665条现有顺序调用允许，5条被新顺序预占排除，530条被现有约束拒绝；其中670条通过分配门并有严格首步证书。另有960条dormant提案：900条任务/触发条件不成立，60条需修改资源合同（同一个30动态容量满事件的2个pending槽×30机）。后60条未评估其他门，不能叫60个可执行机会。两个集合与事件分母不混算。

## 4. 可审计见证

- 30静态，t=1s：既有Search/Relay门同时成立；可让同一UAV接Search热点或Relay anchor，原Search新预占排除了一部分Relay helper边。
- 30动态，t=1s：同样存在共享资源竞争；公开动态warning继续参与anchor/路线过滤。
- 60静态，t=1s：存在真实双角色和不同首步选择；本次原轨迹未恰好排除可见Relay边，说明“可联合选择”不要求实际先发生损失。
- 60动态，t=2s：先前公开Find释放资源后出现共享候选；不将t=0尚未触发的Relay当合法任务。

| 域、t | UAV | Search slot / waypoint | Relay slot / waypoint | 合法Search/Relay边 | 共享UAV数 |
|---|---:|---|---|---:|---:|
| 30静态，1s | 0 | 100002 / (24,72) | 200015 / (20,72) | 19 / 27 | 15 |
| 30动态，1s | 1 | 100000 / (56,16) | 200003 / (60,24) | 59 / 27 | 27 |
| 60静态，1s | 1 | 100001 / (68,68) | 200051 / (68,64) | 130 / 58 | 58 |
| 60动态，2s | 6 | 100001 / (72,20) | 200002 / (72,24) | 75 / 56 | 55 |

上述waypoint不是瞬移后的位置；物理next_position只移动≤.1m，具体值见JSON。每个见证给出case、event index、public-state SHA、UAV、Search/Relay slot和position、两个首步waypoint/next_position、实际GPPO分配及被顺序排除的边。脚本执行证书而不执行这些反事实动作，不给它们编造完成时间。

## 5. Shadow不干预验证

- 新shadow合同测试13项；相关回归共91项通过、0失败、0跳过。
- 所有严格捕获边界两次枚举结果相同，真实公开状态、既有cache/counter和Python/NumPy/Torch RNG摘要相同。
- 独立测试额外比对physical robot/task/clock/release状态及已有模型tensor前后相同，并用异常对象阻止隐藏字段读取。
- 每case检查原MARVEL与高层模型tensor指纹不变。
- 30静态同seed、同配置、同原选择器开启/关闭shadow对照：初态、配置、真实配对、碰撞、进度和逐tick物理观测要求字节一致；事件仅排除顶层wall timestamp比较全部语义字段。原始字节及proof保留。

原型不调用真实step、不改真实任务、不触发新模型推理；观测实际模型选择的wrapper返回原对象，未改图或mask。旧checkpoint绝对路径仅在测试进程映射，生产/native源码零修改。

实现入口及本轮实际行号：`joint_shadow.py:101 capture_public_state`、`:181 _sandbox`、`:256 _prefix`、`:326 enumerate_joint`、`:443 attach_actual`、`:464 summarize`；`joint_shadow_recorder.py:69 _search`、`:99 _allocate`、`:120 _apply`。这些新文件没有被生产模块导入。

## 6. 结论与限制

**GO_CONDITIONAL**：最小下一步是把`_high_level_update:1200`之后的两类候选统一收集，逐选择更新角色容量/互斥和Relay forecast，再原子提交既有执行结构。需要明确新增matching的资源事务、版本与失败回退，不得只拼接图后继续原双批commit。

尚不选择`GO_JOINT_GPPO`的原因：联合提交器、Defer语义、整批mask演化及闭环全窗安全未实现/验证；本次机会集中于早期，不能证明长期频率足以训练或性能有15%提升。也不选择`NO_GO_CURRENT_DESIGN`：四域真实公开状态已构成跨角色、共享资源和不同首步动作的直接见证。

限定：未运行新的RL训练、大规模H300或最终验收；未改冻结权重/Rule/生产scheduler；首步证书不等于整条物理路线存在；本轮公开snapshot足够重放候选，不包含完整历史6D model heading batch或所有actor logits，不能声称已精确重放历史策略概率。原始实际动作是在线记录，不是后验伪造。

架构与公式见`JOINT_GPPO_ARCHITECTURE.md`；R0/P0/R1/P1和15%可证伪合同见`JOINT_GPPO_EXPERIMENT_PLAN.md`；全部实际命令见`JOINT_SHADOW_REPRODUCTION.md`。
