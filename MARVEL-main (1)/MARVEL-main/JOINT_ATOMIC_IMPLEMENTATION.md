# Phase 2 联合候选与原子提交

主要实现提交为 `4567b638caf4c4f0dc9e445ca9946b178628ff28`，Relay 路线缓存修订为 `ac3cd49b54d4957eac2c0792d6d2392d10f12fa9`。分支基底是已验证的 Shadow 交付 `5cf0cc6f9ac8ee13fb8f135ca6a5549aa5fc0ca5`，其祖先包含交接基准 `5692ca4fe5c56f398707deea87c1efc844297b71`。GitHub 启动检查只返回交接分支。工作区中的 Shadow 分支与经 `git bundle verify` 验证的第一阶段 bundle 对应同一提交。

## 接口和时机

| 位置 | 职责 |
| --- | --- |
| `utils/policy_adapter.py:123,294` | 配置选择调度器。联合事务一直覆盖到 adapter 返回最终 waypoint 和 heading |
| `integrations/gppo/scheduler.py:1180,1208` | 复用控制时钟、Safety、Search 完成和强制释放，调用分配钩子。Legacy 钩子仍依次刷新 Search 和 Relay |
| `integrations/gppo/joint_shadow.py:101,255,325` | 复用白名单公开投影、首步证书及隔离的原阶段刷新 |
| `integrations/gppo/joint_atomic.py:35,121` | 构造版本化快照、单张混合 TaskGraph、候选和拒绝理由 |
| `integrations/gppo/joint_atomic.py:98,267,280,297` | Rule 或 GPPO 选择、提案转换、提交前重验证、原子资源写入 |
| `integrations/gppo/joint_atomic.py:201,352,372` | 回滚范围、所有权不变量、最终指令与证书比较 |
| `integrations/gppo/mesh_relay.py:104` | 既有 Relay retarget 接受本拍新 helper 的固定目标，保留其已认证 anchor |

`task_scheduler.mode` 为 `joint_rule` 或 `joint_gppo` 时启用扩展调度器。现有 `gppo` 默认语义沿用原实现，正式 R0 仍通过同一个旧 Rule 配对选择器接入。联合 GPPO 要求经校验的 `extended-gppo-observed-heading-v2` checkpoint，默认做确定性选择。显式设置布尔值 `joint_deterministic: false` 可用同一 masked categorical 采样。原 MARVEL PolicyNet、原 GPPO 特征维数、物理和 Safety 实现仍由现有组件提供。

`JointSnapshot` 存为不可变白名单 JSON 字节，版本由接口标签、分配修订号及公开状态 SHA-256 组成，同时携带当前物理 step。捕获时已消费既有 Safety 抢占、Search 已发现目标与强制释放。输入包括延迟机群位置、own sensor/反馈、已知占据、公开热区和 task status、既有承诺、Relay 稳定计数、动态预警、MARVEL 局部图和当拍真实 MARVEL base actions。枚举器以隔离对象处理原 Search 和 Relay 门，有限的 Relay lifecycle 更新先留在草案中。它只读公开状态。物理最初布局的核验摘要在独立评测脚本生成。

## 候选与选择

两角色分别复用原资源和合法性筛选，再把行合入同一张 TaskGraph。这样 Search deadline 和支持门只作用于其原行，Relay radio guard 也只作用于 Relay 行。最终掩码还要求现有恢复接口产生可认证首步，包括公开可通行、动态预警、同伴保守距离和实际 MARVEL fallback。已使用的 Relay anchor 按现有 Shield 的 UAV 间距保留，同一 UAV、同一 slot 或过近的 Relay anchor 是显式冲突。通过门的记录包含角色、稳定 slot/task/UAV ID、处理和旅行估计、anchor、已认证 waypoint 和 heading。其余对记录拒绝原因。无合法候选时只执行既有维持和规则状态处理。

联合 Rule 与 GPPO 都在这张图上返回一个索引，`JointPlan.propose` 映射到有版本的稳定 ID、角色、anchor 和 value。Rule 复用原排序 `(-priority, raw_total_time, subtask_id, uav_id, task_index, uav_index)`。Search 的高优先级在两类同时合法时可以领先。GPPO 的旧 6D actor 可从同一 mask 比较两类边，保留 AHGNN、pair actor、critic 和 delayed heading 输入。混合图会影响原图内归一化。旧权重用于兼容验证，critic 对联合资源的估值仍待训练与校准。

## 提交边界与状态不变量

1. 在同步调度器中校验 event step、版本、ID、角色、anchor、合法掩码与有限值。重新捕获完整规划输入并校验版本，因而任务状态、资源所有权、Safety、锚点或观察变化会使提案失效。策略仅拿到独立 graph 副本，内部 plan 和写入函数不由策略提供。
2. 暂存 Relay 释放和计数结果，把选中 Search heat 或 Relay target/anchor 登记为真实承诺，更新已选 UAV 的可用性、任务和 busy 时间。每事件最多一个新策略分配。既有探测和 retarget 在同一事务里维持。增援先判断到达，新 anchor 固定至本次命令生成结束。新 Relay 承诺丢弃该 helper 之前的路线缓存，其他 UAV 的缓存仍可复用，回滚恢复该缓存。可能与新 anchor 冲突的 retarget 草案整体放弃。
3. 重用原控制链准备真实任务指令、发布当前 Search UAV、执行最终 adapter 整理。选中 UAV 的最终指令须与证书一致。事务成功后返回指令和 receipt，随后由正常 runtime step 处理动态 Safety、Shield、运动与检测。
4. 提案失效和重复提交在资源写入前返回具体拒绝码。助手写入、锚点、生命周期计数、探测、路线命令或最终 adapter 失败时，恢复保存的资源、执行缓存与 receipt。作用域位于强制事件完成和物理推进之间。物理步由已有 runtime 按实际观测继续判断并允许 Safety 抢占。

同一 UAV 至多持有一个 Search、Relay 或 probe 承诺，承诺与 UAV role 和 busy 状态一致。Safety 优先并可暂时压过既有 probe。Search 分配不会把 TaskManager 公共任务标为完成，真正完成仍由 Find 与既有生命周期处理。失败不会丢掉可在后续事件重新枚举的任务。

## 当前边界

本版每个控制事件只提交一个新动作。Legacy 同一事件可安排多个动作，因此四臂会测出接口和分配节奏的共同效应。首步证书覆盖 .1 秒，通过后仍需正常物理 Safety 和 Shield。完整路线、H300、安全尾部、任务效率和跨地图泛化需要独立闭环评测。批量匹配、学习 Retarget、主动 Defer 与新 critic 特征留给后续独立变更。CPU smoke 中候选量增长会提高路线证书成本，现阶段优先保证证据和合同。
