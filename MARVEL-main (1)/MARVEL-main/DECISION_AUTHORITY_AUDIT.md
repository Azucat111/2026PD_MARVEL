# 当前决策权限审计（第一阶段）

基准提交：`5692ca4fe5c56f398707deea87c1efc844297b71`，团队仓库 `Azucat111/2026PD_MARVEL`。本研究分支 `codex/joint-gppo-shadow-20261121`。路径均相对于本项目 `MARVEL-main (1)/MARVEL-main`；以下行号来自该提交的实际源码，生产文件本轮未修改。

预置目录 `/workspace/marmotlab/marvel` 原来是上游 `marmotlab/MARVEL@318c2a6`。先通过 `git ls-remote` 核实交接 tip，再独立 clone 到 `/workspace/2026PD_MARVEL`，准确检出交接提交后新建分支。原工作区及交接分支历史未改动。已阅读用户指定的八份交接/历史材料；历史 104 次纯角色图只作为已存描述性证据，不重新运行历史验收。

## 实际调用链与边界

```text
benchmark_fair_search.run                         scripts/benchmark_fair_search.py:26
  MARVELPolicyAdapter.get_actions                 utils/policy_adapter.py:244
    _policy_actions → 冻结 MARVEL waypoint
    scheduler.observe                            :287（公开感知、航向、恢复反馈）
    GPPOTaskScheduler.apply                       integrations/gppo/scheduler.py:1727
      clock.should_update                        :1730；protocol_profile.py:150
      tracker.observe                            :1736
      safety.pre_motion_update                   :1746（先抢占）
      _high_level_update                         :1180
        tracker.positions                        :1181（控制快照插值延迟）
        _maybe_activate_search                   :1189
        _sync_search_completions                 :1193
        _release_proven_unreachable_search       :1194
        _release_unsupported_search_routes       :1195
        _reconcile_uav_state                     :1200
        [本轮 shadow 的唯一同步边界：_refresh_search 入口]
        _refresh_search                          :1202 → allocator.allocate :863
        _refresh_relay                           :1203
          初始 demand / 已到达增援                :962 / :1038
          _allocate_relay_result                 :1054 → allocator.allocate :1093
        _reconcile_uav_state                     :1207
      Search/Relay override、冲突检查、Safety 最终覆盖
    _certify_extended_actions                     utils/policy_adapter.py:290,319
    _stationkeep_completed_tasks                  :291,293
  SimulationRuntime.step                         utils/simulation_runtime.py:423
    before_physics_tick / shield / 同步 dynamics / 真实碰撞 / 感知 / 完成
    scheduler.post_physics_step                   integrations/gppo/scheduler.py:1419
```

`allocate` 不是只选一次。它复制 UAV/slot 后循环建图，每次配对将**局部** UAV 设为不可用、task 标 assigned；调用方取得整批结果后才写实际所有权。Search 的整批真实预占已在 Relay 建图前发生。各配对的 `current_time` 可以相同，仍是不同资源状态，不能拼接历史两张图声称原 actor 看到了联合图。

## 决策权限表

表中“不可放宽”指本阶段原型和下一代首版合同；部分启发式可在未来独立消融，但不能伪装为物理必需条件。

| 决策 | 当前控制者、源码位置 | GPPO 能否更改 | 不能放宽的约束 / 最小改造位置 |
|---|---|---|---|
| 是否发生高层更新 | `GPPOControlClock.should_update`，`protocol_profile.py:150`；`apply:1730` | 否 | 保留 1s 控制 / .1s 物理时钟；事件触发可加公开事件队列，不能自行推进时钟 |
| 公开状态年龄 | `StalePositionTracker`，`scheduler:1181,1736` | 否 | 主图继续用同一延迟快照；own-sensor 只用于执行证书，不偷偷替换 actor 中的同伴位置 |
| Safety 抢占 | `apply:1746`、`before_physics_tick:1532`；`safety_layer.py:344` | 否 | Safety 最高优先；不得供 actor 拒绝。联合候选边界放在抢占/释放之后 |
| Search 激活 | `_maybe_activate_search:578`（公开 exploration_rate、threshold、heat prior）；TaskManager `_should_activate`，`utils/task_manager.py:60` | 否 | 两层任务激活均需满足。未触发热点不是当前合法动作 |
| Search 完成 | `_sync_search_completions:611`、`_update_search_service:643`；`runtime.completed_search_heat_ids:876` | 否 | extended dwell 不够，须环境反馈的 Find 确认；不访问隐藏坐标 |
| Search 活跃槽数量 | `_refresh_search:781`，`max_search_uavs` 来自 checkpoint mixed_config，`scheduler:127` | 否 | 并发容量守恒。可在统一 residual mask 中执行相同容量，无须扩大预算 |
| Search 候选目录、排序 | `SearchSlotBuilder.build`，`event_sources.py:61`；`scheduler:801` extended `include_all_pending=True` | 可选 pending 任务，不能创造任务 | extended 已提供全 pending 目录；native 仍截断。无需重新实现 Search catalog |
| Search 活跃任务/热点排除 | `_refresh_search:753,790`；builder `event_sources.py:76` | 否 | active、非 serviced、非已占用 hotspot；保留任务生命周期 |
| Search 资源屏蔽 | `_refresh_search:814`（Search、Relay helper、Safety）；`TaskGraph._uav_is_available:402` | 否 | 不得抢占旧承诺。新事件中尚未提交的同拍资源可延后统一 commit |
| 已有 Search 无法到达时释放 | `_release_proven_unreachable_search:1137` | 否 | 观测 closed-cut 证据，不能将 UNKNOWN 当墙。联合接口在释放后取快照 |
| Search 支持优先 / yield | `_release_unsupported_search_routes:1157`；`route_support.py:11,41` | 否 | 这是候选支持启发式，**不是物理不可达证明**。首版复用；未来调整须单独合同并保持 R1/P1 相同 |
| Relay 是否需要 | `_refresh_relay:945,954`；extended `mesh_relay.py:33` | 否 | 公开最大连通分量比例低于 threshold；extended 不是 native 基站两跳模型 |
| Relay 任务与 anchor | `ObservedMeshRelayDemandBuilder.build:70`、`_choose_anchor:41` | 否 | 从公开 FREE anchor 候选（`scheduler:1506`）选确定点，通信端点预测不等于运动证书。首版不学 anchor |
| Relay target / endpoint 保留 | `mesh_relay.py:83,84` | 否 | selected target 与 backbone endpoint 禁止充当 helper；保留该约束 |
| Relay 初始容量 | builder `max_demands`；`mesh_relay.py:83` | 否 | 不提前用补充 helper 总预算替代初始预算；统一接口显式保存 event 类型 |
| Relay 增援条件 | `_augment_arrived_relay:1038` | 否 | 全部旧 helper 已到 anchor .5m 内、通信仍不足、剩余预算。首版不取消到达条件 |
| Relay 补充预算 | `extended_relay_helper_budget`，`mesh_relay.py:13` | 否 | `max(frozen_budget, ceil(N/10))`，只在到达增援时使用 |
| Relay 到达 / retarget 顺序 | `_refresh_relay:928`（先 augment，后 `refresh_assignments`） | 否 | 保留已修复顺序；不能又先移动 anchor 导致漏掉到达事件 |
| Relay 非 GPPO retarget | `mesh_relay.refresh_assignments:104` | 否 | 公开预测须保持非 helper 连通、LCC 不减。未来候选快照须固定 anchor 版本 |
| Relay 稳定释放 | `_refresh_relay:915`；`mesh_relay.py:184`；`relay_release.py:48` | 否 | 移除 helpers 的反事实连通、target 恢复、连续3控制步；释放当拍直接 return，不立刻重触发 |
| Relay 额外 probe | `_refresh_observed_relay_probe:972` | 否，规则直接选择 | 独立观测探索动作，非 actor task slot。本轮记录已有 probe 预占，不伪装为 Relay 配对；首版联合动作不纳入 probe |
| 两角色先后顺序 | `_high_level_update:1202` | 否 | **最小关键改造点**：在 reconcile 后统一 collect、mask、choose、commit；不能仅在 adapter 拼图 |
| 两角色新预占谁先取得 | `_refresh_search:875` / `_allocate_relay_result:1115` | 否 | 新承诺采用原子资源事务；一机一槽、旧承诺不可偷释放 |
| UAV 角色能力 | `RuntimeGraphBuilder._allowed_types_for:53`、`_build_uavs:86` | 否 | task 的 assigned_robot_types、alive、available、busy_until；不扩充能力标签 |
| 图基础合法性 | `TaskGraph._task_is_executable:387`、`_build_action_mask:409` | 只在 `~mask` 内选择 | `True=invalid`；active、未完成、未分配、前驱已完成、alive/capability/availability |
| 旅行时间 | allocator `_travel_time_matrix:85`，`extended_travel.py:11` | 选择配对但不改估计 | 已知路由 + UNKNOWN residual 是估计下界，不证明全路径可行 |
| Search deadline | `_apply_deadline_mask:99`；`search_deadline.py:10` | 否 | 使用剩余物理窗口、已认证恒速与感知上界；通过不证明能完成 |
| Search 可达性 | `_apply_observed_reachability_mask:123`；`observed_reachability.py:59` | 否 | 保守已观测 closed-cut；不读真值地图 |
| Relay radio/motion guard | `_allocate_relay_result:1090`、`mesh_relay.candidate_mask:150` | 否 | 计划端点 LCC 严格增加且原分量非 donor 节点仍连通；观测运动不可达过滤；统一图 guard 只对 Relay 行应用 |
| Rule 选择 | `benchmark_fair_search.py:57`；`choose_rule_assignment`，`integrations/rule_scheduling.py:10` | 不适用 | 精确词典序 `(-priority, raw_total_time, taskID, UAVID, ti, ui)`，非有限跳过；本轮原函数零修改 |
| GPPO 选择 | allocator `event_allocator.py:286`；`heading_adapter.py:31` | **可选当前图中的 task–UAV 配对** | deterministic masked argmax；无 Continue/Defer、release、anchor 或 role-order 动作 |
| 局部批内资源更新 | `event_allocator.py:332` | 只能通过此前选中配对间接影响 | assigned UAV、busy_until、task assigned；联合候选须逐选择更新 residual 状态 |
| 实际承诺落实 | `_apply_uav_state:431`、`_reconcile_uav_state:472`；`scheduler:880,1120` | 否 | 一机不可同时 Search/Relay；释放不是 busy_until 到期自动替代任务完成 |
| waypoint 与传感服务 | `apply:1775,1851`；`_search_action:1592`、`_relay_action:1521` | 否 | MARVEL fallback、观测恢复、服务扫描、公开 warning；保留原动作语义 |
| 物理最终权威 | `runtime.step:423,438,465,497,525`；`SafetyShield.filter_motion:77` | 否 | .2m 五探针、.5m Shield 间隔、.4m 物理重叠判定、动态 Safety；shadow 不触碰 |

## 判断与未决事项

现有 GPPO 仍有真实 task/UAV 学习自由度，纯角色图不等于无事可学。但它无法决定“同一新可用 UAV 先用于 Search 还是 Relay”。本轮的同边界枚举证据见 `JOINT_CANDIDATE_SHADOW_AUDIT.md`，避免用历史摘要反推不存在联合机会。

真正的最小改造是调度器的候选与承诺接口，而非换掉 MARVEL、重新造路径规划器或放松 Safety。现有动作执行原语支持候选的第一步，不等于已经实现了原子联合批分配、完整任务可达性证明或 15% 性能提升。
