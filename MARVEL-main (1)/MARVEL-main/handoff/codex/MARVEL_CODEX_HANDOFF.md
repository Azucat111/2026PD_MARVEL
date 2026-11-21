# MARVEL + GPPO：Claude Code → Codex 本地交接

> 本文依据截至 2026-10-09 用户展示的 Claude Code 报告及 Git 输出整理。**不是新的现场审计**。Codex 开始工作时，应先核对本地仓库、提交、目录和文件是否仍与本文一致；不一致则以现场证据为准并说明差异。

## 1. 接手方式与工作环境

- **直接在原来的 WSL2 项目上工作，不重新 clone、不切换到 Codex Cloud、不移动代码到 Windows 文件系统。**
- WSL/Linux 项目路径：`<local-home>/MARVEL_INTEGRATION/MARVEL-main (1)/MARVEL-main`。
- Windows Codex App 需要使用 **Local** 工作方式，并确保代理/命令**实际运行在 WSL**（而非 PowerShell）。可以在应用里选择 WSL 环境并重启，再通过 `\\wsl$\<distro>\home\nick\MARVEL_INTEGRATION\MARVEL-main (1)\MARVEL-main` 选中目录。发行版名称必须实际检测，不能凭空假设是 `Ubuntu`。
- 先检查 `pwd`、`git status --short`、`git branch --show-current`、`git log -1 --oneline`、`python -c 'import torch; print(torch.cuda.is_available())'`。CUDA 检查还可以用 `nvidia-smi`；若环境不满足，**先报告，不要隐式改 CPU 或执行长仿真**。
- 本地未跟踪的 `artifacts/`、场景、脚本是历史证据，不是垃圾。未经确认不得清理、覆盖或批量 `git add .`。

## 2. Git 与冻结边界

- 当前目标分支：`extended-scalability`；用户最近一次展示 HEAD：`e7246a8`，提交说明 `fix: align extended execution and node admission with the physical UAV`。
- 用户已于 2026-10-09 成功 `git push team extended-scalability` 到团队仓库 `Azucat111/2026PD_MARVEL`。此后是否有其他提交必须现场检查。
- `gppo-integration @ b7a167e` 是先前通过的集成基线；**不要修改或推送该分支**。
- `marvel_native` 是与冻结实现保持一致的正确性路径，300-step parity 通过（301 个快照一致）。**不能为了 extended 改动它的决策、感知、运动、Relay、Safety 语义**。
- `main`、`origin`（`marmotlab/MARVEL`）都不在本轮操作范围。
- 不得擅自 commit、push、merge、rebase、reset/clean 或改写实验清单。第一项任务仅只读诊断。
- 最近 `git status` 曾看到三份追踪的 `__pycache__/*.pyc` 显示 modified，以及很多 untracked files；不要把它们当成需清理的错误，也不要擅自 restore。

## 3. 已完成且不要重做的阶段

1. MARVEL-native GPPO 集成正确性：Search、Relay、Safety、GPPO 行为与冻结参考在 300 mission steps 内对齐。
2. Extended 规模优化：shared planning graph 缓存；heap Dijkstra；坐标/邻接索引优化；CUDA batched PolicyNet 可选后端。默认 CPU+sequential，不得未经验证切换 native。
3. 局部性能实验曾否定一些方案：CPU batching 对 60 UAV 无稳定收益；全图 swept-edge clearance 约使运行变慢 25 倍；raycast vectorization 变慢；GPPO travel-time 的简单 heap/index 改造收益很小且已回退。**不要未经新证据反复尝试这些路线。**
4. `e7246a8` 修复了 extended 模式 360-node window **索引空间不匹配**：`get_observation` 对邻居索引做了窗口重映射，但旧 `decode_waypoint` 用它索引未裁剪全图坐标，导致本应是 2.9m 的 next-hop 解码成数百米外点。修复还在新节点入图时检查 UAV 半径 `0.2m` 的 clearance（不做昂贵的全边 sweep）。PolicyNet 原始 logits、action_index、waypoint_index 未改变，只修正落点解析。
5. 修复后 120-step permanent parking 明显下降：8 UAV 5/8→0/8、16 UAV 15/16→2/16、30 UAV 26/30→5/30、60 UAV 48/60→3~4/60；`shield_static=0`。仍存在少量 `grid_obstacle` 和残余停车，不能宣称碰撞全无。

## 4. 最近的权威实验与边界

**最新验证：`POSTFIX_LIVE_GPPO = PARTIAL`**，没有新生产代码修改。

本地证据目录（需检查存在）：
`artifacts/extended_postfix_live_gppo_20261002T151131Z/`

优先读取：
`POSTFIX_LIVE_GPPO_REPORT.md`、`search_progress.csv`、`search_events.csv`、`refill_events.csv`、`relay_events.csv`、`parking_health.csv`、`timing.csv`、`summary.json`。

实验设置：extended ~500m 场景；真实 MARVEL PolicyNet + CUDA batch；真实 GPPO checkpoint/TaskScheduler；Search、Relay、Safety 实际路径。**受控 activation-state 初始条件**由传感器视野联合形成，起始 exploration rate ~`0.310452`，随后 Search 通过冻结阈值 `0.30` 正常激活、由 scheduler 真正生成任务行；不是自然从 reset 达到 30%，也没有 synthetic Search slots。

最近报告的结果：

| UAV | 步数 | 首个 Search completion | 首次 refill | 稳态均值 | 最近 approach |
|---:|---:|---|---|---:|---|
| 8 | 450 | 未发生 | 未发生 | 0.606s | 158.6m |
| 16 | 144 | step 93 | step 94 | 1.108s | 11.7m |
| 30 | 228 | step 177 | step 178 | 2.345s | 12.2m |
| 60 | 450 | 未发生 | 未发生 | 5.791s | 117.2m（UAV 43） |

注意：16/30 UAV 的完成来自 **sensor visible-cells detector → `runtime.completed_search_heat_ids()` → `_sync_search_completions`**，**并非** 3.5m dwell/service-start 路线；后者尚未验证。Relay 未自然生成任务行，不能宣称 Relay 完整生命周期成功。

60 UAV 450 步时残余 parking 约 3/60，未随时间持续恶化。60 UAV 稳态 5.791s 的较高成本与更大的随时间生长的图有关；不同阶段/不同图状态下的秒数**不能直接当作相同条件加速比**。60 UAV GPPO scheduler 近期报告 ~29.5ms/step，但由于 Search task rows 较少，不能由此推断 GPPO 在任意高任务密度都没有瓶颈。

相关历史目录（需要时读，不要一次全部扫描）：
- `artifacts/extended_execution_path_20261002T121724Z/`：索引+clearance 修复证据、停车对照。
- `artifacts/extended_live_gppo_baseline_20261001T144955Z/`：修复前 live GPPO baseline。
- `artifacts/extended_sensing_correctness_20261002T050159Z/`：证明真实 GPPO 配置已开 `los_occlusion`，根因不是 occlusion 开关。
- `artifacts/extended_clearance_consistency_20261002T065023Z/`：昂贵 all-edge swept-clearance 原型，无需复跑。

Checkpoint 路径历史记录（必须先确认文件存在，且不要重训练或覆盖）：
- GPPO: `<local-home>/MARVEL/artifacts/phase14_v9_frozen/gppo_phase14_v9_ckpt80.pth`
- MARVEL low-level checkpoint: `load_model/MARVEL/checkpoint.pth`

本地曾有辅助脚本 `scripts/trace_search_stall.py`（未跟踪，**尚不代表已经验证过输出**）；Codex 必须先读代码和已有结果再决定能否使用，不能直接启动大规模实验。

## 5. 现在唯一要做的事（第一轮）

**60-UAV SEARCH TRAVEL-STALL DIAGNOSIS，优先只读。**

问题：为什么在 60 UAV、450 mission steps 下，2 个 Search UAV 没能使热区进入探测完成条件？是距离本来太远、分配质量、travel-time 估计不可靠、路径绕行、图连通性、拥挤、残余停车，还是其他？

建议工作顺序：
1. 确认 WSL 工作目录、Git HEAD、已有 artifacts，以及 CUDA/checkpoint 基本可用性。不要修改工作树。
2. **只读取已存在**的 `search_progress.csv`、`search_events.csv`、`parking_health.csv`、`summary.json`，必要时延伸至对应代码。先搞清 CSV 列定义、采样频率、是否含位置/预测 travel-time；没有的字段不能猜。
3. 重建 60 UAV 两个 Search 分配：heat ID/坐标、分配给哪两个 UAV（历史报告列出 UAV 0、43；以 CSV 核准）、初始距离、时间序列的距离/实际移动量/停滞阶段/碰撞事件。和成功的 16/30 UAV 作同定义对比。
4. 评估“选择的 UAV 在可行候选中的排名”，**只有在原始分配时刻的全体候选信息或可重建 snapshot 存在时**才计算。没有证据就写“不可判定”，不要用最终位置替代初始候选。
5. 区分：`BAD_ALLOCATION`、`TRAVEL_TIME_ESTIMATE_MISMATCH`、`PATH_MEANDERING`、`CROWDING`、`RESIDUAL_PARKING`、`GRAPH_CONNECTIVITY`、`DISTANCE_LIMITED`、`OTHER`。用可量化数据支持主因，给出最小的下一项验证动作。

严格限制：
- **第一轮不运行 450-step 实验，不训练，不修改源代码，不提交/推送**。
- 若现有 CSV 不足以确定原因，只提出一个**单规模、短时间、有停止条件**的最小增量 trace 设计，等待用户批准再执行。
- 输出直接使用 Markdown 摘要和引用的本地文件路径/列名；区分“已证明”“推测”“缺证据”。

## 6. 测试与结果口径

历史报告：Native parity 通过（最近专项 65 tests passed）；平台全测 `166 passed, 1 failed`，失败为早已存在的 `test_platform.py::test_dynamic_obstacle_event_and_target_detection`。这些是**历史记录**，新任务若修改代码必须自行回归，不能把旧数据说成本次执行结果。

推荐首轮输出：
- 环境/仓库/证据可用性确认。
- 60 UAV Search 任务分配和轨迹的事实表。
- 与 16/30 的同口径对比。
- 因果解释、局限性、根因分类和下一步单一建议。
- `git status --short`（前后）；`new_runs=0`；`production_changes=0`；`pushed=NO`。

**核心原则：复用已有成果、保护 native frozen parity、把 60-UAV 停滞解释清楚以后再决定是否动代码。**
