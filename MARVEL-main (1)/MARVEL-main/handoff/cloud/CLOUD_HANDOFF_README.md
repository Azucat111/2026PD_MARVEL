# MARVEL + GPPO 云端交接

从 [云端首任务](CLOUD_AGENT_START_PROMPT.md) 接手。Git checkout 后，项目工作目录是仓库内的 `MARVEL-main (1)/MARVEL-main`，本文及命令以该项目目录为基准。团队仓库为 `Azucat111/2026PD_MARVEL`，交接分支为 `codex/cloud-handoff-20261121`。

本地源码基底是 `extended-scalability` 的 `5143fbdc4bdadf907f59c433e81ab11b2e98c6dc`，也是迁移前实际查询的团队远端分支 SHA。交接快照叠加了已核对的实际本地源码、配置和历史证据。文件来源、哈希和公开副本的编辑记录见 [快照清单](snapshot_manifest.json)。

目标是在静态与动态障碍、30–60 架无人机场景下，相对同执行条件的规则基线缩短完整任务时间至少 15%，并同时满足安全和可达性条件。MARVEL PolicyNet 是冻结的低层探索策略，上游训练入口 `driver.py` 使用离散 SAC。高层 GPPO 由本地 AHGNN、配对 actor 和图 critic 组成。执行器负责事件触发、合法性、容量、任务生命周期、切换、运动和物理安全。现有 GPPO 是对论文高层思想的物理闭环适配，未核实到论文作者的官方训练仓库。论文出处见 [DOI](https://doi.org/10.1109/TASE.2025.3593125)。

当前实现支持公开/延迟观测、Search 可达性和恢复前缀、Relay 锚点和增援、动态 Safety、批量 MARVEL 推理。已核实的共享修复将到达的 Relay 增援处理移到锚点刷新之前。历史已见场景由 120.4 秒降到 44.4 秒，规则和 GPPO 都受益。

| 证据 | 结果 | 使用边界 |
|---|---|---|
| Round 8 原目标独立验收 | 12 对，点估计约 28.98%，完整门槛失败 | 见历史 acceptance/verification JSON |
| 11 月 10 日独立套件 | 15/16 臂完整，缺一个规则臂 | 不完整，不能推断总体 15% |
| 11 月 11 日受保护更新 | 一步保留，第二步越界同时回退模型和 Adam。已见 30 动态 38.0→29.9 s | 同一 80 m 图的新三种子上旧/新均为 85.0 s，规则 81.6 s |
| 最新 donor 跨场景验证 | 四新种子、三臂、12 条均安全，规则 103.5 s、两种 GPPO 均 82.4 s | donor 增量 0%，四组动作和指标轨迹与原 GPPO 一致 |
| 同权重七组回顾合计 | 规则 185.1 s、GPPO 167.4 s，降幅 9.56% | 单地图描述性汇总，整体验收待完成 |

受保护的 6D 模型是最新固定比较的参考权重，两份较新的私有训练元数据原件在本地保留。云端推理导出和各 checkpoint 的协议、尺寸、原始与传输哈希见 [模型清单](checkpoint_manifest.json)。

训练代码入口 `scripts/train_extended_gppo_batched.py` 及 `scripts/gppo_parallel_rollouts.py` 已迁移。历史组内基线和事务回退位于 `artifacts/gppo_learnability_two_hours_20261111/guarded/`。它依赖历史样本和原训练合同。新环境的 smoke 与配对入口参见 [复现说明](CLOUD_REPRODUCTION.md)。

云端第一项研究为只读事件候选枚举。先证明真实事件图中存在跨类型、可执行且语义不同的动作，然后定位代码对 Search/Relay 先后顺序、资源预占和动作集合的限制。训练、扩大评测以及改动安全或执行语义均需另立技术合同。实现与历史材料按 [源码和实验清单](CODE_AND_EXPERIMENT_MANIFEST.md) 和 [规则审计](RULE_BASELINE_AUDIT.md) 找到。
