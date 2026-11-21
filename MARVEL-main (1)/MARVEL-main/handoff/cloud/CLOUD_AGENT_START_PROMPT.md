# 给云端 Codex 的首任务

你在团队 GitHub 仓库 `Azucat111/2026PD_MARVEL` 的 `codex/cloud-handoff-20261121` 分支接手 MARVEL + GPPO。项目在 `MARVEL-main (1)/MARVEL-main`。目标是在静态和动态障碍下，让 30–60 UAV 的完整任务耗时相对同条件 Rule 降低至少 15%，并满足物理安全、可达性及公平配对。当前尚未完成整体正式验收。

1. 在 Git 根执行 `git status --short --branch` 和 `git rev-parse HEAD`，核对交接分支、源码和 [验证记录](HANDOFF_VALIDATION.md)。进入项目目录，读 [交接说明](CLOUD_HANDOFF_README.md)、[源码清单](CODE_AND_EXPERIMENT_MANIFEST.md)、[Rule 审计](RULE_BASELINE_AUDIT.md) 和 `handoff/codex/MARVEL_GPPO_CONVERSATION_CONTEXT_20261121.md`。历史材料中的日期和旧授权属于相应实验。按 [复现说明](CLOUD_REPRODUCTION.md) 做 CPU 校验，确定可用依赖和模型哈希。
2. 保留原始 MARVEL PolicyNet 和 frozen native GPPO 语义，延续 GPPO 强化学习方向。这阶段只读研究 GPPO 的真实决策权。在 `scheduler._high_level_update` 的事件边界沿 Search/Relay 触发、候选生成、合法性、预占、`GPPOEventAllocator.allocate`、`TaskGraph.action_mask` 和 `ExtendedGPPOInferenceAdapter.assign` 追踪一条真实数据链。确认当前哪些决策在规则预筛选中已经完成。约束必须基于实际公开观测，真值只用于离线物理核验。
3. 先做事件级 Search/Relay 联合候选的只读 shadow enumeration。复用现有 `SearchSlotBuilder`、Relay demand/anchor、路线可达性、deadline 和 safety 门。先从公开保存状态及轻量运行中挑少量可重建事件。记录每个候选的角色、源触发、合法性掩码原因、UAV/任务 ID、轨迹执行接口、资源冲突和不同动作语义。若没有足够状态恢复同一时刻的候选集，明确具体缺的字段。从相同公共同步状态验证是否存在两个以上安全、合法且能影响不同任务的动作，并区分真实联合图与两塔顺序调用。保留无候选、只有一个候选和跨角色同时可选三类分母。优先复用 GPPOEventAllocator 的候选与上下文接口，以及现有 Search catalog 用例，禁止重做无增量收益的 donor 硬掩码。
4. 基于枚举证据写最小接口设计。仅当确有有意义的动作自由度时，再固定有限预算、配对对照与停止条件，在独立实验分支上做小验证和 PPO。训练优先复用现有模型、冻结行为及 PPO 更新，把 KL 越界时恢复 model＋Adam 的机制与原训练协议分清。避免过早启动大规模 H300。对照共享相同合法候选和执行器，报告完整任务时钟、全窗安全、固定分层和失败；避免把选模型的数据当独立验收。
5. 使用可审查分支或 PR 交付源码证据、候选审计、实际命令、有限结果与未知项。第一项任务完成标准是能以文件、行号和公开可观测输入说明当前决策权限、确定联合候选接口的可行性边界，并用证据说明是否值得推进。缺失资产按清单登记并复用现存摘要。

