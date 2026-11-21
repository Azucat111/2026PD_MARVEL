# Codex 本地首轮任务：接手 Claude Code 的 MARVEL+GPPO

你现在接手本地 WSL2 中 Claude Code 已开发的 MARVEL+GPPO 项目；**不是云端任务**。先阅读同目录的 `MARVEL_CODEX_HANDOFF.md`，并用当前项目 Git 状态、现有代码及实验产物核对。不要重复此前的长仿真。

预期路径：`<local-home>/MARVEL_INTEGRATION/MARVEL-main (1)/MARVEL-main`；预期分支：`extended-scalability`；历史 HEAD：`e7246a8`。不要修改 `gppo-integration`、`main`、冻结 `marvel_native`。保留本地 untracked artifacts 和脚本，绝不执行 `git clean` / `reset --hard` / 批量 `git add .`。

**本轮唯一任务：60-UAV Search travel-stall 只读诊断**。优先检查 `artifacts/extended_postfix_live_gppo_20261002T151131Z/` 中的 `search_progress.csv`、`search_events.csv`、`refill_events.csv`、`parking_health.csv`、`summary.json`，结合真实 allocator、observed travel time、PolicyNet next-hop 执行链。对比 16/30 UAV 成功案例，判断 60 UAV 两个 Search UAV 未在 450 步完成的主要原因。只有现场证据足够时才给“最佳可行候选排名”和预测 travel-time 的量化结论；否则明确说明缺什么。

本轮禁止修改生产代码、训练、重新运行 450 步、提交或推送。如证据不足，只设计**一个最小、短时、单一 60-UAV trace** 的补测方案并等待批准。

先报告环境与文件检查结果，再给诊断结论；结尾报告 Git 状态、是否发生新运行、是否修改文件，并提出**唯一**下一步建议。
