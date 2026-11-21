# 交接验证

本地 Git 根目录与项目目录分离。`extended-scalability` 原 HEAD 与团队远端基线一致，均为 `5143fbdc4bdadf907f59c433e81ab11b2e98c6dc`。交接使用单独的 Git worktree 和 `codex/cloud-handoff-20261121`。原始状态共 8,567 个已跟踪文件，15 个跟踪修改（12 Python、3 个旧 pyc）、104,999 个未跟踪文件及 101 个忽略文件。原 Git index、跟踪对象工作副本和状态清单由本地审计保留。未发现 Git submodule 或该根目录下的嵌套仓库/符号链接。另有独立 native 工作树及其冻结模型。

- 按 [快照清单](snapshot_manifest.json) 校验实际最新源码与迁移副本。对精选的历史文档和实验 JSON 做个人路径删减，记录变更前后的原始字节哈希。归档代码可读但部分协议保留了原始路径哈希与过期截止时间。
- 模型与配置解析、8 份模型 SHA-256 及 strict load、CPU masked forward、4 UAV × 2 臂 × 4 mission steps 在 WSL 已实际通过。相关单元测试 43 项通过。训练与兼容旧评测两个入口的 `--help` 通过。
- 使用 `run_scenario.py --prepare-only` 探测发现已有 `logs/latest` 真实目录导致同名 symlink 创建失败。云端说明改用已验证的 `smoke.py` 和 `paired_eval.py` 路径，既有运行器行为如实保留。
- 主要本地依赖为 Linux/Python 3.11.8、PyTorch 2.5.1 CUDA 12.4，依赖与 CPU、GPU 命令见 [复现说明](CLOUD_REPRODUCTION.md)。目标云端环境的实际 GPU 和独立 pip 安装待接手确认。
- 普通 Git 上传 50 MiB 以上警告、100 MiB 以上阻止。本仓库未使用 Git LFS 迁移这些模型。约 58.4 MiB 的 MARVEL actor 已在基底跟踪且按原哈希保留。新增二进制小于 50 MiB。扫描覆盖新提交内容、模型序列化元数据、凭据样式、缓存与大文件，并禁止将私人聊天附件路径纳入新快照。仓库已公开，已有远端基底的历史对象未重写。文件限制见 [GitHub 文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
- 两个近期模型的原容器含本地路径，公开推理导出逐 tensor 与原模型相同。原训练参数及 optimizer 保留于本地。未传输的完整长轨迹、私有目标/输入、日志和采样张量见 [轨迹索引](raw_trace_index.json) 和 [模型清单](checkpoint_manifest.json)。
- 交接期间运行轻量测试和短场景；历史独立 H300 和训练证据来自已保存文件。整个项目的 15% 目标仍需独立验证。


## 本地交接核验结果

独立 worktree 共准备 428 个与实际本地工作区字节一致的新增或修改源码与配置，精选历史归档 150 份，八份模型文件和 181 项原始轨迹索引。在此前 478 文件的冻结核对中，477 项与本地原字节一致。另一个 `scripts/read_original_gppo_paper.py` 含私人附件路径，仅在原工作区保留，清单记有排除原因和 SHA。对 533 个当前或归档 Python 文件完成语法解析。新增最大单文件为 12,043,710 字节。暂存内容的凭据样式和私人附件扫描零命中。历史源码中的本机路径默认值保留原字节供溯源，可移植入口已用 manifest 路径覆盖这一运行依赖。

原始工作区交接前后核验相同。分支、HEAD、index、所有跟踪文件的工作副本校验值、未跟踪与忽略文件的文件名清单均一致。八个原始关键模型由模型清单的 SHA 复核。新作品仅在独立 worktree。该 Git base 早先已在团队远端公开，沿用其历史。远端默认分支核对为 `main`。

验证覆盖本地 WSL CPU 和 GPU smoke。原始的历史训练续跑及完整 H300 执行未在目标云端验证，精确续训需要私有转移原模型容器和采样包。云端首任务可以使用源码、配置、推理模型和归档证据开展只读研究。


## 远端交接记录

已将内容提交 `c6df4512628c9216a286d787030e345fc306502d` 推送到 [专用交接分支](https://github.com/Azucat111/2026PD_MARVEL/tree/codex/cloud-handoff-20261121)。GitHub API 已读取到该内容提交，`git ls-remote` 验证专用分支指向它。随后以该内容提交为父提交补录本验证记录和一份决策空间审计。最终分支 tip 用 `git rev-parse HEAD` 或 GitHub 分支页面读取。

推送前后团队分支 `extended-scalability` 为 `5143fbdc4bdadf907f59c433e81ab11b2e98c6dc`，`main` 为 `4104ce0a97b72d295192b9f91d0580c35c52e982`。WSL 的 GitHub 网络响应超时且本地 gh 凭据验证失败，实际传输采用 Windows Git 和可用的凭据管理器，直接读取 WSL Git 对象。发布通过普通推送，已有分支与 PR 保持原状。
