# Phase 2 验证记录

裁决 **ATOMIC_READY**，适用于扩展模拟器的单动作联合接口，支持下一阶段 PPO collector 接线和有界训练。下面的物理证据来自 CPU 短窗口，覆盖已见开发 seeds。完整 H300 的完成时长、安全和效益留待独立评测。

## 自动化测试

Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 1.26.4。在 `ac3cd49b54d4957eac2c0792d6d2392d10f12fa9` 运行 `python -B scripts/verify_joint_shadow.py --phase2 --out /tmp/atomic-phase2-validation.json`，结果为 157 通过、0 失败、2 跳过。包含全部 Phase 1 的 91 项。对 428 项交接源文件核验，三个有意变动的文件另以交接 Git blob 核验原件。八个冻结资产通过 SHA 和 CPU strict load，兼容 6D GPPO 另外通过 masked forward。

额外扩大 Relay 回归时有两个用例首先报 `FileNotFoundError`。这两个用例依赖历史 `artifacts/extended_gppo_joint_batch_20261012_attempt18/readonly_static64_112_control_inputs.json`，当前工作区及 Git 历史均缺该文件。运行器按文件存在性跳过这两个已有用例，保留其他同文件测试和原始断言。没有实际 H300 作业。四臂评测脚本另外用模拟 runtime 验证 3000 tick 完整计分及第 19 tick 截断不计分。

新增检查涵盖混合、单角色、空和单候选、相同 UAV/slot/anchor 冲突、容量、可达性、deadline、动态预警、纯枚举、旧快照失效、竞争占用、任务已结束、变更后的锚点、重复提交和下一事件。冻结 GPPO 与 Rule 使用完全相同的 mask，选中 ID 通过归位及当前版本校验。注入 helper 写入、Relay probe、提交后指令及 adapter finalizer 失败，核对资源、目标、计数、route cache 和 receipt 回滚。

## 四个归档事件的同状态反事实

对四个 Phase 1 归档快照重跑联合枚举。原 Search 与 Relay 见证全部位于联合 mask 内。两种角色各以相同环境 seed 从 reset 重放，在目标事件前运行相同 Legacy 路径。重建的公开快照摘要与 Phase 1 归档完全匹配。分叉前再配对机器人全字段、任务对象、隐藏布局的离线摘要及 Python、NumPy 和 Torch RNG 状态摘要。这里使用确定性重放，没有用公开快照去重建缺少的隐藏物理字段。所有摘要核验都在评测侧进行。

| 域 | 事件 step | 共享 UAV | 合法联合配对 | Search/Relay 第一个物理位置的相差 |
| --- | ---: | ---: | ---: | ---: |
| 30 静态 / 1280021 | 10 | 0 | 46 | 0.13734 m |
| 30 动态 / 1280022 | 10 | 1 | 86 | 0.09316 m |
| 60 静态 / 9164 | 10 | 1 | 188 | 0.12063 m |
| 60 动态 / 1981204 | 20 | 6 | 131 | 0.04430 m |

定向选择 8 次各得到真实 receipt、角色接管和控制指令，首步位置达到归档证书的数值结果，之后继续验证，总计 24 个分叉后物理 tick。各定向窗口均未观测到碰撞或禁飞违规。两位置差是反事实末端间距，每次单独的物理移动不超过 0.1 m。

在 30 静态同一混合图，额外执行冻结 GPPO 的真实分类采样。Relay 概率总质量约 0.000207667，诊断种子 3886 抽中 UAV 29 和 Relay slot 200015。动作通过原子提交，并在第一个物理 tick 移动 0.1 m。连续三个 tick 无观测到安全违规。种子是为覆盖 Relay 机制选取的。冻结旧模型的混合图贪心偏向 Search，后续 PPO 需重新学习跨角色的选择概率。

## 四臂短窗口与 Legacy

`python -B scripts/verify_joint_atomic.py --suite all --case CASE --steps 3 --out OUT` 对四域分别运行四臂。每个 arm/域跑 3 秒，合计 16 个窗口和 480 个物理 tick。两个联合臂留存 mask、动作和实际 receipt，四臂均留存物理反馈。两个联合选择器共享一个 planner 和提交路径。混合图同状态比较只以定向重放口径判断，闭环分岔后的 mask 不要求逐元素相同。

30 静态 Legacy 的 30 个物理 tick 与第一阶段归档完全匹配，Shadow 开关前后逐 tick 观测也完全匹配。额外的 30 静态 sampled 诊断在 Relay 缓存修订后再次通过。主四域采集及定向验证使用核心实现提交 `4567b638caf4c4f0dc9e445ca9946b178628ff28`，最后一笔路线缓存修订使用专项回归和 sampled 物理复验。原始 trace 采用无损 gzip 保存，运行器 identity 与 SHA-256 可在证据清单核查。

立即可说明的是联合选择、事务和真实物理接管。尚未验证特定长程 route、H300 任务时间和安全尾部、冻结 critic 的联合训练稳定性以及正式 15% 效益。单事件只提交一个新动作，会影响分配节奏，需用四臂独立量化。

命令见 `JOINT_PHASE2_REPRODUCTION.md`，原始证据见 `artifacts/joint_atomic_20261121/manifest.json`。
