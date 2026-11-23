# 真正的事件级回放训练

## 相同历史预算

首批原始归档中有 22 条可读完整物理轨迹，去重后使用 15 条完整回合和 78 条实提交事件。它们来自 9 个独立 seed 组。四域事件数为 30 静态 38、30 动态 23、60 静态 9、60 动态 8。Search-only 30、Relay-only 15、混合 33。执行总数为 Search 58、Relay 20，其中混合图执行 Relay 5 次。行为概率有记录 39/78，缺失从原始存档继承。15/15 任务安全成功。

两个模型均从独立随机权重训练，使用同一份回放 `1f3372488e97d143d554eca66029dc086c4b382f2de93b1be6e00e573d5b241a` 和相同组均衡抽样，各完成 600 次 Adam 更新和 12 次目标同步。每个训练 19,200 次抽样，其中重复体验未计为新增环境样本。

| 基于 78 个事件 | DDQN | Discrete SAC |
|---|---:|---:|
| 优化墙钟，不含启动（秒） | 10.62 | 24.60 |
| 第一次 critic Huber | 0.06060 | 0.06952 |
| 第 600 次 critic Huber | 0.00553 | 0.00738 |
| 贪心与资料中执行动作相同 | 23/78 | 25/78 |
| 贪心无该确切公开状态的执行记录 | 54/78 | 52/78 |
| 混合图贪心选 Relay | 4/33 | 6/33 |

离线贪心与单条已执行动作的差异只证明网络输出发生变化，物理证据来自独立评测。对已执行动作的离线预测与原行为继续回报的平均绝对差分别为 0.408、0.398（30 秒缩放单位）。它们受离策略继续行为和 SAC 熵正则影响，用于诊断，不作为实际回报的无偏估计。

| 模型 | SHA-256 |
|---|---|
| `archive-ddqn.pth` | `1716181d971779f8c54e9801d45862f71367925fae9289b7d5be9ab991ac5d4e` |
| `archive-sac.pth` | `4aa14655806f551414814fd2ca621109a4d67c46183fa7c3555e0ca1fc437666` |

参见 [回放审计](EVENT_LEVEL_REPLAY_AUDIT.md) 和 [运行复现](MATURE_RL_REPRODUCTION.md)。

## 长时延与值分布诊断

同一 78 事件回放、seed 和更新数下，三步 DDQN 把已执行行为继续回报的预测 MAE 从 0.408 降至 0.328，但 78/78 的贪心动作在同一公开状态无实执行支持。11 次贪心 Q 为正，而原无折扣时间成本回报始终非正。它只作为离线负证据，保留模型 `f0ac76c73cb9f5c3c422d6a3afde67cd94e813112d29c59287a8a4bbd10eabb2`。

在相同三步目标上附加 0.25 权重的 CQL 项后，同状态无支持贪心降为 15/78，正 Q 降为 1 次，混合图上 0/33 贪心执行 Relay。该权重下保守项压低了候选外推，并损失需要研究的跨角色行为。模型 `c84722f46c7f601913c813591ba894ef61585f9d41a8ce844f8a83300399caec` 保留为离线诊断。两者均未进入主竞赛和冻结选型。结果支持采集更多有真实 Relay 回报的分支。

```bash
"$py" -B scripts/train_mature_rl.py --replay "$out/historical-replay.json.gz" --algo ddqn \
  --seed 6177 --updates 600 --n-step 3 --out "$out/archive-ddqn-n3.pth"
"$py" -B scripts/train_mature_rl.py --replay "$out/historical-replay.json.gz" --algo ddqn \
  --seed 6177 --updates 600 --n-step 3 --cql-weight .25 --out "$out/archive-ddqn-n3-cql.pth"
```

历史回放的 63 个有后继决策的间隔中位数为 1.0 秒，范围 1–33 秒；15 个末端决策的物理间隔为 262–296 秒。全体训练事件成本的 69.3% 挂在各回合最后一次决策之后。这是对长时延回报的实际量化，也是检验多步回传的依据。

## 中间回放诊断

在四个新 J0 对照和首两个探索回合完结后构建中间回放。去重后为 20 个完整 episode、13 个 seed 组、105 个事件。混合图增至 39 个，实执行 Relay 仍为 5 个。按原规定给 DDQN 与 SAC 各 600 次更新，作为数据扩充过程诊断。

在 `30_dynamic_24110602` 第 1 秒的同一快照上，两个已执行 Search 的真实剩余回报分别为 −3.5200 与 −1.3833。中期 DDQN 对两个动作估值 −1.5012、−1.3018，SAC 最小 Q 估值 −1.7481、−1.4742。两个算法都排对了这对实执行动作，估计间距只有真实间距的约 9% 和 13%。DDQN 的贪心是另一个 Search，SAC 贪心是未实执行的 Relay。这个条件对比不能为具体贪心动作提供真实收益。

| 中间回放 105 事件 | DDQN | Discrete SAC | DDQN+CQL 0.06 |
|---|---:|---:|---:|
| 无同状态实执行支持的贪心 | 87 | 92 | 19 |
| 贪心正 Q | 10 | 2 | 1 |
| 混合图贪心 Relay | 14/39 | 27/39 | 1/39 |

0.06 的保守项延续了显著压制少数角色的现象，故保留为离线负结果。额外的控制模型都保存在证据目录，中间模型不进入最终物理选型。主比较继续考察物理数据中真实混合 Relay 覆盖扩大之后的两种算法。

复现中间回放时可使用已封存的 `historical-replay.json.gz` 为 `--parent-replay`，按序加入 `j0-30_static_24110601`、`j0-30_dynamic_24110602`、`j0-60_static_24110603`、`j0-60_dynamic_24110604`、`explore-30_static_24110601`、`explore-30_dynamic_24110602`。训练仍用 seed 6177 和 600 更新。低权重组加 `--cql-weight .06`。

## 主对照的扩展回放训练

相同扩展数据上每个算法重新从固定随机种子初始化，完成 600 个 Adam 更新、12 次目标同步和 19,200 个实际回放抽样。两臂共享 126 个独立动作 transition 和真实数据预算，使用同一入组序列。

| 指标 | 最终 DDQN | 最终 Discrete SAC |
|---|---:|---:|
| 优化墙钟（秒） | 7.35 | 14.25 |
| 第 600 次 critic Huber | 0.01180 | 0.00937 |
| 贪心无同状态实执行支持 | 100/126 | 117/126 |
| 贪心正 Q | 7 | 2 |
| 混合图贪心 Relay | 4/43 | 11/43 |

DDQN 的正 Q 仍违背无折扣纯负成本的上界，表明合法图覆盖和必要的价值约束仍需改善。SAC 软值含未来熵项，正软 Q 不能按同一上界解释。SAC 最末批平均合法熵 1.842 nat，目标 2.669 nat，温度 0.0371。此批的低熵是诊断，完整物理比较使用原耗时评分。

| 主对照 checkpoint | SHA-256 |
|---|---|
| `expanded-ddqn.pth` | `712ffda5bbcf29223af24b61fee2d7f33f1ce67b6c6fca948fb1460389ee0717` |
| `expanded-sac.pth` | `19b414bb25a21dc695f1db8fecc8de3d6a803842ef59240a4770a16eb61570fb` |

## 核对长期价值排序

将最终两个 critic 的同图分数与八组已验证的完整共享前缀物理分叉对照。六组来自 P2-O 归档，两组来自本轮。DDQN 只排对 2/8，SAC 最小软 Q 排对 3/8。四个已知能缩短完整任务的旧 Search→Relay 分叉，两种 critic 都给出相反方向。新的 30 架动态 Search 配对真实差 64.1 秒，最终 DDQN 的 Q 差按奖励单位换算为 −1.70 秒，最终 SAC 为 +3.82 秒。两组符号和幅度属于已入回放的诊断。分叉使用固定但各自不同的后续策略，最优 Q 与 SAC 的软目标也各有不同，结果用来检验估值是否保留已观测的长期效应。

完整对照保存在 `credit-contrasts-fit.json`。复算方法为

```bash
a=artifacts/joint_mature_rl_20261123
"$py" -B scripts/audit_mature_rl_credit.py --replay "$a/expanded-replay.json.gz" \
  --benchmark "$a/benchmark.json" \
  --paired-labels artifacts/joint_autoresearch_20261122/runs/models/p2-o-calibrated.json \
  --fits "$a/expanded-ddqn-fit.json" "$a/expanded-sac-fit.json" --out /tmp/credit-contrasts-fit.json
```
