# Joint GPPO 两小时自主科研迭代

## 起点与预算

开始于 2026-11-22 20:39:02（UTC+8），计划停止时间 22:39:02。在 CPU 四核配额内，每条物理运行使用确定性单线程 PyTorch。核实 Phase 2 提交 `7b4144349a0b7849257e58e870e1b7e95faa3791`、Phase 3 提交 `3d070512dd7701b79d1dc2f6c4258aa6440b1988` 与继承的 Phase 3.5 `7c294d0f616d84dc8a784080b9a0a87b5b36a7b3`。对 Phase 3 和 Phase 3.5 的两个资产 SHA-256 清单复核通过。

继承的直接因果证据表明，在 `30_static_1280021`，J0/P1 均为 82.1 秒，实际分配和 3000 tick 评分逐条一致。第 3 秒的一次同角色 Search 分配使轨迹从未及时到达的单 Relay 路径切换到可及时增援的路径，90 秒诊断中首次稳定完成为 39.0 秒。本轮先在原评分下重走完整反事实，并测试真实混合图上的两种合法 Relay 选择。

## 研究假设

在几次相同公开状态的分叉中，完整任务耗时差可以为稀疏高层决策提供动作条件监督。先从同角色 Search 对照学习重排，再检查包含实际 Relay 回报的联合模型是否更好。改进只安装于实验 runner 的高层选择器，保留原 MARVEL、联合合法图、原子提交、观测合同、Safety 和 H300 评分。

### 迭代一的训练与行为合同

冻结 J0 的打分 `z(s,a)`，对合法候选使用残差 `f(s,a)`。训练数据为共享前缀和冻结后续策略的单动作分叉，使用真实完整 H300 的两条计时 C。目标为

```text
y_i = clip((C_J0 - C_alt)/10, -8, 8) - 0.25(z_alt - z_J0)
min_f sum_i (f(s_i,a_alt) - f(s_i,a_J0) - y_i)^2 + 0.20 ||f||_K^2
```

核函数以平方距离比较公开状态和候选描述，输入包括合法任务和现有承诺数量、延迟公开通信分量、合法路径成本、候选涉及 UAV 的接触数，以及沿已有首步证书方向移动时的邻接与连通估计。估计仅用于排序。在线既不读取隐藏目标，也不按 case、seed 或稳定 ID 查表。配对要求初态/配置摘要、干预前评分和提交、分叉处快照/输入张量摘要及 mask 一致。安全与成功仍由原物理评估，训练使用原 C。

部署分数为 `0.25 z + clip(f, -8, 8)`。只有计划改选的两端都达到核支持阈值，且分数差达到预先设置的门槛，才提交改选；否则使用原 J0 贪心。角色内版本使每个角色的最大分数保留原排序，允许重新排序同角色配对。联合版本允许受完整跨角色配对监督的改选。所有合法动作保留在记录的概率分布中。这个训练是冻结策略后续条件下的离策略配对回归，采样记录不会进入 PPO。

## 证据与复现入口

含完整窗口的运行只能通过原 `joint_completion_metric` 重新计分。短窗口只作行为与状态诊断。运行台账由 `scripts/summarize_joint_autoresearch.py` 从实际日志和匹配的归档对照生成。代码保留每个实际提交、同行冻结 J0 动作、候选输入摘要、合法分数和概率。对独立场景的复核只在冻结后执行。

静态异常 `30_static_1280021` 与动态 `30_dynamic_1280022` 是模型开发对照。优先冻结两个场景均安全成功、动态场景相对 J0 不退化超过 5 秒且两场合计 C 最低的候选，平局保留角色内模型。预留 `30_static_240327` 用于独立迁移检查，训练和模型选型均不使用该 seed。原 60 架开发场景留作冻结后检查，正式验收种子不进入任何本轮运行。

## 第一批实际 H300 与第一次训练

在 20:40–20:59（UTC+8）完成首批四条物理运行。三条静态干预均完整安全成功，验证了同状态的长期结果确实取决于选择与时序。动态 J0 控制为 41.2 秒，和原 P1 的 3,000 条评分记录一致。

| 干预 | 实际配对改变 | Search 完成 | 首次通信 ≥0.9 | 完成时间 C | 相对 J0 |
|---|---|---:|---:|---:|---:|
| 第 1 秒较短成本 Relay | Search 100002／UAV 27 → Relay 200015／UAV 29 | 12.3 | 75.2 | 78.2 | +4.75% |
| 第 3 秒冻结 actor 最优合法 Relay | Search 100001／UAV 4 → Relay 200015／UAV 29 | 14.1 | 78.1 | 81.1 | +1.22% |
| 第 3 秒改选 Search | Search 100001／UAV 4 → Search 100000／UAV 14 | 10.1 | 36.0 | 39.0 | +52.50% |

因此第一次学习仅用同角色 Search 配对，预先保留原有跨角色最高分顺序。生成并冻结 P2-S，SHA-256 为 `2195937d27252785fc1049a511aff893e9f0a692d73cdddeeb33f371162df509`。在真实配对日志的同一输入上，第 3 秒排序从原 J0 的 UAV 4 翻到 UAV 14，改选后两项校准分数分别为 −6.378 与 −2.618。后续完整部署用于核实该排序是否经实际提交传导到任务。下一轮将区分场景差异和 Search 顺序造成的 Relay 条件变化，而不一律提升 Relay 优先级。

静态分叉也给出可证伪的负证据。两次提前 Relay 都把初始桥接交给 UAV 29，分别在第 9 和第 14 秒改派远锚，到第 24 秒仍只有一个 Relay 承诺。用这三组完整回报拟合的联合原型 `p2-j-static-only` 在已记录的静态和动态真实图上，仅在第 3 秒重现 P2-S 的同角色改选。该原型留作离线结果，未重复占用一个完整物理窗口。选择下一条跨角色实验时，在动态配对发现最快合法 Relay 与 J0 首选 Search 均使用 UAV 24。该图由真实 59 个 Search 和 27 个 Relay 配对组成，支持直接检查同一 UAV 的机会成本。

第一次部署的两个 H300 均成功且安全。P2-S 在静态例得到 39.0 秒，和实际单动作干预的全窗口评分及稳定提交相符。它在 6 次真实提交中有 1 次相对同图 J0 改选，且角色仍为 Search。第 3 秒冻结模型给予改选配对的温度 1 概率约为 0.21%，组合分布给予 64.28%，原首选配对降至 1.50%。贪心执行的实际原子回执为 slot 100000／UAV 14。第 4 秒采到的延迟公开位置中，UAV 4 与 14 相对 J0 分别相差 0.999 和 1.808 米。第 13.0 秒计分连通率首次分开。最后首次通信达标从 79.1 秒变为 36.0 秒。39.0 秒持平 R1，仍比 R0 的 31.0 秒慢 8 秒。

P2-S 在动态例的 7 次提交均与 J0 相同，完整计时维持 41.2 秒，最大动作支持度在采样事件上不超过约 0.104，低于部署门槛 0.42。这是一次未迁移的真实结果。动态最近成本早期 Relay 干预改变了 t=1 的提交并得到 38.0 秒，仍慢于 R0 的 30.7 秒。两轮数据表明，应把研究从单一场景的相似性泛化转向显式区分候选所在的资源机会成本。

## 第二轮物理分叉和供体特征

动态 t=1 的同一张真实合法图上，Relay UAV 24 的原成本为 16.477 秒，对它的独占会使冻结 Search 最高 logit 下降 5.446；UAV 13 的成本为 19.603 秒，扣除该供体不损失首选 Search 分数。第二轮预先冻结供体搜索损失 `g(u)=clip(max_{a in Search} z(a) - max_{a in Search,donor(a)!=u} z(a), 0, 8)`。第一次干预使用公开成本 `ETA+2g` 挑选合法 Relay，随后回到冻结 J0。反事实标签仍使用原 H300 真实 C，从未用 ETA 替代回报。第二个分叉在静态 P2-S 已成功纠正第 3 秒 Search 后，改用第 4 秒合法的最近成本 Relay，检验角色时机是否依赖已纠正的拓扑。两者均使用原合法 mask 与原子回执；干预计入确定性行为，实际执行概率为 1。供体特征型训练进一步把 `g/4` 传入核的动作描述，使同一需求的不同无人机可在学习中区分。常规模型不含该字段。

普通联合原型加入 38.0 秒的动态最近供体标签后仍只在旧静态第 3 秒改选。这一训练产生 `p2-j-nearest`，SHA-256 `3289b7528b4bbb3eb4db2cf1f7558a0c794ee92b01136ffb0353413771d03905`，未追加 H300。已记下它的实际公开图重放，避免用更多参数或训练轮次重复这个行为。

## 复现首轮与第二轮

在项目目录设置下列环境。独立物理运行可在四个终端中执行，每条命令的输出目录需要为空。已有归档可以直接用于训练与汇总，完整复跑保留原时钟。

```bash
export XDG_CACHE_HOME=/tmp/marvel-cache MPLCONFIGDIR=/tmp/marvel-mpl
py=/workspace/marvel-audit-venv/bin/python
out=/tmp/marvel-autoresearch
mkdir -p "$out/models"
run() { "$py" -B scripts/run_joint_autoresearch.py --out "$out/$1" "${@:2}"; }
fit() { "$py" -B scripts/fit_joint_public_preference.py "$@"; }
control=artifacts/joint_causal_20261122/runs/J0
run ct-search --case 30_static_1280021 --policy pair --step 30 --pair 100000 14
run ct-relay1 --case 30_static_1280021 --policy nearest_relay --step 10
run ct-relay3 --case 30_static_1280021 --policy relay --step 30
run j0-dynamic --case 30_dynamic_1280022 --policy J0
run ct-dynamic --case 30_dynamic_1280022 --policy nearest_relay --step 10
run j0-holdout --case 30_static_240327 --policy J0
fit --control "$control" --treatment "$out/ct-search" --variant same_role --out "$out/models/p2-s.json"
fit --control "$control" --treatment "$out/ct-search" "$out/ct-relay1" "$out/ct-relay3" \
  --variant all_roles --out "$out/models/p2-j-static-only.json"
fit --control "$control" --treatment "$out/ct-search" "$out/ct-relay1" "$out/ct-relay3" \
  --extra-pair "$out/j0-dynamic" "$out/ct-dynamic" --variant all_roles --out "$out/models/p2-j-nearest.json"
s=$(sha256sum "$out/models/p2-s.json" | cut -d' ' -f1)
run p2-s-static --case 30_static_1280021 --policy learned --checkpoint "$out/models/p2-s.json" --sha256 "$s"
run p2-s-dynamic --case 30_dynamic_1280022 --policy learned --checkpoint "$out/models/p2-s.json" --sha256 "$s"
run ct-after-search --case 30_static_1280021 --policy learned --checkpoint "$out/models/p2-s.json" --sha256 "$s" --step 40
run ct-donor-aware --case 30_dynamic_1280022 --policy opportunity_relay --step 10
for arm in R0 R1; do
  "$py" -B scripts/joint_four_arm_eval.py --uavs 30 --domain static --seed 240327 --arms "$arm" --out "$out/${arm,,}-holdout"
done
for name in p2-j-static-only p2-j-nearest; do
  sha=$(sha256sum "$out/models/$name.json" | cut -d' ' -f1)
  "$py" -B scripts/replay_joint_public_preference.py --checkpoint "$out/models/$name.json" --sha256 "$sha" \
    --trajectory "$out/ct-search" "$out/ct-relay1" "$out/ct-relay3" "$out/j0-dynamic" --out "$out/$name-offline.json"
done
```

## 第二轮结果与部署校准

供体机会成本分叉真实改选 UAV 13，于 29.6 秒达标并以 32.6 秒完成，较最近成本 UAV 24 分叉的 38.0 秒缩短 5.4 秒，较 J0 的 41.2 秒缩短 8.6 秒，三者均通过全窗口安全。两个分叉的非 Relay 完成均为 11.1 秒。本次较慢的名义 Relay 供体反而产生更快的完整通信结果，与公共资源机会成本假设一致。它仍比原 R0 的 30.7 秒慢 1.9 秒。

静态条件分叉保留 P2-S 第 3 秒纠正的 Search，再提前一秒派 UAV 22 执行 Relay。完整耗时由 39.0 秒增至 80.1 秒，通信首次达标由 36.0 秒推迟至 77.1 秒，非 Relay 完成时间均为 10.1 秒。该臂安全完成。这个负结果排除了把提前 Relay 当作统一规则的做法，并作为一个真实负回报配对保留。此配对的两端在分叉前、分叉状态、遮罩、输入摘要和其余时刻的继续策略上使用相同冻结 P2-S；其来源 checkpoint SHA 与五组 J0 继续策略配对分别存入训练数据。混合这些配对是离策略拟合，部署仍须同环境完整计分。

六组物理对照拟合供体特征模型后，原系数 0.25 在动态图仍偏向 UAV 24。该图中 J0 对 Relay 13 和 Relay 24 的 logit 分别为 −22.84、−8.19。用真实图比较了五个先验/正则组合，相同正则 0.20 下把先验系数调到 0.15 已将这两个候选的组合分数顺序翻到 UAV 13（−1.477 对 −1.536），并保留在已记录 P2-S 静态路径上的 Search 动作。继续降低正则的变体有额外施加第 1 秒静态 Relay 的情形，未追加物理运行。校准只使用两个开发案例。由此冻结要跑完整部署的 P2-O（`p2-o-calibrated.json`），SHA-256 `52b3424b64f0736641e1110eabbfeb83e43427cc549bdf191ea25d128a682276`。

P2-O 真实部署在这两个开发图上分别提交了 Search 100000／UAV 14 和 Relay 200003／UAV 13。候选图的合法边数分别是 37 和 86，掩码未改。一次改选的组合同图概率如下，均为诊断用归一化分布，实际策略是确定性贪心。

| 场景与时刻 | 原选择 → 新提交 | 冻结 J0 对新动作概率 | P2-O 对新动作概率 |
|---|---|---:|---:|
| 30 静态第 3 秒 | Search 100001／4 → Search 100000／14 | 0.2098% | 73.6938% |
| 30 动态第 1 秒 | Search 100003／24 → Relay 200003／13 | 2.7980e−9 | 8.3347% |

两个 P2-O 开发窗口完成，均安全成功。静态 39.0 秒复现角色内修正，动态 32.6 秒复现供体感知干预。两场合计 71.6 秒，相对角色内 P2-S 的 80.2 秒再减少 8.6 秒。2026-11-22 21:58:22（UTC+8）按预先冻结规则选择校准后的 P2-O，后续仅测迁移不再训练。先验校准、拟合和角色先验回放中的其它变体分别保留。

### 供体模型与冻结后物理运行的复现

```bash
fit --control "$control" --treatment "$out/ct-search" "$out/ct-relay1" "$out/ct-relay3" \
  --extra-pair "$out/j0-dynamic" "$out/ct-dynamic" \
  --extra-pair "$out/j0-dynamic" "$out/ct-donor-aware" \
  --extra-pair "$out/p2-s-static" "$out/ct-after-search" \
  --variant donor_aware_calibrated --out "$out/models/p2-o-calibrated.json"
o=$(sha256sum "$out/models/p2-o-calibrated.json" | cut -d' ' -f1)
run p2-o-static --case 30_static_1280021 --policy learned --checkpoint "$out/models/p2-o-calibrated.json" --sha256 "$o"
run p2-o-dynamic --case 30_dynamic_1280022 --policy learned --checkpoint "$out/models/p2-o-calibrated.json" --sha256 "$o"
run j0-60-static --case 60_static_9164 --policy J0
run j0-60-dynamic --case 60_dynamic_1981204 --policy J0
run p2-60-static --case 60_static_9164 --policy learned --checkpoint "$out/models/p2-o-calibrated.json" --sha256 "$o"
run p2-60-dynamic --case 60_dynamic_1981204 --policy learned --checkpoint "$out/models/p2-o-calibrated.json" --sha256 "$o"
run p2-holdout --case 30_static_240327 --policy learned --checkpoint "$out/models/p2-o-calibrated.json" --sha256 "$o"
```

P2-O 的两个开发窗口分别与所对应的静态 Search 分叉、动态供体分叉在全部 3,000 条评分记录和稳定 ID 提交上逐条相同。供体模型从原始配对重拟合得到逐字节相同的 checkpoint。第 21:58 后的模型选择文件与运行日志保存于证据目录。相关测试 67 项通过；继承 Phase 2 回归 157 项通过、2 项历史样例缺失而跳过，8 个冻结 checkpoint 通过 SHA-256 和严格 CPU 加载。
