# 冻结候选 P2-O

部署模型为冻结 6D Joint GPPO（J0）和实测配对训练的供体感知排序残差。下层 MARVEL 的冻结权重、合法候选、原子提交、物理和 Safety 继续由原组件执行。该模型通过独立实验 runner 接入。

| 资产 | SHA-256 |
|---|---|
| P2-O 校准排序 checkpoint | `52b3424b64f0736641e1110eabbfeb83e43427cc549bdf191ea25d128a682276` |
| J0 冻结 6D 底座 | `c419c175619fe1ea27bbf4da25ad43a7e4a8d6e74ec6bd6f7ec09d5cbc97c69d` |
| MARVEL 底层 PolicyNet | `2e992ff83c8b05a346e882be5279f3e3db8b04e6be7aad308dd2b6c0532d99b2` |

checkpoint 为 `artifacts/joint_autoresearch_20261122/runs/models/p2-o-calibrated.json`。六个完整 H300 配对来自四个可分叉决策状态。五组继续使用冻结 J0，一组继续使用冻结 P2-S。模型依赖合法的公开图及原 actor logit。最终先验系数为 0.15，核正则为 0.20，支持阈值 0.42，改选分数门槛 0.20。从公开分叉的完整回报拟合相对价值，无新 PPO 权重更新。

## 冻结依据

候选按两个开发场景的完整安全成功和合计 C 规则，在 2026-11-22 21:58:22（UTC+8）冻结。模型筛选和校准仅使用表中两个案例。

| 始终使用完整 H300 | 30 静态 `1280021` | 30 动态 `1280022` | 合计 |
|---|---:|---:|---:|
| R0 | 31.0 | 30.7 | 61.7 |
| R1 | 39.0 | 41.2 | 80.2 |
| J0 / P1 | 82.1 | 41.2 | 123.3 |
| P2-S | 39.0 | 41.2 | 80.2 |
| P2-O | 39.0 | 32.6 | 71.6 |

两个 P2-O 开发窗口均通过全窗口安全。实际原子回执确认静态改配一次 Search，动态改选 Relay UAV 13，取代冻结 J0 的 Search UAV 24。90 秒旧反事实用于确定假设，本轮用完整 H300 重新采集训练标签。

## 加载并复算

```bash
export XDG_CACHE_HOME=/tmp/marvel-cache MPLCONFIGDIR=/tmp/marvel-mpl
py=${MARVEL_AUDIT_PYTHON:-python}
a=artifacts/joint_autoresearch_20261122
model="$a/runs/models/p2-o-calibrated.json"
sha=$(sha256sum "$model" | cut -d' ' -f1)
(cd "$a" && sha256sum -c --quiet SHA256SUMS)
"$py" -B scripts/run_joint_autoresearch.py --case 30_dynamic_1280022 --policy learned \
  --checkpoint "$model" --sha256 "$sha" --out /tmp/p2-o-dynamic-replay
"$py" -B scripts/summarize_joint_autoresearch.py --runs "$a/runs" \
  --csv /tmp/autoresearch-recomputed.csv --json /tmp/autoresearch-recomputed.json \
  --selection /tmp/autoresearch-frozen.json
```
