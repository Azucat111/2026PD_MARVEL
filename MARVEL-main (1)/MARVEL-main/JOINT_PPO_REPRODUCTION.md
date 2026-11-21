# Joint GPPO Phase 3 复现

在包含 Phase 2 提交 `7b4144349a0b7849257e58e870e1b7e95faa3791` 的项目目录 `MARVEL-main (1)/MARVEL-main` 运行。使用 Phase 2 已有的 CPU venv，另见 `JOINT_PHASE2_REPRODUCTION.md`。本次运行使用 Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 1.26.4。

```bash
export XDG_CACHE_HOME=/tmp/marvel-cache
export MPLCONFIGDIR=/tmp/marvel-mpl
py=/workspace/marvel-audit-venv/bin/python
run=/tmp/joint-ppo-repro
mkdir -p "$run"
"$py" -B scripts/verify_joint_shadow.py --phase2 --out "$run/base-tests.json"
"$py" -B -m pytest -q -p no:cacheprovider platform_tests/test_joint_event_ppo.py
```

## D1 技术诊断

```bash
"$py" -B scripts/train_joint_event_ppo.py collect --case 30_static_240311 \
  --policy-seed 3411 --segment-ticks 20 --wall-seconds 240 --out "$run/d1"
"$py" -B scripts/train_joint_event_ppo.py update --rollout "$run/d1" \
  --allow-segment --epochs 1 --out "$run/d1-update"
```

该片段的最后状态是下一合法决策，文件保留 bootstrap。此 checkpoint 用于诊断，不进入开发性能配对。

## 完整 H300 训练与开发配对

训练协议和四域 seed 列表在 `artifacts/joint_ppo_20261122/protocol.json`。示例为第一条完整轨迹：

```bash
"$py" -B scripts/train_joint_event_ppo.py collect --case 30_static_240311 \
  --policy-seed 3411 --temperature 3 --wall-seconds 3000 \
  --out "$run/30_static_240311"
```

按协议执行其他三域，60 架案例的单回合墙钟上限为 4200 秒。主更新仅传入完整且可训练的轨迹目录。每回合留存公开快照 zip、实际不可变张量与选择回执、时间日志和评分。训练及开发环境 seed 分开。

```bash
"$py" -B scripts/train_joint_event_ppo.py update --epochs 2 \
  --rollout "$run/30_static_240311" "$run/30_dynamic_240312" \
            "$run/60_static_240313" "$run/60_dynamic_240314" \
  --out "$run/main-update"
model="$run/main-update/checkpoint.pth"
sha=$(sha256sum "$model" | cut -d' ' -f1)
"$py" -B scripts/audit_joint_event_ppo.py \
  --rollout "$run/30_static_240311" "$run/30_dynamic_240312" \
            "$run/60_static_240313" "$run/60_dynamic_240314" \
  --checkpoint "$model" --sha256 "$sha" --out "$run/audit.json"
"$py" -B scripts/joint_four_arm_eval.py --uavs 30 --domain static --seed 1280021 \
  --arms R1 P1 --p1-checkpoint "$model" --p1-sha256 "$sha" --wall-seconds 3000 \
  --compare-frozen --out "$run/pair-30s"
"$py" -B scripts/summarize_joint_ppo_evaluation.py --arm-file "$run/pair-30s/R1.json" "$run/pair-30s/P1.json" \
  --protocol artifacts/joint_ppo_20261122/protocol.json --p1-sha256 "$sha" --out "$run/dev-table.json"
```

可单独指定 `--arms R0` 或 `--arms P0` 补充同开发 seed 的 Legacy 对照。汇总会验证物理输入一致、训练模型 SHA 和完整 H300 评分，只有配对双方都完整的窗口才进入比例。跳过、超时和已完成但任务失败各有记录。

可直接核验交付中已归档的模型和物理轨迹。已压缩的评测 JSON 可直接传给汇总器，无需解压。

```bash
(cd artifacts/joint_ppo_20261122 && sha256sum -c SHA256SUMS)
```

使用已训练 P1 和已归档的完整物理日志重算开发配对与公开掩码：

```bash
a=artifacts/joint_ppo_20261122
model="$a/main-update/checkpoint.pth"
sha=$(sha256sum "$model" | cut -d' ' -f1)
"$py" -B scripts/audit_joint_event_ppo.py \
  --rollout "$a/training/30_static_240311" "$a/training/30_dynamic_240312" \
            "$a/training/60_static_240313" "$a/training/60_dynamic_240314" \
  --checkpoint "$model" --sha256 "$sha" --out "$run/archived-public-audit.json"
"$py" -B scripts/summarize_joint_ppo_evaluation.py --arm-file "$a"/evaluation/*/*.json.gz \
  --protocol "$a/protocol.json" --p1-sha256 "$sha" --out "$run/archived-evaluation.json"
```
