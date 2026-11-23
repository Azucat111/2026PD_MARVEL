# 复现与预注册协议

工作基线 `94c13d4ec91a8206287834f10cf24e7e7eac0251`，新分支 `codex/joint-gppo-mature-rl-8h`。三套历史证据 SHA 已验证，Phase 2 `7b4144349a0b7849257e58e870e1b7e95faa3791` 和 Phase 3 `3d070512dd7701b79d1dc2f6c4258aa6440b1988` 均在当前祖先链。现有上层模型和冻结 MARVEL 采用原版本。环境为 CPU 四核配额和单线程确定性 PyTorch。

## 隔离与预算

- 训练第一组为四域 `24110601–24110604`。同一 case 分别运行冻结 J0 和首个混合图的有界探索。第二组为四域 `24110611–24110614`，用于持续有界探索。继承归档视为已见数据。
- 比较 Double DQN 和 Discrete SAC 在同一份历史回放及同一份扩展回放上的模型。两边独立训练 600 次更新，批量 32、seed 6177。主选型指标为全新的 30 静态 `24110701` 和 60 动态 `24110704` 安全成功后合计完整耗时。数值平局保留 Double DQN。四个已进入历史回放的旧种子，在评估中只用于回归配对。两个训练数据端点用于观察同一新开发场景上的变化。
- 冻结胜者模型哈希后，四域 `24110801–24110804` 验证各一次 R0、R1、P2-O 和胜者。正式验收种子保持封存。完整安全成功和四域 `1 - sum(C_model)/sum(C_R0)` 用于裁决。
- 首个新运行于 2026-11-23 14:22（UTC+8）启动，预算 480 分钟。自动队列至多四个物理任务，450 分钟后停止发新长任务，受绝对停止时间和逐臂 wall limit 约束。任何截断单独记录。

## 命令

按 `handoff/cloud/CLOUD_REPRODUCTION.md` 安装 CPU 依赖并进入项目目录。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XDG_CACHE_HOME=/tmp/marvel-cache MPLCONFIGDIR=/tmp/marvel-mpl
py=${MARVEL_AUDIT_PYTHON:-python}
out=/tmp/mature-rl-reproduction
mkdir -p "$out"
date -Is > "$out/start.txt"
for a in artifacts/joint_autoresearch_20261122 artifacts/joint_ppo_20261122 artifacts/joint_causal_20261122; do
  (cd "$a" && sha256sum -c --quiet SHA256SUMS) || exit
done
"$py" -B -m pytest -p no:cacheprovider -q platform_tests/test_mature_event_rl.py \
  platform_tests/test_joint_atomic.py platform_tests/test_joint_event_ppo.py platform_tests/test_joint_public_preference.py
"$py" -B scripts/build_mature_rl_replay.py --historical --out "$out/historical-replay.json.gz" --audit "$out/historical-audit.json"
for algo in ddqn sac; do
  "$py" -B scripts/train_mature_rl.py --replay "$out/historical-replay.json.gz" --algo "$algo" \
    --seed 6177 --updates 600 --out "$out/archive-$algo.pth" > "$out/archive-$algo.train.jsonl"
done
for case in 30_static_24110601 30_dynamic_24110602 60_static_24110603 60_dynamic_24110604; do
  "$py" -B scripts/run_joint_autoresearch.py --case "$case" --policy J0 --wall 4500 \
    --out "$out/j0-$case" > "$out/j0-$case.log" 2>&1 &
done
"$py" -B scripts/run_mature_rl_study.py --live "$out" --phase all > "$out/study.jsonl"
"$py" -B scripts/summarize_mature_rl.py --runs "$out" --csv "$out/experiments.csv" --out "$out/benchmark.json"
```

单场运行可以按 `study.jsonl` 的 `command` 逐条执行。路径可换到新的输出目录，物理时钟和随机种子取原值。

模型初始训练已用相同历史回放重跑。两套纯基线 checkpoint 均与首次训练逐字节一致。继承的 Phase 2 规约检查 157 项通过，2 项历史 fixture 不存在而跳过；同时校验了 8 组冻结模型 SHA、严格加载和动作掩码。

若收集队列释放物理进程，可提前执行已经固定的历史预算开发臂。该次实际执行命令保存于 `early-dev.jsonl`，与后续队列的同名案例合并。开发和独立验证目录均未加入训练回放。
