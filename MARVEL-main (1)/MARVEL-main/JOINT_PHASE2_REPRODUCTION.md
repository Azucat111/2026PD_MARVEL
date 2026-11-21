# Phase 2 复现

工作目录为 `MARVEL-main (1)/MARVEL-main`。实际用 Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 1.26.4。沿用已验证的 `/workspace/marvel-audit-venv`，新环境可由交接依赖重建。

```bash
cd '/workspace/2026PD_MARVEL/MARVEL-main (1)/MARVEL-main'
python -m venv /workspace/marvel-audit-venv
/workspace/marvel-audit-venv/bin/pip install -r handoff/cloud/requirements-joint-shadow.txt
/workspace/marvel-audit-venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
export XDG_CACHE_HOME=/tmp/marvel-cache
export MPLCONFIGDIR=/tmp/marvel-mpl
```

## 依赖、模型及测试

```bash
/workspace/marvel-audit-venv/bin/python -B scripts/verify_joint_shadow.py \
  --phase2 --out /tmp/joint-phase2-verification.json
```

该入口继承上一阶段 91 项集合，附加联合事务、Relay 相关测试和冻结模型核验。Phase 2 检查交接 manifest 并核验三个有意变动文件的原始 Git blob。仅通过 pytest 进程定位交接 checkpoint。旧历史 Relay 测试数据缺失时标记具体两个用例为 skip。

## 逐域快照、定向执行与四臂 smoke

```bash
for case in 30_static_1280021 30_dynamic_1280022 60_static_9164 60_dynamic_1981204; do
  /workspace/marvel-audit-venv/bin/python -B scripts/verify_joint_atomic.py \
    --suite all --case "$case" --steps 3 --out "/tmp/joint-phase2/$case"
done
```

`--suite replay` 只读四个 Phase 1 公开快照并验证两类候选。`--suite directed` 从固定自然 reset 重放至同一事件，经真实提交分别执行 Search 和 Relay，检查实际首步和后续物理 tick。`--suite sampled --case 30_static_1280021` 运行冻结 GPPO 真实分类分布的 Relay 路径诊断。`--suite smoke` 在四个臂各跑 3 秒。`--suite legacy-parity --case 30_static_1280021` 逐 tick 核对 Legacy 与归档、Shadow 开关。输出原始证据含 source commit、公开快照配对摘要、物理反馈、mask、receipt 和命令。

## 启用配置

沿用 `prepare_gppo_config`、基底 checkpoint、搜索场景种子、冻结 MARVEL 及经过 SHA 验证的 6D checkpoint。在 `task_scheduler` 中设置 `mode: joint_rule` 或 `mode: joint_gppo`。原 `mode: gppo` 仍走既有调度器。建议先用上述有界入口校验自己的场景。未来 H300 CLI 参数可用 `python -B scripts/joint_four_arm_eval.py --help` 检查。
