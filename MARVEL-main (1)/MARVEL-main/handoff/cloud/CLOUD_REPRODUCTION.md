# 云端复现

从 Git 仓库根目录执行 `cd 'MARVEL-main (1)/MARVEL-main'`。本地已使用 Linux Python 3.11.8、PyTorch 2.5.1/CUDA 12.4 和 RTX 4060 Laptop 验证模型、短 smoke 及目标单元测试。云端应先执行 CPU 校验，实际 GPU 资源和驱动需在目标环境确认。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r handoff/cloud/requirements-audit.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -B handoff/cloud/smoke.py
python -B -m pytest -p no:cacheprovider -q \
  platform_tests/test_rule_scheduling_baseline.py \
  platform_tests/test_joint_completion_metric.py \
  platform_tests/test_gppo_search_scenario.py \
  platform_tests/test_extended_relay_arrival_priority.py
```

`smoke.py` 验证 8 文件哈希、按协议 strict load、CPU 前向和动作 mask，以及解析 80m 场景与地图。CPU 方案足以进行首阶段源码与离线状态研究。完整 GPU 环境可在单独 venv 将 PyTorch 安装索引换成 `https://download.pytorch.org/whl/cu124`。依照 [PyTorch 版本安装说明](https://pytorch.org/get-started/previous-versions/) 配置。上游 MARVEL SAC 则另需 `ray==2.44.0` 等依赖，`marvel.yml` 是已有环境导出，只作为上游参考。

GPU 上的小配对检查如下，每臂 4 架、4 个 mission steps。模型、配对初始化、配置和物理调用校验通过后写结果。这个短窗口用来确认执行连通。

```bash
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)'
python -B handoff/cloud/paired_eval.py --out /tmp/marvel-pair-smoke --uavs 4 --seed 20261121
```

已有 `scripts/run_scenario.py --prepare-only` 在本仓库若遇到 tracked 的 `logs/latest` 真实目录，会在同名 symlink 创建时失败。配置核验用上面的 `smoke.py`，场景短跑用 `paired_eval.py`。这两个入口以输出目录隔离新日志。

训练入口参数可以无采样确认。

```bash
python -B scripts/train_extended_gppo_batched.py --help
python -B scripts/evaluate_extended_gppo_joint_acceptance.py --help
```

若后续已批准少量新训练，`train_extended_gppo_batched.py` 需要 CUDA、通过 SHA 检查的初始模型、独立的训练 seed、适配的 reward/horizon 协议，以及不存在的输出目录。旧脚本通过 `benchmark_fair_search.CHECKPOINT` 使用历史绝对路径。可以在同进程先 `import benchmark_fair_search`、将 `CHECKPOINT` 绑定到项目内 `handoff/cloud/models/gppo_phase14_v9_ckpt80.pth`，再用 `runpy.run_path` 启动训练入口，`PYTHONPATH=scripts:.`。历史 optimizer 续训需要匹配原始模型、协议和样本合同。两个近期公开导出只含模型与安全的协议元数据。

以下命令展示单个 H300 配对的真实参数。**在选定独立 seed、冻结 30/60 × 静态/动态整套设计、批准计算预算后才运行。** 同套件所有配对统一使用源码、模型和环境哈希。为每个 case 指定未存在的输出目录。动态例加入 `--dynamic`，增长速率固定为当前 connected80 实验的 0.2 m/s。早期正式 Round 8 脚本和计划作为历史证据保留，见 [审计](RULE_BASELINE_AUDIT.md)。

```bash
# 先由新协议实际赋值。每个种子必须与训练、挑选、开发与已见验证隔离。
: "${FROZEN_EVAL_SEED:?Set the held-out seed from the frozen evaluation plan}"
python -B handoff/cloud/paired_eval.py \
  --out "/tmp/marvel-formal-${FROZEN_EVAL_SEED}" \
  --uavs 30 --seed "$FROZEN_EVAL_SEED" --dynamic \
  --full-window --wall-budget-seconds 1800
```

该命令复用当前合法性、Rule、production GPPO 和 joint scorer。单对文件给出两个物理窗口和完整原始计时。跨种子汇总仍需冻结失败处理、安全门槛、各域规模以及统计，方可适用整体 15% 判据。原冻结计划绑定原文件哈希、历史路径以及历史截止时间。将整套历史作业搬到新环境时，先做新合同；如果需要严格的原始二进制或完整轨迹，请从仓库管理员授权的私有存储传输，使用 [模型清单](checkpoint_manifest.json) 与 [轨迹索引](raw_trace_index.json) 中的 SHA-256 核对。
