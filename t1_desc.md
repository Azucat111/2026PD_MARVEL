## 目标

在现有 `SimulationRuntime` 骨架中接入 MARVEL 原有 PolicyNet，让仿真平台能调用真实的图注意力策略生成动作，而不只是 `default_actions()`。

## 背景

- 代码库：`E:\2026PD_MARVEL\MARVEL-main (1)\MARVEL-main`
- 训练入口 `driver.py`，核心模块在 `utils/`；新仿真平台在 `utils/simulation_runtime.py`
- MARVEL 策略：`utils/model.py` 中的 `PolicyNet`，输入是 `NODE_PADDING_SIZE=360` 的图特征，输出动作 log-prob
- checkpoint 位于 `load_model/marvel/`，`parameter.py` 中 `LOAD_MODEL=True`

## 具体任务

1. 新建 `utils/policy_adapter.py`：`MARVELPolicyAdapter` 类
   - 接收 `SimulationRuntime` 的 `observations`（`Dict[robot_id, Dict]`）
   - 转换为 MARVEL `Agent.get_observation()` 期望的图输入（邻域、frontier、相对位置、NODE_PADDING_SIZE=360 padding）
   - 调用 `PolicyNet.forward()` 获得动作，转回 `List[Tuple[np.ndarray, float]]`（waypoint, heading）
   - 支持加载 `load_model/marvel/` checkpoint；机器人数超过原始 `N_AGENTS=4` 时自动处理
   - checkpoint 缺失时回退到 `default_actions()` 并记录警告

2. 在 `scripts/run_scenario.py` 增加 `--use-marvel-policy` 开关接入 adapter

3. 用 `configs/baseline_exploration.yaml`（4 机）跑一个完整 episode 验证：策略加载成功、动作格式正确、探索率随步数递增

## 完成标准

- `python scripts/run_scenario.py configs/baseline_exploration.yaml --use-marvel-policy` 能跑完 128 步无 crash
- 在 issue 上贴出运行日志摘要（步数、探索率、是否使用 policy 或回退）
