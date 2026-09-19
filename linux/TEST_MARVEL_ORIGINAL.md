# MARVEL原始环境测试说明

## 目的
验证MARVEL checkpoint在原始环境中是否正常工作（预期探索率80%+）

## 环境准备

```bash
cd /path/to/linux
pip install -r requirements.txt
```

## 运行测试

```bash
python scripts/test_marvel_original.py
```

## 预期输出

```
============================================================
MARVEL Original Environment Test
============================================================
Checkpoint: load_model/marvel/checkpoint.pth
Agents: 4
Max steps: 128
Sensor range: 10m

Map size: (height, width)
Initial positions: [[x1, y1], [x2, y2], ...]
Device: cuda/cpu
Loading checkpoint from episode 43872
Checkpoint loaded successfully

Step   1: Exploration XX.XX%
Step  10: Exploration XX.XX%
...
Step 128: Exploration XX.XX%

============================================================
Test Results
============================================================
Final exploration rate: XX.XX%
Total steps: 128
Expected: >80% for working checkpoint

✓ Checkpoint appears to be working correctly
```

## 结果判断

- **探索率 > 50%**: Checkpoint正常工作，问题在SimulationRuntime适配层
- **探索率 < 20%**: Checkpoint可能有问题或环境差异太大

## 下一步

### 如果checkpoint正常工作
精准定位SimulationRuntime适配问题：
- 对比原始环境 vs policy_adapter观测构建
- 检查坐标系转换
- 验证地图格式一致性

### 如果checkpoint不工作
考虑重新训练或使用其他checkpoint
