# Joint GPPO Phase 3 训练记录

## 已验证的最小闭环

Phase 2 的 157 项回归重新通过，另有 2 项缺少历史输入而跳过。8 个交接模型通过 SHA-256 校验及加载，新增 PPO 测试 5 项通过。本次 CPU 资源为四核配额，未检测到 GPU。

D1 使用 30 架静态 case 240311 和 policy seed 3411，从冻结 6D 推理 checkpoint `c419c175619fe1ea27bbf4da25ad43a7e4a8d6e74ec6bd6f7ec09d5cbc97c69d` 开始，Adam 从空状态初始化。第 10 tick 有一个实际提交的混合角色选择，含 54 个合法配对。第 20 tick 的新真实决策提供冻结 critic bootstrap。20 个物理 tick 内碰撞为 0，片段没有 H300 完成时间。

一次 PPO 更新实际改变权重，梯度范数 0.050508。sampled KL 为 0.000852，mean exact / max-state exact KL 均为 0.000684。D1 未发生回滚，新增单元测试在已有 Adam 动量上注入越界并验证参数和优化器逐值恢复。D1 checkpoint SHA-256 为 `41aa185820f7a3f1520ad1d6fefe61c33aab9aaa5bf7582a1b2f9163ce6d77a9`。

从第 10 tick 的公开快照独立重建的合法掩码和稳定 ID 与实际存储匹配。相同公开状态下，D1 前后贪心都选择 UAV 29、Search slot 100003。训练温度 3 时该事件的 Relay 概率质量由 2.788% 变为 3.140%。这只是局部配对概率变化，D1 不进入主开发配对。

协议、训练数据、验证日志、模型及 SHA 清单归档在 `artifacts/joint_ppo_20261122`。
