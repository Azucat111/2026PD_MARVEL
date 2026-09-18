#!/usr/bin/env python3
"""
MARVEL多任务训练脚本

基于原MARVEL driver.py扩展，支持：
1. 多任务场景（探索+搜寻+中继）
2. 动态障碍
3. 30-60机规模
4. 配置驱动训练

用法：
    python train_multi_task.py --scenario configs/scenarios/urban_rescue_simple.yaml --num-episodes 10000
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.simulation_runtime import SimulationRuntime
from utils.scenario_config import load_and_validate_scenario
from utils.evaluator import Evaluator

# MARVEL原始组件
from utils.model import PolicyNet, QNet
from parameter import *


class MultiTaskTrainer:
    """多任务MARVEL训练器"""

    def __init__(self, scenario_config, output_dir, device='cuda'):
        self.config = scenario_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # 初始化网络
        self._init_networks()

        # 初始化仿真运行时
        self.runtime = SimulationRuntime(scenario_config)

        # 训练统计
        self.episode_rewards = []
        self.episode_metrics = []

    def _init_networks(self):
        """初始化策略和价值网络"""
        print("[Trainer] 初始化神经网络...")

        # 策略网络
        self.policy_net = PolicyNet(
            node_input_dim=NODE_INPUT_DIM,
            embedding_dim=EMBEDDING_DIM
        ).to(self.device)

        # 双Q网络
        self.q_net1 = QNet(
            node_input_dim=NODE_INPUT_DIM,
            embedding_dim=EMBEDDING_DIM
        ).to(self.device)

        self.q_net2 = QNet(
            node_input_dim=NODE_INPUT_DIM,
            embedding_dim=EMBEDDING_DIM
        ).to(self.device)

        # 优化器
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=LR
        )
        self.q_optimizer1 = torch.optim.Adam(
            self.q_net1.parameters(),
            lr=LR
        )
        self.q_optimizer2 = torch.optim.Adam(
            self.q_net2.parameters(),
            lr=LR
        )

        print(f"  ✓ 策略网络: {sum(p.numel() for p in self.policy_net.parameters())} 参数")
        print(f"  ✓ Q网络: {sum(p.numel() for p in self.q_net1.parameters())} 参数")

    def collect_episode(self):
        """收集一个episode的经验"""
        observations = self.runtime.reset()
        episode_reward = 0
        transitions = []

        max_steps = self.config['duration']['max_steps']

        for step in range(max_steps):
            # TODO: 从观测生成动作
            # 当前简化：使用随机动作
            actions = self._get_actions(observations)

            # 执行动作
            next_observations, info = self.runtime.step(actions)

            # 计算奖励（多任务奖励）
            rewards = self._compute_rewards(observations, actions, next_observations, info)

            # 保存transition
            transitions.append({
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'next_observations': next_observations,
                'done': info.get('terminated', False) or info.get('truncated', False)
            })

            episode_reward += sum(rewards.values())
            observations = next_observations

            if info.get('terminated', False) or info.get('truncated', False):
                break

        return transitions, episode_reward, step + 1

    def _get_actions(self, observations):
        """从观测生成动作（待实现MARVEL策略）"""
        # TODO: 实现MARVEL策略推理
        # 当前返回随机动作
        num_robots = len(self.runtime.robots)
        actions = []
        for i in range(num_robots):
            # 随机目标位置
            target_pos = np.random.uniform([0, 0], [150, 150])
            target_heading = np.random.uniform(0, 360)
            actions.append((target_pos, target_heading))
        return actions

    def _compute_rewards(self, obs, actions, next_obs, info):
        """计算多任务奖励"""
        rewards = {}

        # 任务奖励（从info中提取）
        for event in info.get('task_events', []):
            if event['type'] == 'target_found':
                rewards[f"target_found_{event['target']['x']}"] = 20.0

        # 探索奖励（简化）
        rewards['exploration'] = 1.0

        # 碰撞惩罚
        if info.get('collisions'):
            rewards['collision'] = -50.0 * len(info['collisions'])

        # 通信连通性奖励
        comm_topology = info.get('comm_topology')
        if comm_topology is not None:
            # 简化：全连通给奖励
            is_connected = np.all(comm_topology.sum(axis=1) > 0)
            rewards['connectivity'] = 0.5 if is_connected else -5.0

        return rewards

    def update_networks(self, batch):
        """更新网络参数（待实现SAC更新）"""
        # TODO: 实现离散SAC更新逻辑
        # 参考原MARVEL的training.py
        pass

    def train(self, num_episodes, save_interval=100):
        """训练主循环"""
        print("=" * 72)
        print(f"  开始训练: {self.config['scenario']['name']}")
        print(f"  Episode数: {num_episodes}")
        print(f"  输出目录: {self.output_dir}")
        print("=" * 72)
        print("")

        for episode in range(num_episodes):
            start_time = time.time()

            # 收集经验
            transitions, episode_reward, steps = self.collect_episode()

            # 更新网络（待实现）
            # self.update_networks(transitions)

            # 记录统计
            self.episode_rewards.append(episode_reward)
            elapsed = time.time() - start_time

            # 打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                print(f"Episode {episode:5d} | Steps: {steps:3d} | Reward: {episode_reward:7.2f} | Avg: {avg_reward:7.2f} | Time: {elapsed:.2f}s")

            # 定期保存
            if (episode + 1) % save_interval == 0:
                self.save_checkpoint(episode + 1)

        print("\n训练完成！")
        self.save_checkpoint('final')

    def save_checkpoint(self, episode):
        """保存checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_ep{episode}.pth"

        torch.save({
            'episode': episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'q1_state_dict': self.q_net1.state_dict(),
            'q2_state_dict': self.q_net2.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q_optimizer1.state_dict(),
            'q2_optimizer': self.q_optimizer2.state_dict(),
            'episode_rewards': self.episode_rewards,
            'config': self.config
        }, checkpoint_path)

        print(f"  ✓ Checkpoint保存: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='MARVEL多任务训练')
    parser.add_argument('--scenario', required=True, help='场景配置文件')
    parser.add_argument('--num-episodes', type=int, default=10000, help='训练episode数')
    parser.add_argument('--output-dir', default='checkpoints/multi_task', help='输出目录')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save-interval', type=int, default=100, help='保存间隔')

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # 加载场景配置
    print(f"加载场景配置: {args.scenario}")
    config = load_and_validate_scenario(Path(args.scenario))
    config['scenario']['random_seed'] = args.seed

    # 创建训练器
    trainer = MultiTaskTrainer(
        scenario_config=config,
        output_dir=args.output_dir,
        device=args.device
    )

    # 开始训练
    trainer.train(
        num_episodes=args.num_episodes,
        save_interval=args.save_interval
    )


if __name__ == '__main__':
    main()
