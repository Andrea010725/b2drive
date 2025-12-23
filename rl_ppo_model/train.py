#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RL PPO训练脚本 - 使用B2D数据集进行行为克隆预训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append('/home/ajifang/b2drive')
sys.path.append('/home/ajifang/b2drive/rl_ppo_model')

# 导入数据加载器
from rl_ppo_model.dataset import create_dataloader


class BehaviorCloningTrainer:
    """
    行为克隆训练器 - 用于PPO预训练
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🖥️  使用设备: {self.device}")

        # 创建保存目录
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 创建网络
        self._build_model()

        # 创建优化器
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config['learning_rate']
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 训练统计
        self.train_losses = []
        self.epoch_metrics = []

    def _build_model(self):
        """构建策略网络"""
        print("🏗️  构建策略网络...")

        # 使用你的PPO网络架构
        # 输入：图像特征 + 状态向量
        # 输出：动作（throttle, steer, brake）

        # 这里需要根据你的实际网络结构调整
        # 简化版本：使用一个简单的CNN+MLP

        self.policy_network = SimplePolicyNetwork(
            image_channels=3,
            state_dim=6,
            action_dim=3,
            hidden_dims=self.config.get('hidden_dims', [512, 256])
        ).to(self.device)

        print(f"✅ 网络参数量: {self._count_parameters():,}")

    def _count_parameters(self):
        return sum(p.numel() for p in self.policy_network.parameters() if p.requires_grad)

    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.policy_network.train()

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            images = batch['image'].to(self.device)       # (B, 3, H, W)
            states = batch['state'].to(self.device)       # (B, 6)
            actions = batch['action'].to(self.device)     # (B, 3)

            # 前向传播
            predicted_actions = self.policy_network(images, states)

            # 计算损失
            loss = self.criterion(predicted_actions, actions)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            epoch_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/num_batches:.4f}'
            })

        avg_loss = epoch_loss / num_batches
        return avg_loss

    def train(self, train_loader, num_epochs):
        """完整训练流程"""
        print("="*70)
        print("🚀 开始训练")
        print("="*70)
        print(f"📊 训练配置:")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {train_loader.batch_size}")
        print(f"   - Learning rate: {self.config['learning_rate']}")
        print(f"   - 设备: {self.device}")
        print(f"   - 数据集大小: {len(train_loader.dataset)}")
        print("="*70)
        print()

        best_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            print(f"\n📅 Epoch {epoch}/{num_epochs}")
            print("-"*70)

            # 训练一个epoch
            avg_loss = self.train_epoch(train_loader, epoch)

            # 记录
            self.train_losses.append(avg_loss)

            # 打印统计
            print(f"\n📊 Epoch {epoch} 统计:")
            print(f"   - 平均损失: {avg_loss:.6f}")

            # 保存checkpoint
            if epoch % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(epoch, avg_loss)

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, avg_loss, is_best=True)
                print(f"   ✅ 保存最佳模型 (loss: {best_loss:.6f})")

            print()

        print("="*70)
        print("🎉 训练完成!")
        print(f"📊 最佳损失: {best_loss:.6f}")
        print("="*70)

    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }

        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)
        print(f"   💾 保存: {path}")


class SimplePolicyNetwork(nn.Module):
    """
    简化的策略网络
    输入：图像 + 状态向量
    输出：动作（throttle, steer, brake）
    """

    def __init__(self, image_channels=3, state_dim=6, action_dim=3, hidden_dims=[512, 256]):
        super().__init__()

        # 图像编码器（简单的CNN）
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # MLP层
        self.fc = nn.Sequential(
            nn.Linear(256 + state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Sigmoid()  # 输出到[0, 1]范围
        )

    def forward(self, image, state):
        # 编码图像
        img_features = self.image_encoder(image)
        img_features = img_features.view(img_features.size(0), -1)

        # 拼接图像特征和状态
        combined = torch.cat([img_features, state], dim=1)

        # 预测动作
        action = self.fc(combined)

        # 调整steer的范围到[-1, 1]
        throttle = action[:, 0:1]
        steer = action[:, 1:2] * 2 - 1  # [0,1] -> [-1,1]
        brake = action[:, 2:3]

        return torch.cat([throttle, steer, brake], dim=1)


def main():
    """主函数"""
    print("="*70)
    print("🎯 B2D PPO行为克隆训练")
    print("="*70)
    print()

    # 训练配置
    config = {
        # 数据配置
        'data_root': '/home/ajifang/b2drive/Bench2Drive-RL50GB/datasets--rethinklab--Bench2Drive/snapshots',
        'image_size': (224, 224),
        'batch_size': 32,
        'num_workers': 4,
        'max_clips': None,  # None=使用全部clips

        # 训练配置
        'num_epochs': 50,
        'learning_rate': 3e-4,
        'hidden_dims': [512, 256],

        # 保存配置
        'save_dir': '/home/ajifang/b2drive/rl_ppo_model/checkpoints',
        'save_interval': 5,
    }

    # 找到snapshot目录
    import glob
    snapshot_dirs = glob.glob(f"{config['data_root']}/*")
    if snapshot_dirs:
        config['data_root'] = snapshot_dirs[0]
        print(f"📂 数据目录: {config['data_root']}")
    else:
        print("❌ 找不到snapshot目录")
        exit(1)

    # 创建数据加载器
    print("\n🔄 创建数据加载器...")
    train_loader = create_dataloader(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        max_clips=config['max_clips']
    )

    print(f"✅ 数据加载器创建成功")
    print(f"   - 训练样本数: {len(train_loader.dataset)}")
    print(f"   - Batch数量: {len(train_loader)}")
    print()

    # 创建训练器
    print("🏗️  创建训练器...")
    trainer = BehaviorCloningTrainer(config)
    print()

    # 开始训练
    trainer.train(train_loader, config['num_epochs'])

    print("\n✅ 训练完成!")
    print(f"📂 模型保存在: {config['save_dir']}")


if __name__ == "__main__":
    main()
