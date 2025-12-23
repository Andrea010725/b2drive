#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IL Agent - 使用IL训练的模型
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import carla

# 添加项目路径
sys.path.append('/home/ajifang/b2drive')
sys.path.append('/home/ajifang/b2drive/rl_ppo_model')

from leaderboard.autoagents import autonomous_agent


class SimplePolicyNetwork(nn.Module):
    """
    简化的策略网络（和训练时使用的一致）
    """

    def __init__(self, image_channels=3, state_dim=6, action_dim=3, hidden_dims=[512, 256]):
        super().__init__()

        # 图像编码器
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
            nn.Sigmoid()
        )

    def forward(self, image, state):
        # 编码图像
        img_features = self.image_encoder(image)
        img_features = img_features.view(img_features.size(0), -1)

        # 拼接特征
        combined = torch.cat([img_features, state], dim=1)

        # 预测动作
        action = self.fc(combined)

        # 调整steer范围到[-1, 1]
        throttle = action[:, 0:1]
        steer = action[:, 1:2] * 2 - 1
        brake = action[:, 2:3]

        return torch.cat([throttle, steer, brake], dim=1)


class ILAgent(autonomous_agent.AutonomousAgent):
    """
    IL训练的驾驶agent
    使用训练好的神经网络进行决策
    """

    def setup(self, path_to_conf_file):
        """
        初始化agent
        Args:
            path_to_conf_file: 配置文件路径（这里不使用，使用硬编码的模型路径）
        """
        print("=" * 70)
        print("🤖 初始化IL Agent")
        print("=" * 70)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")

        # 模型路径
        self.model_path = '/home/ajifang/b2drive/rl_ppo_model/checkpoints/best_model.pth'

        # 创建网络
        print("🏗️  创建策略网络...")
        self.policy_network = SimplePolicyNetwork(
            image_channels=3,
            state_dim=6,
            action_dim=3,
            hidden_dims=[512, 256]
        ).to(self.device)

        # 加载权重
        print(f"📂 加载权重: {self.model_path}")
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.policy_network.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 权重加载成功 (Epoch {checkpoint.get('epoch', 'N/A')}, Loss {checkpoint.get('loss', 'N/A'):.6f})")
        else:
            print(f"⚠️  警告: 权重文件不存在，使用随机初始化")

        # 设置为评估模式
        self.policy_network.eval()

        # 图像预处理参数
        self.image_size = (224, 224)

        # 状态信息
        self.speed = 0.0
        self.position = None
        self.rotation = None

        # 目标点信息（从GPS获取）
        self.target_x = 0.0
        self.target_y = 0.0

        print("=" * 70)
        print("✅ IL Agent初始化完成")
        print("=" * 70)
        print()

    def sensors(self):
        """
        定义agent需要的传感器
        Returns:
            传感器配置列表
        """
        sensors = [
            # RGB前置相机
            {
                'type': 'sensor.camera.rgb',
                'id': 'rgb_front',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 100
            },
            # GNSS
            {
                'type': 'sensor.other.gnss',
                'id': 'gps',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            },
            # IMU
            {
                'type': 'sensor.other.imu',
                'id': 'imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            },
            # 速度计
            {
                'type': 'sensor.speedometer',
                'id': 'speed',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            }
        ]
        return sensors

    def run_step(self, input_data, timestamp):
        """
        执行一步决策
        Args:
            input_data: 传感器数据字典
            timestamp: 当前时间戳
        Returns:
            carla.VehicleControl: 车辆控制指令
        """
        # 1. 获取传感器数据
        rgb_front = input_data.get('rgb_front', None)
        gps_data = input_data.get('gps', None)
        imu_data = input_data.get('imu', None)
        speed_data = input_data.get('speed', None)

        if rgb_front is None:
            print("⚠️  警告: 没有RGB图像数据")
            return carla.VehicleControl()

        # 2. 处理图像
        image = rgb_front[1][:, :, :3]  # (H, W, 3)
        image = Image.fromarray(image)
        image = image.resize(self.image_size)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        image = image.to(self.device)

        # 3. 处理状态信息
        # 速度
        if speed_data is not None:
            self.speed = speed_data[1]['speed']

        # GPS位置
        if gps_data is not None:
            gps = gps_data[1]
            self.position = (gps[0], gps[1])  # (lat, lon)

        # IMU姿态
        if imu_data is not None:
            imu = imu_data[1]
            self.rotation = imu[-1]  # yaw

        # 获取目标点（从_command_planner）
        if hasattr(self, '_command_planner') and self._command_planner is not None:
            target_location = self._command_planner.target_location
            if target_location is not None:
                self.target_x = target_location.x
                self.target_y = target_location.y

        # 构造状态向量
        state = torch.tensor([
            self.speed / 10.0,  # 归一化速度
            0.0,  # x (暂时用0)
            0.0,  # y (暂时用0)
            self.rotation if self.rotation is not None else 0.0,
            self.target_x / 100.0,  # 归一化目标x
            self.target_y / 100.0,  # 归一化目标y
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        # 4. 模型推理
        with torch.no_grad():
            action = self.policy_network(image, state)
            action = action.cpu().numpy()[0]  # (3,)

        throttle = float(action[0])
        steer = float(action[1])
        brake = float(action[2])

        # 5. 创建控制指令
        control = carla.VehicleControl()
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.steer = np.clip(steer, -1.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def destroy(self):
        """
        清理资源
        """
        print("🧹 IL Agent清理完成")
        del self.policy_network


def get_entry_point():
    """
    B2D框架要求的入口函数
    """
    return 'ILAgent'
