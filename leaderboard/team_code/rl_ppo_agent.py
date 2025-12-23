"""
Bench2Drive RL PPO Agent Wrapper
将PPO RL Agent适配到Bench2Drive评估框架

作者：基于原始RL agent改编
日期：2024-12-02
"""

import os
import sys
import carla
import numpy as np
import yaml
from collections import deque

# 添加rl_ppo_model到Python路径
rl_ppo_path = os.path.join(os.path.dirname(__file__), '../../rl_ppo_model')
sys.path.insert(0, rl_ppo_path)

# 导入B2D的autonomous agent基类
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

print(f"🔧 [DEBUG] Python路径已添加: {rl_ppo_path}")


class RLPPOAgent(AutonomousAgent):
    """
    RL PPO Agent for Bench2Drive

    这个agent将原始的PPO算法适配到Bench2Drive评估框架
    """

    def setup(self, path_to_conf_file):
        """
        初始化agent（只调用一次）

        Args:
            path_to_conf_file: 配置文件路径（ppo_config.yaml）
        """
        print("=" * 80)
        print("🚀 初始化 RL PPO Agent")
        print("=" * 80)

        # 1. 加载配置文件
        print(f"📄 加载配置文件: {path_to_conf_file}")
        self.config = self._load_config(path_to_conf_file)

        # 2. 设置基本参数
        self.step_count = 0
        self.episode_count = 0
        self.debug = self.config.get('inference', {}).get('debug', False)

        # 3. 初始化状态缓存（用于时序建模）
        buffer_size = self.config.get('inference', {}).get('state_buffer_size', 4)
        self.state_buffer = deque(maxlen=buffer_size)

        # 4. 暂时使用简单控制（后续会替换为真实PPO）
        print("⚠️  [简化模式] 当前使用简单PID控制")
        print("   TODO: 集成真实的PPO网络")

        # 目标速度（km/h）
        self.target_speed = 30.0

        # PID参数
        self.speed_kp = 0.5
        self.speed_ki = 0.01
        self.speed_kd = 0.1
        self.speed_error_integral = 0.0
        self.last_speed_error = 0.0

        self.steer_kp = 1.0

        print("✅ RL PPO Agent 初始化完成")
        print("=" * 80)

    def _load_config(self, path_to_conf_file):
        """加载YAML配置文件"""
        if not os.path.exists(path_to_conf_file):
            print(f"⚠️  配置文件不存在: {path_to_conf_file}")
            print("   使用默认配置")
            return self._get_default_config()

        try:
            with open(path_to_conf_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功")
            return config
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            print("   使用默认配置")
            return self._get_default_config()

    def _get_default_config(self):
        """获取默认配置"""
        return {
            'policy_lr': 3.0e-5,
            'value_lr': 3.0e-5,
            'gamma': 0.9999,
            'lambda_': 0.999,
            'clip_ratio': 0.15,
            'entropy_regularization': 0.5,
            'inference': {
                'load_weights': False,
                'debug': False,
                'state_buffer_size': 4
            }
        }

    def sensors(self):
        """
        定义agent需要的传感器

        Returns:
            List[dict]: 传感器定义列表
        """
        sensors = [
            # RGB相机 - 前置
            {
                'type': 'sensor.camera.rgb',
                'id': 'rgb_front',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900,
                'fov': 100
            },

            # IMU传感器（惯性测量单元）
            {
                'type': 'sensor.other.imu',
                'id': 'imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            },

            # GNSS传感器（GPS的标准名称）
            {
                'type': 'sensor.other.gnss',
                'id': 'gps',
                'x': 0.0, 'y': 0.0, 'z': 0.0
            },

            # 速度计
            {
                'type': 'sensor.speedometer',
                'id': 'speed'
            }
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        每个时间步的控制决策（核心方法）

        Args:
            input_data: dict，包含所有传感器数据
                {
                    'rgb_front': (frame_number, numpy.array),
                    'imu': (frame_number, dict),
                    'gps': (frame_number, tuple),
                    'speed': (frame_number, float),
                    'command': (frame_number, int)  # 导航指令
                }
            timestamp: float，当前仿真时间

        Returns:
            carla.VehicleControl: 车辆控制指令
        """
        self.step_count += 1

        # 1. 提取传感器数据
        speed = self._extract_speed(input_data)
        command = self._extract_command(input_data)

        # 2. 简单的PID速度控制
        control = self._simple_control(speed, command)

        # 3. 调试输出
        if self.debug and self.step_count % 50 == 0:
            print(f"[Step {self.step_count}] "
                  f"speed={speed:.2f} km/h, "
                  f"cmd={command}, "
                  f"throttle={control.throttle:.2f}, "
                  f"steer={control.steer:.2f}")

        return control

    def _simple_control(self, current_speed, command):
        """
        简单的PID速度控制 + 基于命令的转向

        这是一个临时实现，之后会替换为PPO网络的输出

        Args:
            current_speed: 当前速度 (km/h)
            command: 导航指令

        Returns:
            carla.VehicleControl
        """
        # PID速度控制
        speed_error = self.target_speed - current_speed
        self.speed_error_integral += speed_error
        speed_error_derivative = speed_error - self.last_speed_error
        self.last_speed_error = speed_error

        # PID输出
        throttle_brake = (
            self.speed_kp * speed_error +
            self.speed_ki * self.speed_error_integral +
            self.speed_kd * speed_error_derivative
        )

        # 分离throttle和brake
        if throttle_brake >= 0:
            throttle = np.clip(throttle_brake / 100.0, 0.0, 0.75)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-throttle_brake / 100.0, 0.0, 1.0)

        # 基于命令的简单转向逻辑
        steer = 0.0
        if command == 1:  # LEFT
            steer = -0.3
        elif command == 2:  # RIGHT
            steer = 0.3
        elif command == 3:  # STRAIGHT
            steer = 0.0
        # command == 4: LANE_FOLLOW - 保持steer=0

        # 创建控制指令
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False
        )

        return control

    def _extract_speed(self, input_data):
        """
        提取速度数据

        Returns:
            float: speed in km/h
        """
        if 'speed' in input_data:
            frame_number, speed = input_data['speed']
            # speed可能是字典或者直接是float
            if isinstance(speed, dict):
                return float(speed.get('speed', 0.0))
            else:
                return float(speed)
        return 0.0

    def _extract_command(self, input_data):
        """
        提取导航指令

        Returns:
            int: command ID
                0: VOID
                1: LEFT
                2: RIGHT
                3: STRAIGHT
                4: LANE_FOLLOW
                5: CHANGE_LANE
        """
        # B2D会在input_data的顶层提供command
        if 'command' in input_data:
            # command可能是tuple (frame, value) 或者直接是value
            if isinstance(input_data['command'], tuple):
                _, command = input_data['command']
            else:
                command = input_data['command']
            return int(command)
        return 4  # 默认：LANE_FOLLOW

    def destroy(self):
        """
        清理资源（只调用一次）
        """
        print("=" * 80)
        print("🛑 销毁 RL PPO Agent")
        print(f"   总步数: {self.step_count}")
        print("=" * 80)

    # ========== 以下是未来要实现的PPO相关方法 ==========

    def _extract_rgb(self, input_data):
        """
        提取并预处理RGB图像

        TODO: 未来用于PPO网络输入
        """
        if 'rgb_front' in input_data:
            frame_number, rgb_image = input_data['rgb_front']
            # rgb_image shape: (H, W, 4) BGRA
            rgb_image = rgb_image[:, :, :3]  # 去掉alpha
            rgb_image = rgb_image[:, :, ::-1]  # BGR -> RGB

            # TODO: 调整大小到PPO网络需要的尺寸
            # import cv2
            # rgb_image = cv2.resize(rgb_image, (120, 90))

            return rgb_image
        return np.zeros((900, 1600, 3), dtype=np.uint8)

    def _build_state(self, rgb, speed, imu, gps, command):
        """
        构建PPO agent需要的状态

        TODO: 未来实现完整的状态构建
        """
        # 图像归一化
        image = rgb.astype(np.float32) / 255.0

        # 车辆状态
        vehicle = np.array([speed / 30.0, 0.0], dtype=np.float32)

        # 导航指令（one-hot）
        navigation = np.zeros(4, dtype=np.float32)
        if command in [1, 2, 3, 4]:
            navigation[command - 1] = 1.0

        state = {
            'image': image,
            'vehicle': vehicle,
            'navigation': navigation
        }

        return state

    def _action_to_control(self, action):
        """
        将PPO输出的action转换为CARLA控制

        TODO: 未来实现完整的动作转换
        """
        # action: [throttle_brake, steer]
        throttle_brake = float(action[0])
        steer = float(action[1])

        if throttle_brake >= 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        control = carla.VehicleControl(
            throttle=np.clip(throttle, 0.0, 1.0),
            steer=np.clip(steer, -1.0, 1.0),
            brake=np.clip(brake, 0.0, 1.0),
            hand_brake=False,
            reverse=False
        )

        return control


def get_entry_point():
    """
    B2D框架要求的入口函数
    返回Agent类名
    """
    return 'RLPPOAgent'
