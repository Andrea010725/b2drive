#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback
import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


# ==========================================
# 调试工具：直接写文件，防止终端吞日志
# ==========================================
def log_to_file(msg):
    with open("agent_debug.log", "a") as f:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        f.write(f"[{timestamp}] {msg}\n")


log_to_file("============== Agent 文件被加载 ==============")


class AutopilotAgent(AutonomousAgent):
    def __init__(self, carla_host, carla_port, debug=False):
        """
        Leaderboard 使用3个参数调用：host, port, debug
        """
        log_to_file(f"__init__ 被调用! 参数: host={carla_host}, port={carla_port}, debug={debug}")

        # ======================================================
        # 关键修复：正确调用父类初始化
        # ======================================================
        # 父类会初始化 sensor_interface 等必要属性
        try:
            super().__init__(carla_host, carla_port, debug)
            log_to_file("父类初始化完成")
        except Exception as e:
            log_to_file(f"父类初始化异常（这是正常的，因为world可能还没准备好）: {e}")
            # 手动初始化父类的必要属性
            from leaderboard.autoagents.autonomous_agent import Track
            from leaderboard.envs.sensor_interface import SensorInterface
            self.track = Track.SENSORS  # 先设置为默认值
            self._global_plan = None
            self._global_plan_world_coord = None
            self.sensor_interface = SensorInterface()
            self.wallclock_t0 = None
            self.hero_actor = None

        # 保存连接参数
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.debug = debug

        # ======================================================
        # 覆盖父类的 track 设置
        # ======================================================
        # 父类默认设置 track = Track.SENSORS，我们需要改为 MAP
        from leaderboard.autoagents.autonomous_agent import Track
        self.track = Track.MAP  # 使用 Track.MAP 枚举而不是字符串 'MAP'

        log_to_file(f"Track 已覆盖为: {self.track} (类型: {type(self.track).__name__}, 是Track枚举: {isinstance(self.track, Track)})")

        self.autopilot_enabled = False
        self.step_count = 0

    def setup(self, path_to_conf_file):
        log_to_file(f"Setup 被调用. 配置文件: {path_to_conf_file}")
        # 保持 MAP track 设置，不要在 setup 中改变
        self.setup_complete = True
        log_to_file(f"Setup 完成, Track 保持为: {self.track}")

    def sensors(self):
        log_to_file("定义传感器 - MAP track 不需要复杂传感器配置")
        # MAP track 只需要基础传感器
        sensors = [
            {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'id': 'GPS'},
            {'type': 'sensor.speedometer', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'id': 'SPEED'}
        ]
        log_to_file(f"传感器配置: {sensors}")
        return sensors

    def run_step(self, input_data, timestamp):
        self.step_count += 1
        if self.step_count % 50 == 0:
            log_to_file(f"Run Step {self.step_count} (心跳)")

        try:
            # 1. 寻找 Hero 车辆
            if self.hero_actor is None:
                self.hero_actor = self._find_hero_actor()
                if self.hero_actor is None:
                    return carla.VehicleControl(brake=1.0)

            # 2. 激活 Autopilot
            if not self.autopilot_enabled:
                log_to_file("尝试激活 Traffic Manager...")
                tm_port = CarlaDataProvider.get_traffic_manager_port()

                # 如果无法从 DataProvider 获取，使用传入的端口计算
                if tm_port is None:
                    tm_port = int(self.carla_port) + 6000  # 默认偏移
                    log_to_file(f"使用推测的 TM 端口: {tm_port}")
                else:
                    log_to_file(f"使用 DataProvider TM 端口: {tm_port}")

                # 获取 TM
                client = CarlaDataProvider.get_client()
                if not client:
                    client = carla.Client(self.carla_host, int(self.carla_port))
                    client.set_timeout(10.0)

                tm = client.get_trafficmanager(tm_port)

                # 设置 Traffic Manager 参数，让车开得更积极
                tm.ignore_lights_percentage(self.hero_actor, 0)  # 遵守红绿灯
                tm.auto_lane_change(self.hero_actor, True)  # 允许自动变道
                tm.vehicle_percentage_speed_difference(self.hero_actor, -50)  # 比限速快50%（更激进）
                tm.distance_to_leading_vehicle(self.hero_actor, 1.5)  # 进一步减小跟车距离
                tm.ignore_vehicles_percentage(self.hero_actor, 0)  # 不忽略其他车辆（安全驾驶）

                # 设置为全局参数，影响所有AI车辆
                tm.set_global_distance_to_leading_vehicle(2.0)

                self.hero_actor.set_autopilot(True, tm_port)
                self.autopilot_enabled = True
                log_to_file(">>> Autopilot 激活成功（高速驾驶模式）<<<")

            # 3. 获取并返回控制
            current_control = self.hero_actor.get_control()
            control = carla.VehicleControl()
            control.throttle = current_control.throttle
            control.steer = current_control.steer
            control.brake = current_control.brake

            return control

        except Exception as e:
            log_to_file(f"CRITICAL ERROR IN RUN_STEP: {e}")
            log_to_file(traceback.format_exc())
            return carla.VehicleControl(brake=1.0)

    def _find_hero_actor(self):
        try:
            # 优先从 CarlaDataProvider 获取
            world = CarlaDataProvider.get_world()
            if not world:
                log_to_file("CarlaDataProvider.get_world() 是空的，尝试重连...")
                client = carla.Client(self.carla_host, int(self.carla_port))
                client.set_timeout(10.0)
                world = client.get_world()

            for actor in world.get_actors():
                if actor.attributes.get('role_name') == 'hero':
                    log_to_file(f"找到 Hero 车辆: {actor.id}")
                    return actor
            return None
        except Exception as e:
            log_to_file(f"查找 Hero 失败: {e}")
            return None

    def destroy(self):
        log_to_file("Agent Destroying...")
        if self.hero_actor and self.autopilot_enabled:
            try:
                self.hero_actor.set_autopilot(False)
            except:
                pass


def get_entry_point():
    return 'AutopilotAgent'