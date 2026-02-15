from __future__ import annotations

"""
IL 数据采集脚本
目标：使用 CARLA API 的 autopilot + 少量规则触发，
在四个场景（Cones/Jaywalker/Trimma/Construction）下采集：
- vector_map (124 维)
- 未来 N 步的 ego 轨迹 (2N 维, 自车坐标系)
"""

import argparse
import math
import os
import re
import xml.etree.ElementTree as ET
import time
import json
import shutil
import sys
from types import SimpleNamespace
from typing import List, Dict, Any

import numpy as np

# NOTE: 采集脚本历史上依赖本机 CARLA egg 路径，这里保留但不影响 B2D 评测。
sys.path.append('/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
import carla

# ===== 采集相关依赖（可能在 B2D 评测环境里不存在）=====
# 这些模块只用于离线采集；评测时如果缺失，不应导致 import 失败。
_IL_COLLECTOR_AVAILABLE = True
_IL_COLLECTOR_IMPORT_ERROR = None
try:
    from collector.utils.carla_utils import (
        carla_sync_mode,
        get_ego_blueprint,
        set_spectator_follow_ego,
        SpectatorFollower,
        TrajectoryVisualizer,
    )
    from eva_monitor import EvaMonitor
    from collector.scenarios.scenarios import (
        ConesScenario,
        JaywalkerScenario,
        TrimmaScenario,
        ConstructionLaneChangeScenario,
    )
    from collector.il.vector_map import build_vector_map, VECTOR_MAP_SIZE, _world_to_ego_coords
except Exception as e:
    # 评测时缺少这些模块是正常的，先记录下来，运行采集时再提示
    _IL_COLLECTOR_AVAILABLE = False
    _IL_COLLECTOR_IMPORT_ERROR = e
    EvaMonitor = None
    ConesScenario = None
    JaywalkerScenario = None
    TrimmaScenario = None
    ConstructionLaneChangeScenario = None
    build_vector_map = None
    VECTOR_MAP_SIZE = None
    _world_to_ego_coords = None

# ===== B2D Leaderboard 相关依赖 =====
# 评测时由 leaderboard_evaluator 导入本文件，会依赖这些类。
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


# ================================
# 传感器：碰撞检测
# ================================
class CollisionWatcher:
    def __init__(self, world: carla.World, ego: carla.Actor):
        self.world = world
        self.ego = ego
        self.has_collided = False
        self.sensor = None
        self._setup_sensor()

    def _setup_sensor(self):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event: carla.CollisionEvent):
        other = event.other_actor
        print(f"[Collision] ❌ EGO 与 {other.type_id} (id={other.id}) 发生碰撞")
        self.has_collided = True

    def destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


# ================================
# 多视角摄像头
# ================================
class MultiViewCameras:
    """
    前视/左视/右视 RGB 摄像头。
    - 使用 CARLA 的 sensor.camera.rgb
    - 每帧缓存最新图像，记录时再保存
    """
    def __init__(self, world: carla.World, ego: carla.Actor,
                 width: int = 256, height: int = 256, fov: float = 90.0):
        self.world = world
        self.ego = ego
        self.width = width
        self.height = height
        self.fov = fov

        self.front_cam = None
        self.left_cam = None
        self.right_cam = None

        self.latest_front = None
        self.latest_left = None
        self.latest_right = None

        self._setup()

    def _make_camera(self, transform: carla.Transform):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))
        bp.set_attribute("fov", str(self.fov))
        cam = self.world.spawn_actor(bp, transform, attach_to=self.ego)
        return cam

    def _setup(self):
        # 前视
        front_tf = carla.Transform(carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        self.front_cam = self._make_camera(front_tf)
        self.front_cam.listen(lambda img: setattr(self, "latest_front", img))

        # 左视（yaw=-90）
        left_tf = carla.Transform(carla.Location(x=0.8, z=2.2), carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0))
        self.left_cam = self._make_camera(left_tf)
        self.left_cam.listen(lambda img: setattr(self, "latest_left", img))

        # 右视（yaw=+90）
        right_tf = carla.Transform(carla.Location(x=0.8, z=2.2), carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))
        self.right_cam = self._make_camera(right_tf)
        self.right_cam.listen(lambda img: setattr(self, "latest_right", img))

    def save_images(self, out_dir: str, frame_id: int) -> dict:
        """
        保存当前帧图像，返回相对路径。
        """
        os.makedirs(out_dir, exist_ok=True)
        paths = {}
        if self.latest_front is not None:
            p = os.path.join(out_dir, f"front_{frame_id:06d}.png")
            self.latest_front.save_to_disk(p)
            paths["front"] = p
        if self.latest_left is not None:
            p = os.path.join(out_dir, f"left_{frame_id:06d}.png")
            self.latest_left.save_to_disk(p)
            paths["left"] = p
        if self.latest_right is not None:
            p = os.path.join(out_dir, f"right_{frame_id:06d}.png")
            self.latest_right.save_to_disk(p)
            paths["right"] = p
        return paths

    def destroy(self):
        for cam in [self.front_cam, self.left_cam, self.right_cam]:
            if cam and cam.is_alive:
                cam.stop()
                cam.destroy()


# ================================
# 简单规则：强制变道触发器
# ================================
class LaneChangeTrigger:
    def __init__(self, tm: carla.TrafficManager, ego: carla.Actor, trigger_dist: float = 15.0):
        self.tm = tm
        self.ego = ego
        self.trigger_dist = trigger_dist
        self.triggered = False

    def try_force_lane_change(self, obstacles: List[carla.Actor]):
        """
        若前方近距离存在障碍物，则尝试强制变道（优先右侧，否则左侧）。
        """
        if self.triggered:
            return

        ego_tf = self.ego.get_transform()
        fwd = ego_tf.get_forward_vector()
        ego_loc = ego_tf.location
        amap = self.ego.get_world().get_map()
        ego_wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            return

        # 找到最近的“前方障碍”
        nearest = None
        nearest_d = 1e9
        for obs in obstacles:
            if obs is None:
                continue
            try:
                loc = obs.get_location()
            except Exception:
                continue
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            # 前方判定：与 ego 前向向量点积 > 0
            if dx * fwd.x + dy * fwd.y <= 0.0:
                continue
            d = math.hypot(dx, dy)
            if d < nearest_d:
                nearest_d = d
                nearest = obs

        if nearest is None or nearest_d > self.trigger_dist:
            return

        # 触发变道：优先右侧车道
        right_wp = ego_wp.get_right_lane()
        left_wp = ego_wp.get_left_lane()

        if right_wp and right_wp.lane_type == carla.LaneType.Driving:
            print("[Rule] 前方障碍近距离，触发向右变道")
            try:
                self.tm.force_lane_change(self.ego, True)
                self.triggered = True
            except Exception:
                pass
        elif left_wp and left_wp.lane_type == carla.LaneType.Driving:
            print("[Rule] 前方障碍近距离，触发向左变道")
            try:
                self.tm.force_lane_change(self.ego, False)
                self.triggered = True
            except Exception:
                pass


# ================================
# 场景构造器
# ================================
def make_scenario(name: str, world: carla.World, amap: carla.Map, tm_port: int):
    # 采集场景依赖于 collector.* 模块，若缺失则无法运行采集
    if not _IL_COLLECTOR_AVAILABLE:
        raise RuntimeError(f"采集模块缺失，无法创建场景: {_IL_COLLECTOR_IMPORT_ERROR}")
    cfg = SimpleNamespace(tm_port=tm_port, enable_traffic_flow=True)
    if name == "cones":
        return ConesScenario(world, amap, cfg)
    if name == "jaywalker":
        return JaywalkerScenario(world, amap, cfg)
    if name == "trimma":
        return TrimmaScenario(world, amap, cfg)
    if name == "construction":
        return ConstructionLaneChangeScenario(world, amap, cfg)
    raise ValueError(f"未知场景: {name}")


# ================================
# 采集主逻辑
# ================================
def collect_one_episode(
    client: carla.Client,
    scenario_name: str,
    output_dir: str,
    max_steps: int,
    future_steps: int,
    tm_port: int,
    min_speed_to_keep: float = 0.2,
    save_images: bool = True,
    image_width: int = 256,
    image_height: int = 256,
):
    # 采集逻辑依赖 collector.* 与 eva_monitor
    if not _IL_COLLECTOR_AVAILABLE:
        raise RuntimeError(f"采集模块缺失，无法采集: {_IL_COLLECTOR_IMPORT_ERROR}")
    world = client.get_world()
    amap = world.get_map()

    # ---- 创建场景 ----
    scenario = make_scenario(scenario_name, world, amap, tm_port)
    construction_before_ids = None
    if isinstance(scenario, ConstructionLaneChangeScenario):
        # 记录施工场景前已有的 actor，便于回合结束后清理
        try:
            construction_before_ids = set([a.id for a in world.get_actors()])
        except Exception:
            construction_before_ids = None
    if not scenario.setup():
        print(f"[{scenario_name}] ❌ 场景初始化失败，跳过")
        scenario.cleanup()
        return

    # ---- 生成 Ego ----
    ego_bp = get_ego_blueprint(world)
    spawn_tf = scenario.get_spawn_transform()
    if spawn_tf is None:
        print(f"[{scenario_name}] ❌ 场景未提供 spawn transform")
        scenario.cleanup()
        return

    ego = world.try_spawn_actor(ego_bp, spawn_tf)
    if ego is None:
        spawn_tf.location.z += 0.5
        ego = world.try_spawn_actor(ego_bp, spawn_tf)
    if ego is None:
        print(f"[{scenario_name}] ❌ Ego 生成失败")
        scenario.cleanup()
        return

    # 设置 spectator 视角（默认追尾）
    follower = None
    traj_viz = None
    if getattr(client, "_enable_spectator", False):
        try:
            set_spectator_follow_ego(world, ego, mode="chase")
            follower = SpectatorFollower(world, mode="chase")
        except Exception as e:
            print(f"[{scenario_name}] ⚠️ 设置 spectator 失败: {e}")
    if getattr(client, "_enable_traj_viz", False):
        traj_viz = TrajectoryVisualizer(world, max_points=300)

    # ---- Traffic Manager ----
    tm = client.get_trafficmanager(tm_port)
    try:
        tm.set_synchronous_mode(world.get_settings().synchronous_mode)
    except Exception:
        pass

    # ---- Ego autopilot ----
    ego.set_autopilot(True, tm_port)

    # ---- 碰撞监测 ----
    collision = CollisionWatcher(world, ego)

    # ---- EVA Monitor ----
    eva = None
    if getattr(client, "_enable_eva", False):
        try:
            eva = EvaMonitor()
            eva.attach(world, ego)
        except Exception as e:
            print(f"[{scenario_name}] ⚠️ EVA Monitor 初始化失败: {e}")

    # ---- 摄像头 ----
    cameras = None
    if save_images:
        cameras = MultiViewCameras(world, ego, width=image_width, height=image_height, fov=90.0)

    # ---- 规则触发器 ----
    lane_trigger = LaneChangeTrigger(tm, ego, trigger_dist=15.0)
    trimma_forced_change = False
    trimma_force_ticks = 0
    trimma_force_left = True
    trimma_tried_other = False
    trimma_wait_steps = 0
    trimma_slow_front_count = 0

    construction_force_ticks = 0

    # ---- 采集缓冲区 ----
    pending = []  # 每帧保存: {'vector_map': ..., 'ego_matrix': ...}
    positions = []  # 每帧 ego 位置 (world)
    records: List[Dict[str, Any]] = []
    frame_id = 0
    prev_speed = None
    episode_success = True

    # 每个 episode 建一个独立目录（失败就删掉）
    run_tag = f"{scenario_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, "runs", run_tag)
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # ---- 主循环 ----
    for step in range(max_steps):
        world.tick()

        if collision.has_collided:
            print(f"[{scenario_name}] 碰撞，停止本回合采集")
            episode_success = False
            break

        # Jaywalker 需要触发 + 逐帧更新
        if isinstance(scenario, JaywalkerScenario):
            scenario.check_and_trigger(ego.get_location())
            scenario.tick_update()

        # Cones / Construction：简单强制变道规则
        if isinstance(scenario, ConesScenario):
            # ✅ Cones 专用触发：距离第一个锥桶约 8m 时强制变道
            if scenario.first_cone_transform is not None:
                ego_loc = ego.get_location()
                cone_loc = scenario.first_cone_transform.location
                dist_to_cone = math.hypot(ego_loc.x - cone_loc.x, ego_loc.y - cone_loc.y)
                if dist_to_cone <= 25.0:
                    # 使用 TM 强制变道：先看右侧，再看左侧
                    amap = world.get_map()
                    ego_wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if ego_wp is not None:
                        right_wp = ego_wp.get_right_lane()
                        left_wp = ego_wp.get_left_lane()
                        if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                            print("[Cones-Rule] 距离锥桶 <= 8m，触发向右变道")
                            try:
                                tm.force_lane_change(ego, True)
                            except Exception:
                                pass
                        elif left_wp and left_wp.lane_type == carla.LaneType.Driving:
                            print("[Cones-Rule] 距离锥桶 <= 8m，触发向左变道")
                            try:
                                tm.force_lane_change(ego, False)
                            except Exception:
                                pass
        elif isinstance(scenario, TrimmaScenario):
            # ✅ Trimma 专用：速度过慢时强制变道（避免一直跟车）
            ego_loc = ego.get_location()
            ego_speed = ego.get_velocity().length()
            trimma_wait_steps += 1

            # ---- 1) 找到“同车道最近前车” ----
            front_dist = None
            front_speed = None
            amap = world.get_map()
            ego_wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if ego_wp is not None:
                fwd = ego_wp.transform.get_forward_vector()
                nearest = None
                nearest_d = 1e9
                for a in world.get_actors().filter("vehicle.*"):
                    if a.id == ego.id:
                        continue
                    try:
                        loc = a.get_location()
                    except Exception:
                        continue
                    # 只看前方
                    dx = loc.x - ego_loc.x
                    dy = loc.y - ego_loc.y
                    if dx * fwd.x + dy * fwd.y <= 0.0:
                        continue
                    d = math.hypot(dx, dy)
                    if d < nearest_d:
                        wp_a = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                        if wp_a is None:
                            continue
                        # 同一条路同一车道
                        if wp_a.road_id == ego_wp.road_id and wp_a.lane_id == ego_wp.lane_id:
                            nearest_d = d
                            nearest = a
                if nearest is not None:
                    front_dist = nearest_d
                    front_speed = nearest.get_velocity().length()

            # ---- 2) 触发规则（加入“持续慢车”判断）----
            # 先让车辆走一小段再触发，避免一上来就变道
            if trimma_wait_steps < 30:
                should_force = False
            else:
                slow_front = (front_speed is not None) and (front_speed < 3.0)
                near_front = (front_dist is not None) and (front_dist < 30.0)
                slow_ego = ego_speed < 3.0

                # 连续多帧前车慢才触发
                if slow_front and near_front:
                    trimma_slow_front_count += 1
                else:
                    trimma_slow_front_count = 0

                should_force = (trimma_slow_front_count >= 15) or (slow_ego and front_dist is not None and front_dist < 18.0)

            # 额外：保持前车距离不要太近（通过 TM 跟车距离调节）
            try:
                tm.distance_to_leading_vehicle(ego, 8.0)
            except Exception:
                pass

            if (not trimma_forced_change) and should_force:
                if ego_wp is not None:
                    # 优先右侧（你需求里第一车道希望向右）
                    left_wp = ego_wp.get_left_lane()
                    right_wp = ego_wp.get_right_lane()

                    def lane_clear(target_wp, radius=10.0) -> bool:
                        if target_wp is None:
                            return False
                        # 简单判断：目标车道半径内是否有车辆
                        for a in world.get_actors().filter("vehicle.*"):
                            if a.id == ego.id:
                                continue
                            try:
                                loc = a.get_location()
                            except Exception:
                                continue
                            if loc.distance(ego_loc) > radius:
                                continue
                            wp_a = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                            if wp_a is None:
                                continue
                            if wp_a.lane_id == target_wp.lane_id and wp_a.road_id == target_wp.road_id:
                                return False
                        return True

                    # 先判断右侧是否可变道
                    if right_wp and right_wp.lane_type == carla.LaneType.Driving and lane_clear(right_wp, 10.0):
                        print("[Trimma-Rule] 速度过慢且前车近，触发向右变道")
                        trimma_forced_change = True
                        trimma_force_left = False
                        trimma_force_ticks = 25
                        try:
                            tm.auto_lane_change(ego, True)
                            tm.force_lane_change(ego, True)  # True = right
                        except Exception:
                            pass
                    elif left_wp and left_wp.lane_type == carla.LaneType.Driving and lane_clear(left_wp, 10.0):
                        print("[Trimma-Rule] 速度过慢且前车近，触发向左变道")
                        trimma_forced_change = True
                        trimma_force_left = True
                        trimma_force_ticks = 25
                        try:
                            tm.auto_lane_change(ego, True)
                            tm.force_lane_change(ego, False)  # False = left
                        except Exception:
                            pass

            # 触发后连续几帧重复 force（提高成功率）
            if trimma_force_ticks > 0:
                trimma_force_ticks -= 1
                try:
                    # 按已选择方向重复 force（True=右，False=左）
                    tm.force_lane_change(ego, not trimma_force_left)
                except Exception:
                    pass

            # 如果变道失败（车道未变化），尝试另一侧一次
            if trimma_forced_change and (not trimma_tried_other):
                try:
                    cur_wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if cur_wp is not None and ego_wp is not None and cur_wp.lane_id == ego_wp.lane_id:
                        # 仍在原车道，尝试另一侧
                        trimma_tried_other = True
                        if trimma_force_left:
                            alt = ego_wp.get_right_lane()
                            if alt and alt.lane_type == carla.LaneType.Driving:
                                print("[Trimma-Rule] 左侧变道失败，尝试向右")
                                tm.force_lane_change(ego, True)
                        else:
                            alt = ego_wp.get_left_lane()
                            if alt and alt.lane_type == carla.LaneType.Driving:
                                print("[Trimma-Rule] 右侧变道失败，尝试向左")
                                tm.force_lane_change(ego, False)
                except Exception:
                    pass
        elif isinstance(scenario, ConstructionLaneChangeScenario):
            # ✅ Construction 专用：提前变道 + 连续force
            ego_loc = ego.get_location()
            amap = world.get_map()
            ego_wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

            # 触发距离：离施工点 25m 内就准备变道
            dist_to_construction = None
            if getattr(scenario, "construction_location", None) is not None:
                dist_to_construction = scenario.construction_location.distance(ego_loc)

            should_force = (dist_to_construction is not None and dist_to_construction < 25.0)

            # 触发一次后持续force
            if should_force and construction_force_ticks == 0:
                construction_force_ticks = 30

                # 优先使用场景标记的相邻车道方向
                if ego_wp is not None:
                    left_wp = ego_wp.get_left_lane()
                    right_wp = ego_wp.get_right_lane()

                    if getattr(scenario, "adjacent_lane_id", None) == "left" and left_wp:
                        print("[Construction-Rule] 进入施工区前，触发向左变道")
                        try:
                            tm.auto_lane_change(ego, True)
                            tm.force_lane_change(ego, False)
                        except Exception:
                            pass
                    elif getattr(scenario, "adjacent_lane_id", None) == "right" and right_wp:
                        print("[Construction-Rule] 进入施工区前，触发向右变道")
                        try:
                            tm.auto_lane_change(ego, True)
                            tm.force_lane_change(ego, True)
                        except Exception:
                            pass
                    else:
                        # 兜底：右侧优先
                        if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                            print("[Construction-Rule] 进入施工区前，触发向右变道")
                            try:
                                tm.auto_lane_change(ego, True)
                                tm.force_lane_change(ego, True)
                            except Exception:
                                pass
                        elif left_wp and left_wp.lane_type == carla.LaneType.Driving:
                            print("[Construction-Rule] 进入施工区前，触发向左变道")
                            try:
                                tm.auto_lane_change(ego, True)
                                tm.force_lane_change(ego, False)
                            except Exception:
                                pass

            # 连续force（提高成功率）
            if construction_force_ticks > 0:
                construction_force_ticks -= 1
                try:
                    # 按场景推荐方向重复force
                    if getattr(scenario, "adjacent_lane_id", None) == "left":
                        tm.force_lane_change(ego, False)
                    elif getattr(scenario, "adjacent_lane_id", None) == "right":
                        tm.force_lane_change(ego, True)
                except Exception:
                    pass

            # ✅ Construction：同车道不生成前车（移除同车道前方近距离车辆）
            if ego_wp is not None:
                for a in world.get_actors().filter("vehicle.*"):
                    if a.id == ego.id:
                        continue
                    try:
                        loc = a.get_location()
                    except Exception:
                        continue
                    # 仅处理同车道、前方 60m 内车辆
                    wp_a = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if wp_a is None:
                        continue
                    if wp_a.road_id == ego_wp.road_id and wp_a.lane_id == ego_wp.lane_id:
                        # 前方判定
                        fwd = ego_wp.transform.get_forward_vector()
                        dx = loc.x - ego_loc.x
                        dy = loc.y - ego_loc.y
                        if dx * fwd.x + dy * fwd.y > 0.0 and math.hypot(dx, dy) < 60.0:
                            try:
                                a.destroy()
                            except Exception:
                                pass

        # spectator 持续跟随（每帧更新）
        if follower is not None:
            try:
                follower.update(ego)
            except Exception:
                pass
        if traj_viz is not None:
            try:
                traj_viz.add_point(ego.get_location())
                traj_viz.draw()
            except Exception:
                pass

        # EVA 面板刷新
        if eva is not None:
            try:
                data = eva.tick()
                eva.render(data)
            except Exception:
                pass

        # 当前 ego 状态
        ego_tf = ego.get_transform()
        ego_matrix = np.array(ego_tf.get_matrix())
        speed = ego.get_velocity().length()
        yaw_rad = math.radians(ego_tf.rotation.yaw)
        yaw_rate = ego.get_angular_velocity().z  # rad/s
        if prev_speed is None:
            acc = 0.0
        else:
            acc = (speed - prev_speed) / max(1e-6, world.get_settings().fixed_delta_seconds or 0.05)
        prev_speed = speed

        # 若 ego 几乎不动，可跳过（避免僵住的数据）
        if speed < min_speed_to_keep:
            continue

        # 采集 vector_map
        vector_map = build_vector_map(ego, world, amap)
        if vector_map.shape[0] != VECTOR_MAP_SIZE:
            print(f"[{scenario_name}] ⚠️ vector_map 维度异常: {vector_map.shape}")
            continue

        pending.append({
            "vector_map": vector_map,
            "ego_matrix": ego_matrix,
            "ego_features": np.array([speed, acc, yaw_rad, yaw_rate], dtype=np.float32),
        })
        positions.append(ego_tf.location)

        # 当累计了足够的未来轨迹后，生成标签
        if len(positions) > future_steps:
            base = pending.pop(0)
            base_idx = len(positions) - future_steps - 1

            # 未来 N 步位置 → 转到当前 ego 坐标系
            future_xy = []
            for k in range(1, future_steps + 1):
                future_loc = positions[base_idx + k]
                future_world = np.array([future_loc.x, future_loc.y, future_loc.z], dtype=np.float32)
                future_ego = _world_to_ego_coords(future_world, base["ego_matrix"])[0]
                future_xy.append(future_ego[:2])

            traj = np.array(future_xy, dtype=np.float32).reshape(-1)  # [2N]

            # traffic light（前方 15m 内）
            tl_state = -1  # -1=无信号灯
            try:
                tl = ego.get_traffic_light()
                if tl is not None:
                    dist_tl = tl.get_transform().location.distance(ego_tf.location)
                    # 判断是否在前方
                    fwd = ego_tf.get_forward_vector()
                    to_tl = tl.get_transform().location - ego_tf.location
                    ahead = (to_tl.x * fwd.x + to_tl.y * fwd.y) > 0.0
                    if dist_tl <= 15.0 and ahead:
                        tl_state = int(tl.state)  # CARLA enum
            except Exception:
                pass

            # 保存图像
            img_paths = {}
            if cameras is not None:
                img_paths = cameras.save_images(images_dir, frame_id)

            records.append({
                "vector_map": base["vector_map"],
                "trajectory": traj,
                "ego_features": base["ego_features"],
                "controls": np.array([ego.get_control().steer,
                                      ego.get_control().throttle,
                                      ego.get_control().brake], dtype=np.float32),
                "traffic_light_state": tl_state,
                "images": img_paths,
            })
            frame_id += 1

    # ---- 保存数据 ----
    if (not episode_success) or (len(records) == 0):
        print(f"[{scenario_name}] ⚠️ 本回合未完成或无有效样本，丢弃数据")
        # 删除本回合临时目录（包括图片）
        try:
            if os.path.isdir(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass
    else:
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"{scenario_name}_il_{ts}.npz")

        vecs = np.stack([r["vector_map"] for r in records], axis=0)
        trajs = np.stack([r["trajectory"] for r in records], axis=0)

        ego_feats = np.stack([r["ego_features"] for r in records], axis=0)
        controls = np.stack([r["controls"] for r in records], axis=0)
        tl_states = np.array([r["traffic_light_state"] for r in records], dtype=np.int32)
        img_meta = [r["images"] for r in records]

        np.savez_compressed(
            out_path,
            vector_map=vecs,
            trajectory=trajs,
            ego_features=ego_feats,
            controls=controls,
            traffic_light_state=tl_states,
            images=np.array(img_meta, dtype=object),
        )
        print(f"[{scenario_name}] ✅ 保存 {len(records)} 条样本到 {out_path}")

    # ---- 清理 ----
    if cameras is not None:
        cameras.destroy()
    collision.destroy()
    if ego.is_alive:
        ego.set_autopilot(False)
        ego.destroy()
    scenario.cleanup()

    # 施工场景额外清理：销毁本回合新增的 actor（锥桶/障碍/施工区车辆等）
    if construction_before_ids is not None:
        try:
            after_actors = world.get_actors()
            for a in after_actors:
                if a.id in construction_before_ids:
                    continue
                # 避免重复销毁 ego / 传感器
                try:
                    a.destroy()
                except Exception:
                    pass
            print(f"[{scenario_name}] ✅ 施工场景新增 actor 已清理")
        except Exception:
            pass

    # 返回是否成功
    return bool(episode_success and len(records) > 0)


# ================================
# B2D 评测适配 Agent
# ================================
def get_entry_point():
    """
    Leaderboard 会通过这个函数拿到 Agent 类名
    """
    return "ILPlannerAgent"


class ILPlannerAgent(AutonomousAgent):
    """
    将原有 IL Planner 的规则逻辑适配到 B2D Leaderboard 的 Agent 接口。
    - 不修改 B2D 代码
    - 只在本文件内做连接/适配
    """

    # B2D 路由里的 scenario type -> 任务类别（可按需调整）
    _SCENARIO_GROUPS = {
        "cones": {
            "ParkedObstacle",
            "ParkedObstacleTwoWays",
            "HazardAtSideLane",
            "HazardAtSideLaneTwoWays",
            "Accident",
            "AccidentTwoWays",
            "VehicleOpensDoorTwoWays",
        },
        "construction": {
            "ConstructionObstacle",
            "ConstructionObstacleTwoWays",
        },
        "jaywalker": {
            "PedestrianCrossing",
            "ParkingCrossingPedestrian",
            "DynamicObjectCrossing",
            "VehicleTurningRoutePedestrian",
        },
        "trimma": {
            "HardBreakRoute",
            "StaticCutIn",
            "ParkingCutIn",
            "HighwayCutIn",
            "SequentialLaneChange",
            "MergerIntoSlowTraffic",
            "MergerIntoSlowTrafficV2",
            "EnterActorFlow",
            "InterurbanActorFlow",
            "InterurbanAdvancedActorFlow",
        },
    }

    def setup(self, path_to_conf_file):
        """
        B2D 会在每条路线开始前调用 setup
        """
        self.track = Track.SENSORS

        # 基础句柄
        self._client = CarlaDataProvider.get_client()
        self._world = CarlaDataProvider.get_world()
        self._tm_port = CarlaDataProvider.get_traffic_manager_port()
        self._tm = self._client.get_trafficmanager(self._tm_port) if self._client else None

        # 重新获取一次 hero（更稳妥）
        if self.hero_actor is None:
            self.get_hero()

        # 开启 autopilot，便于 TM 接管基础驾驶
        if self.hero_actor is not None:
            try:
                self.hero_actor.set_autopilot(True, self._tm_port)
            except Exception:
                pass

        # ===== 从 agent_config 解析当前 route 信息 =====
        self._route_id = self._parse_route_id(path_to_conf_file)
        self._route_info = self._load_route_info(self._route_id)
        self._scenario_type = self._route_info.get("type")
        self._scenario_name = self._route_info.get("name")
        self._trigger_point = self._route_info.get("trigger_point")
        self._scenario_group = self._scenario_type_to_group(self._scenario_type)

        # ===== 触发窗口参数 =====
        self._trigger_dist = 35.0  # 与 trigger_point 的触发距离（米）
        self._active_window = 120  # 触发后保持 active 的 tick 数
        self._active_ticks = 0

        # ===== 各场景状态 =====
        self._cones_triggered = False
        self._construction_force_ticks = 0
        self._construction_force_right = None

        self._trimma_wait_steps = 0
        self._trimma_slow_front_count = 0
        self._trimma_forced_change = False
        self._trimma_force_ticks = 0
        self._trimma_force_left = True
        self._trimma_tried_other = False
        self._trimma_origin_lane_id = None
        self._trimma_origin_road_id = None

        self._jaywalker_brake_ticks = 0

        # 打印一次关键信息，便于确认分类是否正确
        print(f"[ILPlannerAgent] route_id={self._route_id}, scenario_type={self._scenario_type}, group={self._scenario_group}", flush=True)

    def sensors(self):
        """
        本 Agent 不依赖传感器输入，直接从世界读取状态即可
        """
        return []

    # ----------------------------
    # 规则入口
    # ----------------------------
    def run_step(self, input_data, timestamp):
        """
        每个 tick 调用一次
        """
        # 兜底补齐 world/client/tm（避免偶发空指针）
        if self._client is None:
            self._client = CarlaDataProvider.get_client()
        if self._world is None:
            self._world = CarlaDataProvider.get_world()
        if self._tm is None and self._client is not None:
            try:
                self._tm = self._client.get_trafficmanager(self._tm_port)
            except Exception:
                pass

        if self.hero_actor is None or not self.hero_actor.is_alive:
            self.get_hero()
            if self.hero_actor is None:
                return carla.VehicleControl()

        ego = self.hero_actor
        ego_tf = ego.get_transform()
        control = ego.get_control()

        if self._world is None:
            return control

        # 更新 trigger 激活状态
        self._update_active(ego_tf)

        # 如果没有分类信息，直接返回 autopilot 控制
        if self._scenario_group is None:
            return control

        # 根据分类执行规则
        if self._scenario_group == "cones" and self._is_active():
            self._apply_cones_rule(ego_tf)
        elif self._scenario_group == "construction" and self._is_active():
            self._apply_construction_rule(ego_tf)
        elif self._scenario_group == "trimma" and self._is_active():
            self._apply_trimma_rule(ego_tf)
        elif self._scenario_group == "jaywalker" and self._is_active():
            control = self._apply_jaywalker_rule(ego_tf, control)

        return control

    # ----------------------------
    # 分类与触发
    # ----------------------------
    def _parse_route_id(self, agent_config: str):
        """
        从 agent_config 中解析 RouteScenario_XXXX
        """
        if not agent_config:
            return None
        # agent_config 会被 evaluator 拼接为：<user_config>+<save_name>
        save_name = agent_config.split("+")[-1]
        match = re.search(r"RouteScenario_(\d+)", save_name)
        return match.group(1) if match else None

    def _resolve_routes_file(self):
        """
        解析 routes 文件路径（优先环境变量 ROUTES）
        """
        candidates = []
        env_routes = os.environ.get("ROUTES")
        if env_routes:
            candidates.append(env_routes)
        candidates.extend([
            "leaderboard/data/bench2drive220.xml",
            "leaderboard/data/drivetransformer_bench2drive_dev10.xml",
        ])
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def _load_route_info(self, route_id: str):
        """
        从 routes.xml 中找到当前 route 的 scenario type + trigger_point
        """
        info = {}
        if not route_id:
            return info
        routes_file = self._resolve_routes_file()
        if not routes_file:
            return info
        try:
            tree = ET.parse(routes_file)
            root = tree.getroot()
            for route in root.findall("route"):
                if route.get("id") != route_id:
                    continue
                scenarios = route.find("scenarios")
                if scenarios is None:
                    return info
                scenario = scenarios.find("scenario")
                if scenario is None:
                    return info
                info["type"] = scenario.get("type")
                info["name"] = scenario.get("name")
                trigger = scenario.find("trigger_point")
                if trigger is not None:
                    info["trigger_point"] = carla.Transform(
                        carla.Location(
                            x=float(trigger.get("x")),
                            y=float(trigger.get("y")),
                            z=float(trigger.get("z")),
                        ),
                        carla.Rotation(yaw=float(trigger.get("yaw", 0.0))),
                    )
                return info
        except Exception as e:
            print(f"[ILPlannerAgent] 解析 routes 失败: {e}", flush=True)
            return info

        return info

    def _scenario_type_to_group(self, scenario_type: str):
        """
        将 scenario type 映射到 IL Planner 的四类规则
        """
        if not scenario_type:
            return None
        for group, types in self._SCENARIO_GROUPS.items():
            if scenario_type in types:
                return group
        return None

    def _update_active(self, ego_tf: carla.Transform):
        """
        更新 trigger 激活窗口
        """
        if self._trigger_point is None:
            # 无 trigger_point 时认为始终可触发（便于调试）
            self._active_ticks = self._active_window
            return

        ego_loc = ego_tf.location
        trg_loc = self._trigger_point.location
        dx = trg_loc.x - ego_loc.x
        dy = trg_loc.y - ego_loc.y
        dist = math.hypot(dx, dy)
        fwd = ego_tf.get_forward_vector()
        ahead = (dx * fwd.x + dy * fwd.y) > 0.0

        if dist <= self._trigger_dist and ahead:
            self._active_ticks = self._active_window
        elif self._active_ticks > 0:
            self._active_ticks -= 1

    def _is_active(self):
        return self._active_ticks > 0

    # ----------------------------
    # 规则实现（复用 il_planner 的核心逻辑）
    # ----------------------------
    def _apply_cones_rule(self, ego_tf: carla.Transform):
        if self._cones_triggered or self._tm is None:
            return
        amap = self._world.get_map()
        ego_wp = amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            return

        right_wp = ego_wp.get_right_lane()
        left_wp = ego_wp.get_left_lane()

        if right_wp and right_wp.lane_type == carla.LaneType.Driving:
            print("[ILPlannerAgent][Cones] 触发向右变道", flush=True)
            self._force_lane_change(go_right=True)
            self._cones_triggered = True
        elif left_wp and left_wp.lane_type == carla.LaneType.Driving:
            print("[ILPlannerAgent][Cones] 触发向左变道", flush=True)
            self._force_lane_change(go_right=False)
            self._cones_triggered = True

    def _apply_construction_rule(self, ego_tf: carla.Transform):
        if self._tm is None:
            return
        if self._construction_force_ticks == 0:
            amap = self._world.get_map()
            ego_wp = amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if ego_wp is None:
                return

            right_wp = ego_wp.get_right_lane()
            left_wp = ego_wp.get_left_lane()

            # 优先右侧，否则左侧
            if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                print("[ILPlannerAgent][Construction] 触发向右变道", flush=True)
                self._construction_force_right = True
                self._construction_force_ticks = 30
                self._force_lane_change(go_right=True)
            elif left_wp and left_wp.lane_type == carla.LaneType.Driving:
                print("[ILPlannerAgent][Construction] 触发向左变道", flush=True)
                self._construction_force_right = False
                self._construction_force_ticks = 30
                self._force_lane_change(go_right=False)

        # 连续 force，提高成功率
        if self._construction_force_ticks > 0 and self._construction_force_right is not None:
            self._construction_force_ticks -= 1
            self._force_lane_change(go_right=self._construction_force_right)

    def _apply_trimma_rule(self, ego_tf: carla.Transform):
        if self._tm is None:
            return
        ego_loc = ego_tf.location
        ego_speed = self.hero_actor.get_velocity().length()
        self._trimma_wait_steps += 1

        # ---- 找到同车道最近前车 ----
        front_dist = None
        front_speed = None
        amap = self._world.get_map()
        ego_wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp is not None:
            fwd = ego_wp.transform.get_forward_vector()
            nearest = None
            nearest_d = 1e9
            for a in self._world.get_actors().filter("vehicle.*"):
                if a.id == self.hero_actor.id:
                    continue
                try:
                    loc = a.get_location()
                except Exception:
                    continue
                dx = loc.x - ego_loc.x
                dy = loc.y - ego_loc.y
                if dx * fwd.x + dy * fwd.y <= 0.0:
                    continue
                d = math.hypot(dx, dy)
                if d < nearest_d:
                    wp_a = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if wp_a is None:
                        continue
                    if wp_a.road_id == ego_wp.road_id and wp_a.lane_id == ego_wp.lane_id:
                        nearest_d = d
                        nearest = a
            if nearest is not None:
                front_dist = nearest_d
                front_speed = nearest.get_velocity().length()

        # ---- 判断是否触发变道 ----
        if self._trimma_wait_steps < 30:
            should_force = False
        else:
            slow_front = (front_speed is not None) and (front_speed < 3.0)
            near_front = (front_dist is not None) and (front_dist < 30.0)
            slow_ego = ego_speed < 3.0

            if slow_front and near_front:
                self._trimma_slow_front_count += 1
            else:
                self._trimma_slow_front_count = 0

            should_force = (self._trimma_slow_front_count >= 15) or (slow_ego and front_dist is not None and front_dist < 18.0)

        try:
            self._tm.distance_to_leading_vehicle(self.hero_actor, 8.0)
        except Exception:
            pass

        # ---- 执行变道 ----
        if (not self._trimma_forced_change) and should_force and ego_wp is not None:
            left_wp = ego_wp.get_left_lane()
            right_wp = ego_wp.get_right_lane()

            def lane_clear(target_wp, radius=10.0) -> bool:
                if target_wp is None:
                    return False
                for a in self._world.get_actors().filter("vehicle.*"):
                    if a.id == self.hero_actor.id:
                        continue
                    try:
                        loc = a.get_location()
                    except Exception:
                        continue
                    if loc.distance(ego_loc) > radius:
                        continue
                    wp_a = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if wp_a is None:
                        continue
                    if wp_a.lane_id == target_wp.lane_id and wp_a.road_id == target_wp.road_id:
                        return False
                return True

            if right_wp and right_wp.lane_type == carla.LaneType.Driving and lane_clear(right_wp, 10.0):
                print("[ILPlannerAgent][Trimma] 速度过慢，触发向右变道", flush=True)
                self._trimma_forced_change = True
                self._trimma_force_left = False
                self._trimma_force_ticks = 25
                self._trimma_origin_lane_id = ego_wp.lane_id
                self._trimma_origin_road_id = ego_wp.road_id
                self._force_lane_change(go_right=True)
            elif left_wp and left_wp.lane_type == carla.LaneType.Driving and lane_clear(left_wp, 10.0):
                print("[ILPlannerAgent][Trimma] 速度过慢，触发向左变道", flush=True)
                self._trimma_forced_change = True
                self._trimma_force_left = True
                self._trimma_force_ticks = 25
                self._trimma_origin_lane_id = ego_wp.lane_id
                self._trimma_origin_road_id = ego_wp.road_id
                self._force_lane_change(go_right=False)

        if self._trimma_force_ticks > 0:
            self._trimma_force_ticks -= 1
            self._force_lane_change(go_right=not self._trimma_force_left)

        # 变道失败时，尝试另一侧一次
        if self._trimma_forced_change and (not self._trimma_tried_other):
            try:
                cur_wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if cur_wp is not None and self._trimma_origin_lane_id is not None:
                    if cur_wp.lane_id == self._trimma_origin_lane_id and cur_wp.road_id == self._trimma_origin_road_id:
                        self._trimma_tried_other = True
                        if self._trimma_force_left:
                            alt = ego_wp.get_right_lane() if ego_wp is not None else None
                            if alt and alt.lane_type == carla.LaneType.Driving:
                                print("[ILPlannerAgent][Trimma] 左侧失败，尝试向右", flush=True)
                                self._force_lane_change(go_right=True)
                        else:
                            alt = ego_wp.get_left_lane() if ego_wp is not None else None
                            if alt and alt.lane_type == carla.LaneType.Driving:
                                print("[ILPlannerAgent][Trimma] 右侧失败，尝试向左", flush=True)
                                self._force_lane_change(go_right=False)
            except Exception:
                pass

    def _apply_jaywalker_rule(self, ego_tf: carla.Transform, control: carla.VehicleControl):
        """
        简单行人刹停规则：前方近距离出现行人则短暂制动
        """
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        has_ped = False
        for w in self._world.get_actors().filter("walker.pedestrian.*"):
            try:
                loc = w.get_location()
            except Exception:
                continue
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            if dx * fwd.x + dy * fwd.y <= 0.0:
                continue
            dist = math.hypot(dx, dy)
            if dist < 12.0:
                has_ped = True
                break

        if has_ped:
            self._jaywalker_brake_ticks = 20  # 持续刹车一小段时间

        if self._jaywalker_brake_ticks > 0:
            self._jaywalker_brake_ticks -= 1
            control.throttle = 0.0
            control.brake = max(control.brake, 0.6)

        return control

    def _force_lane_change(self, go_right: bool):
        if self._tm is None or self.hero_actor is None:
            return
        try:
            self._tm.auto_lane_change(self.hero_actor, True)
        except Exception:
            pass
        try:
            self._tm.force_lane_change(self.hero_actor, go_right)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--sync", action="store_true", default=True)
    parser.add_argument("--fixed-dt", type=float, default=0.05)
    parser.add_argument("--scenario", type=str, default="cones",
                        choices=["cones", "jaywalker", "trimma", "construction", "all"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--future-steps", type=int, default=12)
    parser.add_argument("--out", type=str, default="/home/ajifang/il_data_collect/il_data")
    parser.add_argument("--spectator", action="store_true", default=False,
                        help="开启 spectator 追尾视角（需要 CARLA 图形窗口）")
    parser.add_argument("--draw-traj", action="store_true", default=False,
                        help="实时绘制Ego ground-truth轨迹")
    parser.add_argument("--no-eva", action="store_true", default=False,
                        help="关闭 EVA 监控面板（默认开启）")
    parser.add_argument("--no-images", action="store_true", default=False,
                        help="不保存前/左/右视图图像")
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    # 标记是否开启 spectator
    client._enable_spectator = bool(args.spectator)
    client._enable_traj_viz = bool(args.draw_traj)
    client._enable_eva = not bool(args.no_eva)

    # 同步模式（可选）
    with carla_sync_mode(client, world, enabled=args.sync, fixed_dt=args.fixed_dt):
        scenario_list = ["cones", "jaywalker", "trimma", "construction"] if args.scenario == "all" else [args.scenario]

        for s in scenario_list:
            for ep in range(args.episodes):
                print(f"\n=== Scenario: {s} | Episode {ep+1}/{args.episodes} ===")
                collect_one_episode(
                    client=client,
                    scenario_name=s,
                    output_dir=args.out,
                    max_steps=args.max_steps,
                    future_steps=args.future_steps,
                    tm_port=args.tm_port,
                    save_images=(not args.no_images),
                    image_width=args.image_width,
                    image_height=args.image_height,
                )


if __name__ == "__main__":
    main()
