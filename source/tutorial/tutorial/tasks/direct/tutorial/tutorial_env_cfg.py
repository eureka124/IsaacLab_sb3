# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from tutorial.assets.five_in_drone import FIVE_IN_DRONE
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
import isaacsim.core.utils.prims as prim_utils
import numpy as np
from gymnasium import spaces
import torch
from random import gauss, seed

_robot_spawn_cfg = FIVE_IN_DRONE.spawn.copy()
_robot_spawn_cfg.activate_contact_sensors = True
@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 无人机模型
    robot_cfg: ArticulationCfg = FIVE_IN_DRONE.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=_robot_spawn_cfg
    )
    # 无人机主观深度相机
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/front_cam",
        update_period=0.04,
        height=16,
        width=12,  # 深度相机的输出(16, 12)
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.4775,
            focus_distance=10.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 15),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        ),
    )
    # 接触传感器
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/prop.*",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )


# 定义多个静止障碍物的位置
obstacle_positions = [
    (7.0, 7.0, 2.5),
    (2.3333, 7.0, 2.5),
    (-2.3333, 7.0, 2.5),
    (-7.0, 7.0, 2.5),
    (7.0, 2.3333, 2.5),
    (2.3333, 2.3333, 2.5),
    (-2.3333, 2.3333, 2.5),
    (-7.0, 2.3333, 2.5),
    (7.0, -2.3333, 2.5),
    (2.3333, -2.3333, 2.5),
    (-2.3333, -2.3333, 2.5),
    (-7.0, -2.3333, 2.5),
    (7.0, -7.0, 2.5),
    (2.3333, -7.0, 2.5),
    (-2.3333, -7.0, 2.5),
    (-7.0, -7.0, 2.5),
]

# 循环创建并添加障碍物配置到 MySceneCfg
for i, pos in enumerate(obstacle_positions):
    obstacle_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
        spawn=sim_utils.CylinderCfg(
            radius=gauss(0.5, 0.1),
            height=5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )
    setattr(MySceneCfg, f"Obstacle_{i}", obstacle_cfg)


@configclass
class TutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5      # 5个dt进行一次决策
    episode_length_s = 40       # 导航任务最长时间40s

    action_space = spaces.Box(
        low=np.array([-10.0, -10.0], dtype=np.float32),  # 每个维度的最小值
        high=np.array([10.0, 10.0], dtype=np.float32),  # 每个维度的最大值
        shape=(2,),  # 动作空间的形状，2表示两个连续动作
        dtype=np.float32
    )  # 连续动作空间，表示机器人的线速度和角速度

    # - spaces definition
    # observation_space = 4
    observation_space = {"camera": [3, 16, 12], "robot-state": 4}
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = MySceneCfg(num_envs=4, env_spacing=25.0, replicate_physics=True)
    # static_obstacle: StaticObstacle = StaticObstacle()
