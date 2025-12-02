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
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
import isaacsim.core.utils.prims as prim_utils
import numpy as np
from gymnasium import spaces
import torch
from random import gauss, seed


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 无人机模型
    robot_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=CRAZYFLIE_CFG.spawn.replace(activate_contact_sensors=True)
    )
    # 无人机主观深度相机
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/front_cam",
        update_period=0.04,
        height=32,
        width=32,  # 深度相机的输出(32, 32)
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
        prim_path="{ENV_REGEX_NS}/Robot/body",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )


# 定义多个静止障碍物的位置
obstacle_positions = [
    (0.4, 0.2, 1.5),
    (0.8, -0.5, 1.5),
    (-0.2, 0.6, 1.5),
    (0.5, 0.8, 1.5),
    (-0.6, -0.4, 1.5),
    (-0.3, -0.7, 1.5),
    (0.0, 0.5, 1.5),
]

# 循环创建并添加障碍物配置到 MySceneCfg
for i, pos in enumerate(obstacle_positions):
    obstacle_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
        spawn=sim_utils.CylinderCfg(
            radius=gauss(0.1, 0.05),
            height=3,
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
    action_space = spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),  # 每个维度的最小值
        high=np.array([1.0, 1.0], dtype=np.float32),  # 每个维度的最大值
        shape=(2,),  # 动作空间的形状，2表示两个连续动作
        dtype=np.float32
    )  # 连续动作空间，表示机器人的线速度和角速度
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    # observation_space = 4
    observation_space = {"camera": [3, 16, 16], "relative-position": 2}
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = MySceneCfg(num_envs=4, env_spacing=2.0, replicate_physics=True)
    # static_obstacle: StaticObstacle = StaticObstacle()
