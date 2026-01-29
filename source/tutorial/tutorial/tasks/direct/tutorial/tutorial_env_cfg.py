# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from tutorial.assets.hummingbird import HUMMINGBIRD_CFG
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
import numpy as np
from gymnasium import spaces

_robot_spawn_cfg = HUMMINGBIRD_CFG.spawn.copy()
_robot_spawn_cfg.activate_contact_sensors = False


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 无人机模型
    robot_cfg: ArticulationCfg = HUMMINGBIRD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", spawn=_robot_spawn_cfg
    )
    # 无人机主观深度相机
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_cam",
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
            pos=(0.4, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        ),
    )

    # 接触传感器
    # contact_forces: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=True,
    # )


# 定义围挡（墙壁）
# 场景范围 11.2 x 5.2 (-5.6~5.6, -2.6~2.6)
wall_definitions = [
    # (pos(x,y,z), size(x,y,z)) - 墙壁厚度0.2，高度2.0
    ((5.6, 0.0, 1.0), (0.2, 5.2, 2.0)),  # Front (X+)
    ((-5.6, 0.0, 1.0), (0.2, 5.2, 2.0)),  # Back (X-)
    ((0.0, 2.6, 1.0), (11.4, 0.2, 2.0)),  # Left (Y+)
    ((0.0, -2.6, 1.0), (11.4, 0.2, 2.0)),  # Right (Y-)
    ((0.0, 0.0, 3.0), (11.6, 5.6, 0.2)),  # Ceiling (Z+)
]

# 添加围挡到 MySceneCfg
for i, (pos, size) in enumerate(wall_definitions):
    wall_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Wall_{i}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )
    setattr(MySceneCfg, f"Wall_{i}", wall_cfg)


# 定义障碍物
# 长方体: 0.6 x 0.6 x 1.8 -> Pos Z = 0.9
# 圆柱体: d=0.16 (r=0.08), h=1.04 -> Pos Z = 0.52
# 均匀分布在 11.2 x 5.2 区域内
obstacles_data = [
    # Cuboids
    {"type": "cuboid", "pos": (-3.0, -1.0, 0.9), "size": (0.6, 0.6, 1.8)},
    {"type": "cuboid", "pos": (0.0, 1.0, 0.9), "size": (0.6, 0.6, 1.8)},
    {"type": "cuboid", "pos": (3.0, -1.0, 0.9), "size": (0.6, 0.6, 1.8)},
    # Cylinders
    {"type": "cylinder", "pos": (-3.0, 1.0, 0.52), "radius": 0.16, "height": 1.04},
    {"type": "cylinder", "pos": (0.0, -1.0, 0.52), "radius": 0.16, "height": 1.04},
    {"type": "cylinder", "pos": (3.0, 1.0, 0.52), "radius": 0.16, "height": 1.04},
]

# 添加障碍物到 MySceneCfg
for i, data in enumerate(obstacles_data):
    if data["type"] == "cuboid":
        spawn_cfg = sim_utils.CuboidCfg(
            size=data["size"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.5, 0.7)),
        )
    else:
        spawn_cfg = sim_utils.CylinderCfg(
            radius=data["radius"],
            height=data["height"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 0.3)),
        )

    obstacle_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=data["pos"]),
    )
    setattr(MySceneCfg, f"Obstacle_{i}", obstacle_cfg)


@configclass
class TutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5  # 5个dt进行一次决策
    episode_length_s = 40  # 导航任务最长时间40s
    random_reset = False  # 是否随机重置环境

    action_space = spaces.Box(
        low=np.array([-0.1, -10.0, -np.pi], dtype=np.float32),  # 每个维度的最小值
        high=np.array([10.0, 10.0, np.pi], dtype=np.float32),  # 每个维度的最大值
        shape=(3,),  # 动作空间的形状，3表示三个连续动作
        dtype=np.float32,
    )  # 连续动作空间，表示机器人的线速度(vx, vy)和偏航角速度(yaw_rate)
    # - spaces definition
    # observation_space = 4
    observation_space = {
        "camera": spaces.Box(low=0, high=255, shape=(3, 16, 12), dtype=np.uint8),
        "robot-state": 8,
        "critic-state": 15,
    }
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    # scene
    scene: InteractiveSceneCfg = MySceneCfg(
        num_envs=4, env_spacing=12.0, replicate_physics=True
    )
