# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from tutorial.assets.hummingbird import HUMMINGBIRD_CFG
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
import numpy as np
from gymnasium import spaces

_robot_spawn_cfg = HUMMINGBIRD_CFG.spawn.copy()
_robot_spawn_cfg.activate_contact_sensors = True


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 无人机模型
    robot_cfg: ArticulationCfg = HUMMINGBIRD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", spawn=_robot_spawn_cfg
    )
    # 无人机主观深度相机
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_cam",
        update_period=0.04,
        height=48,
        width=64,  # 深度相机的输出(48, 64)
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.4775,
            focus_distance=10.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 15),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -1.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        ),
    )
    # 接触传感器
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link|rotor_0|rotor_1|rotor_2|rotor_3",
        update_period=1 / 120,
        history_length=6,
        debug_vis=True,
    )
    # 添加地板
    floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(25.0, 25.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.04)),
    )

    # 墙壁 (迷宫外墙)
    wall_north = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallNorth",
        spawn=sim_utils.CuboidCfg(
            size=(25.0, 0.5, 4.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 12.5, 2.0)),
    )
    wall_south = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallSouth",
        spawn=sim_utils.CuboidCfg(
            size=(25.0, 0.5, 4.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -12.5, 2.0)),
    )
    wall_east = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallEast",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 25.0, 4.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(12.5, 0.0, 2.0)),
    )
    wall_west = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallWest",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 25.0, 4.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-12.5, 0.0, 2.0)),
    )


# 定义迷宫障碍物 (使用 Cuboid 代替 Cylinder)
# 这里仅作为一个示例迷宫布局，你可以根据需要调整位置
maze_obstacles = [
    # ((x, y, z), (sx, sy, sz))
    ((0.0, 0.0, 2.0), (10.0, 0.5, 4.0)),  # 中心横向墙
    ((5.0, 5.0, 2.0), (0.5, 10.0, 4.0)),  # 右侧纵向墙
    ((-5.0, -5.0, 2.0), (0.5, 10.0, 4.0)),  # 左侧纵向墙
    ((0.0, 8.0, 2.0), (10.0, 0.5, 4.0)),  # 北部横向墙
    ((0.0, -8.0, 2.0), (10.0, 0.5, 4.0)),  # 南部横向墙
]

for i, (pos, size) in enumerate(maze_obstacles):
    obstacle_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )
    setattr(MySceneCfg, f"Obstacle_{i}", obstacle_cfg)


@configclass
class TutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5  # 5个dt进行一次决策
    episode_length_s = 40  # 导航任务最长时间40s
    random_reset = False  # 是否随机重置环境

    action_space = spaces.Box(
        low=np.array([-0.1, -0.5, -np.pi / 6], dtype=np.float32),  # 每个维度的最小值
        high=np.array([2.0, 0.5, np.pi / 6], dtype=np.float32),  # 每个维度的最大值
        shape=(3,),  # 动作空间的形状，3表示三个连续动作
        dtype=np.float32,
    )  # 连续动作空间，表示机器人的线速度(vx, vy)和偏航角速度(yaw_rate)
    # - spaces definition
    # observation_space = 4
    observation_space = {
        "camera": spaces.Box(low=-1, high=1, shape=(3, 12, 16), dtype=np.float32),
        "robot-state": 8,
        "critic-state": 15,
    }
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = MySceneCfg(
        num_envs=4, env_spacing=25.0, replicate_physics=True
    )
