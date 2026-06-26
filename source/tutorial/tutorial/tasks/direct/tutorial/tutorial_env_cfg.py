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


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 无人机模型
    robot_cfg: ArticulationCfg = HUMMINGBIRD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, -1.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    # 接触传感器 (监听机身和旋翼的接触力)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(base_link|rotor_0|rotor_1|rotor_2|rotor_3)",
        update_period=1 / 120,
        history_length=6,
        debug_vis=True,
    )
    # 添加地板
    floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(30.0, 30.0, 0.1),
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
    sub_floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SubFloor",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -11.0)),
    )


# # 定义迷宫障碍物
# maze_obstacles = [
#     # ((x, y, z), (sx, sy, sz))
#     ((-7.5, -12.0, 2.0), (10.0, 1, 4.0)),
#     ((-12.0, -6.5, 2.0), (1, 10.0, 4.0)),
#     ((7.5, -12.0, 2.0), (10.0, 1, 4.0)),
#     ((12.0, -6.5, 2.0), (1, 10.0, 4.0)),
#     ((-6.5, 12.0, 2.0), (10.0, 1, 4.0)),
#     ((-12.0, 7.5, 2.0), (1, 10.0, 4.0)),
#     ((6.5, 12.0, 2.0), (10.0, 1, 4.0)),
#     ((12.0, 7.5, 2.0), (1, 10.0, 4.0)),
# ]

# for i, (pos, size) in enumerate(maze_obstacles):
#     obstacle_cfg = RigidObjectCfg(
#         prim_path=f"{{ENV_REGEX_NS}}/maze_Obstacle_{i}",
#         spawn=sim_utils.CuboidCfg(
#             size=size,
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                 kinematic_enabled=False,
#                 disable_gravity=True,
#                 linear_damping=1000.0,
#                 angular_damping=1000.0,
#             ),
#             mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
#             collision_props=sim_utils.CollisionPropertiesCfg(),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
#     )
#     setattr(MySceneCfg, f"maze_Obstacle_{i}", obstacle_cfg)


# 定义多个静止障碍物的位置
obstacle_positions = [
    # (位置(x, y, z), 半径, 高度)
    ((0.976, 4.304, 1.7725), 0.327, 3.545),  # Obstacle_0
    ((2.918, -1.248, 1.982), 0.315, 3.964),  # Obstacle_1
    ((5.835, 0.578, 1.963), 0.221, 3.926),  # Obstacle_2
    ((-8.257, -9.596, 1.889), 0.461, 3.778),  # Obstacle_3
    ((9.572, 5.983, 1.8905), 0.235, 3.781),  # Obstacle_4
    ((2.798, -7.133, 1.761), 0.324, 3.522),  # Obstacle_5
    ((-4.709, 5.485, 1.784), 0.206, 3.568),  # Obstacle_6
    ((2.353, 2.242, 1.972), 0.405, 3.944),  # Obstacle_7
    ((-2.810, -1.259, 1.530), 0.400, 3.060),  # Obstacle_8
    ((-2.726, 1.404, 1.994), 0.231, 3.988),  # Obstacle_9
    ((-5.822, -6.774, 1.6265), 0.340, 3.253),  # Obstacle_10
    ((-8.058, 6.759, 1.988), 0.341, 3.976),  # Obstacle_11
    ((9.535, 2.097, 1.5195), 0.285, 3.039),  # Obstacle_12
    ((-7.596, -4.077, 1.659), 0.324, 3.318),  # Obstacle_13
    ((-8.717, 3.849, 1.6325), 0.357, 3.265),  # Obstacle_14
    ((-8.121, 1.519, 1.6595), 0.400, 3.319),  # Obstacle_15
    ((1.730, -9.598, 1.5025), 0.403, 3.005),  # Obstacle_16
    ((9.055, -1.057, 1.8495), 0.289, 3.699),  # Obstacle_17
    ((6.276, -2.070, 1.7905), 0.465, 3.581),  # Obstacle_18
    ((3.851, 4.505, 1.978), 0.393, 3.956),  # Obstacle_19
    ((3.203, -4.198, 1.7145), 0.241, 3.429),  # Obstacle_20
    ((9.977, -7.011, 1.581), 0.385, 3.162),  # Obstacle_21
    ((-1.856, -8.617, 1.727), 0.417, 3.454),  # Obstacle_22
    ((7.328, 9.510, 1.506), 0.308, 3.012),  # Obstacle_23
    ((-6.000, -9.630, 1.612), 0.304, 3.224),  # Obstacle_24
    ((8.562, 4.088, 1.5825), 0.386, 3.165),  # Obstacle_25
    ((-2.036, -5.803, 1.972), 0.422, 3.944),  # Obstacle_26
    ((-1.312, -3.764, 1.689), 0.254, 3.378),  # Obstacle_27
    ((0.732, 7.933, 1.6085), 0.399, 3.217),  # Obstacle_28
    ((7.453, -4.529, 1.593), 0.486, 3.186),  # Obstacle_29
    ((-5.851, -1.506, 1.732), 0.283, 3.464),  # Obstacle_30
    ((-4.157, -5.183, 1.508), 0.479, 3.016),  # Obstacle_31
    ((-3.237, 9.231, 1.9745), 0.482, 3.949),  # Obstacle_32
    ((5.984, 2.609, 1.6465), 0.455, 3.293),  # Obstacle_33
    ((0.226, -5.514, 1.931), 0.492, 3.862),  # Obstacle_34
    ((9.217, 8.131, 1.6665), 0.224, 3.333),  # Obstacle_35
    ((4.512, -9.771, 1.5735), 0.224, 3.147),  # Obstacle_36
    ((6.310, -6.812, 1.699), 0.219, 3.398),  # Obstacle_38
    ((-5.345, 2.289, 1.508), 0.329, 3.016),  # Obstacle_39
    ((-2.789, 6.573, 1.523), 0.270, 3.046),  # Obstacle_40
    ((-7.882, 9.019, 1.845), 0.218, 3.690),  # Obstacle_41
    ((4.614, 7.634, 1.690), 0.312, 3.380),  # Obstacle_42
    ((4.976, -5.244, 1.7245), 0.291, 3.449),  # Obstacle_43
    ((-9.842, -2.546, 1.961), 0.227, 3.922),  # Obstacle_44
    ((-3.440, 3.607, 1.8035), 0.343, 3.607),  # Obstacle_45
    ((2.752, 6.261, 1.945), 0.429, 3.890),  # Obstacle_46
    ((-8.829, -0.588, 1.7285), 0.494, 3.457),  # Obstacle_47
    ((0.698, -1.915, 1.6825), 0.257, 3.365),  # Obstacle_48
    ((8.544, -9.421, 1.6965), 0.464, 3.393),  # Obstacle_49
    ((3.816, 9.747, 1.6825), 0.350, 3.365),  # Obstacle_50
    ((-8.729, -5.835, 1.6075), 0.458, 3.215),  # Obstacle_51
    ((-5.512, 9.074, 1.5535), 0.286, 3.107),  # Obstacle_52
    ((-1.398, 3.042, 1.7375), 0.491, 3.475),  # Obstacle_53
    ((-9.987, 8.833, 1.593), 0.407, 3.186),  # Obstacle_54
    ((6.328, 6.151, 1.8135), 0.351, 3.627),  # Obstacle_55
    ((7.973, 0.743, 1.837), 0.393, 3.674),  # Obstacle_56
    ((-1.396, 8.182, 1.849), 0.491, 3.698),  # Obstacle_57
    ((-6.795, 4.592, 1.5925), 0.202, 3.185),  # Obstacle_58
    ((9.459, -3.231, 1.9395), 0.352, 3.879),  # Obstacle_59
]

# 循环创建并添加障碍物配置到 MySceneCfg
for i, (pos, ra, hei) in enumerate(obstacle_positions):
    obstacle_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
        spawn=sim_utils.CylinderCfg(
            radius=ra,
            height=hei,
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

    action_space = spaces.Box(
        low=np.array([-0.1, -0.5, -np.pi / 3], dtype=np.float32),  # 每个维度的最小值
        high=np.array([2.0, 0.5, np.pi / 3], dtype=np.float32),  # 每个维度的最大值
        shape=(3,),  # 动作空间的形状，3表示三个连续动作
        dtype=np.float32,
    )  # 连续动作空间，表示机器人的线速度(vx, vy)和偏航角速度(yaw_rate)
    # - spaces definition
    # observation_space = 4
    observation_space = {
        "camera": spaces.Box(low=-1.0, high=1.0, shape=(1, 12, 16), dtype=np.float32),
        "robot-state": 8,
        # "critic-toa": spaces.Box(low=-1.0, high=1.0, shape=(1, 201, 201), dtype=np.float32),
    }
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = MySceneCfg(num_envs=4, env_spacing=30.0, replicate_physics=True)
