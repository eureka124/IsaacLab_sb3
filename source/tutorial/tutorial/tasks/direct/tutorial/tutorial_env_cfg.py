# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
import numpy as np
from gymnasium import spaces


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 飞机参数，激活接触传感器
    robot1_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot1",
        spawn=CRAZYFLIE_CFG.spawn.replace(activate_contact_sensors=True),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot1/body/front_cam",
        update_period=0.04,
        height=16,
        width=16,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.4775,
            focus_distance=10.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 15),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot1/body", update_period=0.0, history_length=6, debug_vis=True
    )
    # obstacle = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Obstacle_13",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.3,
    #         height=0.5,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(1, -1, 0.5),
    #     )
    # )


@configclass
class TutorialEnvCfg(DirectRLEnvCfg):
    # env
    action_space = spaces.Box(
        low=np.array([-0.5, -0.5], dtype=np.float32),  # 每个维度的最小值
        high=np.array([0.5, 0.5], dtype=np.float32),  # 每个维度的最大值
        shape=(2,),  # 动作空间的形状，2表示两个连续动作
        dtype=np.float32,
    )
    decimation = 5  # 控制频率与模拟频率的比率
    episode_length_s = 5.0
    # - spaces definition
    # observation_space = {"camera":(3,16,16), "robot-state": 4}
    observation_space = 4
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # scene
    scene: InteractiveSceneCfg = MySceneCfg(num_envs=8, env_spacing=2.0, replicate_physics=True)
    # # custom parameters/scales
    # # - controllable joint
    # cart_dof_name = "slider_to_cart"
    # pole_dof_name = "cart_to_pole"
    # # - action scale
    # action_scale = 100.0  # [N]
    # # - reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    # rew_scale_pole_pos = -1.0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_vel = -0.005
    # # - reset states/conditions
    # initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    # max_cart_pos = 3.0  # reset if cart exceeds this position [m]
