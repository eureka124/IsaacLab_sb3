# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from isaac_lab_tutorial.assets.five_in_drone import FIVE_IN_DRONE 

from isaaclab.assets import ArticulationCfg ,RigidObjectCfg,AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg,UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, UsdFileCfg, RigidBodyPropertiesCfg, CollisionPropertiesCfg, MassPropertiesCfg,CylinderCfg
import isaacsim.core.utils.prims as prim_utils
import gymnasium as gym
from gymnasium import spaces
import numpy as np
@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = FIVE_IN_DRONE.replace(prim_path="/World/envs/env_.*/Robot")

    """Design the scene with sensors on the robot."""
    contact_forces = ContactSensorCfg(

        prim_path="{ENV_REGEX_NS}/Robot/prop.*", update_period=0.0, history_length=6, debug_vis=True,

    )
        # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/front_cam",
        update_period=0.04,
        height=16,
        width=16,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.4775, focus_distance=10.0, horizontal_aperture=20.955, clipping_range=(0.1, 15)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    Obstacle_0 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obstacle_0",
            spawn=sim_utils.CylinderCfg(
                radius=0.5,
                height=5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(7.0, 7.0, 2.5),)  # 修改后
        )
    Obstacle_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.3,
            height=6,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.3333, 7.0, 3),)  # 修改后
    )
    Obstacle_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_2",
        spawn=sim_utils.CylinderCfg(
            radius=0.2,
            height=3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.3333, 7.0, 1.5),)  # 修改后
    )
    Obstacle_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_3",
        spawn=sim_utils.CylinderCfg(
            radius=0.45,
            height=5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, 7.0, 2.5),)  # 修改后
    )
    Obstacle_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_4",
        spawn=sim_utils.CylinderCfg(
            radius=0.4,
            height=7,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.0, 2.3333, 3.5),)  # 修改后
    )
    Obstacle_5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_5",
        spawn=sim_utils.CylinderCfg(
            radius=0.25,
            height=9,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.3333, 2.3333, 4.5),)  # 修改后
    )
    Obstacle_6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_6",
        spawn=sim_utils.CylinderCfg(
            radius=0.3,
            height=5.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.3333, 2.3333, 2.75),)  # 修改后
    )
    Obstacle_7 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_7",
        spawn=sim_utils.CylinderCfg(
            radius=0.25,
            height=4.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, 2.3333, 2.0),)  # 修改后
    )
    Obstacle_8 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_8",
        spawn=sim_utils.CylinderCfg(
            radius=0.25,
            height=5.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.0, -2.3333, 2.5),)  # 修改后
    )
    Obstacle_9 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_9",
        spawn=sim_utils.CylinderCfg(
            radius=0.35,
            height=6.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.3333, -2.3333, 3.0),)  # 修改后
    )
    Obstacle_10 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_10",
        spawn=sim_utils.CylinderCfg(
            radius=0.45,
            height=7.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.3333, -2.3333, 3.5),)  # 修改后
    )
    Obstacle_11 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_11",
        spawn=sim_utils.CylinderCfg(
            radius=0.55,
            height=8.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, 2.3333, 4.0),)  # 修改后：y从1.5调整为2.3333
    )
    Obstacle_12 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_12",
        spawn=sim_utils.CylinderCfg(
            radius=0.65,
            height=4.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.0, -7.0, 2.25),)  # 修改后
    )
    Obstacle_13 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_13",
        spawn=sim_utils.CylinderCfg(
            radius=0.75,
            height=5.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.3333, -7.0, 2.6),)  # 修改后
    )
    Obstacle_14 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_14",
        spawn=sim_utils.CylinderCfg(
            radius=0.85,
            height=6.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.3333, -7.0, 3.0),)  # 修改后
    )
    Obstacle_15 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_15",
        spawn=sim_utils.CylinderCfg(
            radius=0.75,
            height=4.8,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.2), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, -7.0, 2.4),)  # 修改后
    )

    prim_utils.create_prim(f"/World/move_obstacle", "Xform")#生成一个路径，用于存放可动障碍物，可动障碍物的生成在env.py的set scene



@configclass
class IsaacLabTutorial3EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5      #5个dt进行一次决策
    episode_length_s = 40       #导航任务最长时间

    action_space = spaces.Box(
        low=np.array([0.0, -np.pi/3], dtype=np.float32),   # 每个维度的最小值
        high=np.array([2.0, np.pi/3], dtype=np.float32),   # 每个维度的最大值
        shape=(2,),  # 动作空间的形状，2表示两个连续动作
        dtype=np.float32
    )
    observation_space = {
        "camera": [ 3,16,16],  
        "robot-state": 5 
    }

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: InteractiveSceneCfg = SensorsSceneCfg(num_envs=4, env_spacing=25, replicate_physics=True)

