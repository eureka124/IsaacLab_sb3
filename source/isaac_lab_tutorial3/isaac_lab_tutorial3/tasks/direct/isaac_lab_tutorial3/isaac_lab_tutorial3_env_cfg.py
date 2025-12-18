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
        width=12,
        data_types=[ "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.4775, focus_distance=10.0, horizontal_aperture=20.955, clipping_range=(0.1, 15)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    Obstacle_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_0",
        spawn=sim_utils.CylinderCfg(
            radius=0.327,
            height=3.545,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.976, 4.304, 1.7725)) 
    )

    Obstacle_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.315,
            height=3.964,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.918, -1.248, 1.982)) 
    )

    Obstacle_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_2",
        spawn=sim_utils.CylinderCfg(
            radius=0.221,
            height=3.926,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.835, 0.578, 1.963)) 
    )

    Obstacle_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_3",
        spawn=sim_utils.CylinderCfg(
            radius=0.461,
            height=3.778,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.257, -9.596, 1.889)) 
    )

    Obstacle_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_4",
        spawn=sim_utils.CylinderCfg(
            radius=0.235,
            height=3.781,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(9.572, 5.983, 1.8905)) 
    )

    Obstacle_5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_5",
        spawn=sim_utils.CylinderCfg(
            radius=0.324,
            height=3.522,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.798, -7.133, 1.761)) 
    )

    Obstacle_6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_6",
        spawn=sim_utils.CylinderCfg(
            radius=0.206,
            height=3.568,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.709, 5.485, 1.784)) 
    )

    Obstacle_7 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_7",
        spawn=sim_utils.CylinderCfg(
            radius=0.405,
            height=3.944,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.353, 2.242, 1.972)) 
    )

    Obstacle_8 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_8",
        spawn=sim_utils.CylinderCfg(
            radius=0.400,
            height=3.060,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.810, -1.259, 1.530)) 
    )

    Obstacle_9 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_9",
        spawn=sim_utils.CylinderCfg(
            radius=0.231,
            height=3.988,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.726, 1.404, 1.994)) 
    )

    Obstacle_10 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_10",
        spawn=sim_utils.CylinderCfg(
            radius=0.340,
            height=3.253,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.822, -6.774, 1.6265)) 
    )

    Obstacle_11 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_11",
        spawn=sim_utils.CylinderCfg(
            radius=0.341,
            height=3.976,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.058, 6.759, 1.988)) 
    )

    Obstacle_12 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_12",
        spawn=sim_utils.CylinderCfg(
            radius=0.285,
            height=3.039,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(9.535, 2.097, 1.5195)) 
    )

    Obstacle_13 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_13",
        spawn=sim_utils.CylinderCfg(
            radius=0.324,
            height=3.318,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.596, -4.077, 1.659)) 
    )

    Obstacle_14 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_14",
        spawn=sim_utils.CylinderCfg(
            radius=0.357,
            height=3.265,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.717, 3.849, 1.6325)) 
    )

    Obstacle_15 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_15",
        spawn=sim_utils.CylinderCfg(
            radius=0.400,
            height=3.319,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.121, 1.519, 1.6595)) 
    )

    Obstacle_16 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_16",
        spawn=sim_utils.CylinderCfg(
            radius=0.403,
            height=3.005,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.730, -9.598, 1.5025)) 
    )

    Obstacle_17 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_17",
        spawn=sim_utils.CylinderCfg(
            radius=0.289,
            height=3.699,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(9.055, -1.057, 1.8495)) 
    )

    Obstacle_18 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_18",
        spawn=sim_utils.CylinderCfg(
            radius=0.465,
            height=3.581,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.276, -2.070, 1.7905)) 
    )

    Obstacle_19 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_19",
        spawn=sim_utils.CylinderCfg(
            radius=0.393,
            height=3.956,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.851, 4.505, 1.978)) 
    )

    Obstacle_20 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_20",
        spawn=sim_utils.CylinderCfg(
            radius=0.241,
            height=3.429,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.203, -4.198, 1.7145)) 
    )

    Obstacle_21 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_21",
        spawn=sim_utils.CylinderCfg(
            radius=0.385,
            height=3.162,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(9.977, -7.011, 1.581)) 
    )

    Obstacle_22 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_22",
        spawn=sim_utils.CylinderCfg(
            radius=0.417,
            height=3.454,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.856, -8.617, 1.727)) 
    )

    Obstacle_23 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_23",
        spawn=sim_utils.CylinderCfg(
            radius=0.308,
            height=3.012,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.328, 9.510, 1.506)) 
    )

    Obstacle_24 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_24",
        spawn=sim_utils.CylinderCfg(
            radius=0.304,
            height=3.224,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.000, -9.630, 1.612)) 
    )

    Obstacle_25 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_25",
        spawn=sim_utils.CylinderCfg(
            radius=0.386,
            height=3.165,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.562, 4.088, 1.5825)) 
    )

    Obstacle_26 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_26",
        spawn=sim_utils.CylinderCfg(
            radius=0.422,
            height=3.944,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.036, -5.803, 1.972)) 
    )

    Obstacle_27 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_27",
        spawn=sim_utils.CylinderCfg(
            radius=0.254,
            height=3.378,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.312, -3.764, 1.689)) 
    )

    Obstacle_28 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_28",
        spawn=sim_utils.CylinderCfg(
            radius=0.399,
            height=3.217,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.732, 7.933, 1.6085)) 
    )

    Obstacle_29 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_29",
        spawn=sim_utils.CylinderCfg(
            radius=0.486,
            height=3.186,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.453, -4.529, 1.593)) 
    )

    Obstacle_30 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_30",
        spawn=sim_utils.CylinderCfg(
            radius=0.283,
            height=3.464,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.851, -1.506, 1.732)) 
    )

    Obstacle_31 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_31",
        spawn=sim_utils.CylinderCfg(
            radius=0.479,
            height=3.016,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.157, -5.183, 1.508)) 
    )

    Obstacle_32 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_32",
        spawn=sim_utils.CylinderCfg(
            radius=0.482,
            height=3.949,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.237, 9.231, 1.9745)) 
    )

    Obstacle_33 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_33",
        spawn=sim_utils.CylinderCfg(
            radius=0.455,
            height=3.293,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.984, 2.609, 1.6465)) 
    )

    Obstacle_34 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_34",
        spawn=sim_utils.CylinderCfg(
            radius=0.492,
            height=3.862,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.226, -5.514, 1.931)) 
    )

    Obstacle_35 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_35",
        spawn=sim_utils.CylinderCfg(
            radius=0.224,
            height=3.333,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(9.217, 8.131, 1.6665)) 
    )

    Obstacle_36 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_36",
        spawn=sim_utils.CylinderCfg(
            radius=0.224,
            height=3.147,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.512, -9.771, 1.5735)) 
    )

    Obstacle_37 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_37",
        spawn=sim_utils.CylinderCfg(
            radius=0.239,
            height=3.278,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.200, 0.724, 1.639)) 
    )

    Obstacle_38 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_38",
        spawn=sim_utils.CylinderCfg(
            radius=0.219,
            height=3.398,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.310, -6.812, 1.699)) 
    )

    Obstacle_39 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_39",
        spawn=sim_utils.CylinderCfg(
            radius=0.329,
            height=3.016,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.345, 2.289, 1.508)) 
    )

    Obstacle_40 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_40",
        spawn=sim_utils.CylinderCfg(
            radius=0.270,
            height=3.046,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.789, 6.573, 1.523)) 
    )

    Obstacle_41 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_41",
        spawn=sim_utils.CylinderCfg(
            radius=0.218,
            height=3.690,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.882, 9.019, 1.845)) 
    )

    Obstacle_42 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_42",
        spawn=sim_utils.CylinderCfg(
            radius=0.312,
            height=3.380,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.614, 7.634, 1.690)) 
    )

    Obstacle_43 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_43",
        spawn=sim_utils.CylinderCfg(
            radius=0.291,
            height=3.449,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.976, -5.244, 1.7245)) 
    )

    Obstacle_44 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_44",
        spawn=sim_utils.CylinderCfg(
            radius=0.227,
            height=3.922,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-9.842, -2.546, 1.961)) 
    )

    Obstacle_45 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_45",
        spawn=sim_utils.CylinderCfg(
            radius=0.343,
            height=3.607,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.440, 3.607, 1.8035)) 
    )

    Obstacle_46 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_46",
        spawn=sim_utils.CylinderCfg(
            radius=0.429,
            height=3.890,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.752, 6.261, 1.945)) 
    )

    Obstacle_47 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_47",
        spawn=sim_utils.CylinderCfg(
            radius=0.494,
            height=3.457,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.829, -0.588, 1.7285)) 
    )

    Obstacle_48 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_48",
        spawn=sim_utils.CylinderCfg(
            radius=0.257,
            height=3.365,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.698, -1.915, 1.6825)) 
    )

    Obstacle_49 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_49",
        spawn=sim_utils.CylinderCfg(
            radius=0.464,
            height=3.393,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.544, -9.421, 1.6965)) 
    )

    Obstacle_50 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_50",
        spawn=sim_utils.CylinderCfg(
            radius=0.350,
            height=3.365,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.816, 9.747, 1.6825)) 
    )

    Obstacle_51 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_51",
        spawn=sim_utils.CylinderCfg(
            radius=0.458,
            height=3.215,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.729, -5.835, 1.6075)) 
    )

    Obstacle_52 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_52",
        spawn=sim_utils.CylinderCfg(
            radius=0.286,
            height=3.107,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.512, 9.074, 1.5535)) 
    )

    Obstacle_53 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_53",
        spawn=sim_utils.CylinderCfg(
            radius=0.491,
            height=3.475,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.398, 3.042, 1.7375)) 
    )

    Obstacle_54 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_54",
        spawn=sim_utils.CylinderCfg(
            radius=0.407,
            height=3.186,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-9.987, 8.833, 1.593)) 
    )

    Obstacle_55 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_55",
        spawn=sim_utils.CylinderCfg(
            radius=0.351,
            height=3.627,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.328, 6.151, 1.8135)) 
    )

    Obstacle_56 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_56",
        spawn=sim_utils.CylinderCfg(
            radius=0.393,
            height=3.674,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.973, 0.743, 1.837)) 
    )

    Obstacle_57 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_57",
        spawn=sim_utils.CylinderCfg(
            radius=0.491,
            height=3.698,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.396, 8.182, 1.849)) 
    )

    Obstacle_58 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_58",
        spawn=sim_utils.CylinderCfg(
            radius=0.202,
            height=3.185,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.795, 4.592, 1.5925)) 
    )

    Obstacle_59 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_59",
        spawn=sim_utils.CylinderCfg(
            radius=0.352,
            height=3.879,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(9.459, -3.231, 1.9395)) 
    )

    prim_utils.create_prim(f"/World/move_obstacle", "Xform")#生成一个路径，用于存放可动障碍物，可动障碍物的生成在env.py的set scene



@configclass
class IsaacLabTutorial3EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5      #5个dt进行一次决策
    episode_length_s = 30      #导航任务最长时间

    action_space = spaces.Box(
        low=np.array([-0.000001, -np.pi/6], dtype=np.float32),   # 每个维度的最小值
        high=np.array([2.0, np.pi/6], dtype=np.float32),   # 每个维度的最大值
        shape=(2,),  # 动作空间的形状，2表示两个连续动作
        dtype=np.float32
    )
    observation_space = spaces.Dict({
        "camera": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32),
        "robot-state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    })

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: InteractiveSceneCfg = SensorsSceneCfg(num_envs=4, env_spacing=50, replicate_physics=True)

