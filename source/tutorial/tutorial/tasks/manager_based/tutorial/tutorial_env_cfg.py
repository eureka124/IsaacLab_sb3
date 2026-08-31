from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from tutorial.assets.hummingbird import HUMMINGBIRD_CFG
from tutorial.tasks.direct.tutorial.tutorial_env_cfg import (
    OBSTACLE_GRID_SIZE,
    maze_obstacles,
    obstacle_positions,
    u_obstacle_positions,
    u_obstacle_yaws,
)

from . import mdp


U_OBSTACLE_USD_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../assets/usd/u_obstacle_collision.usda")
)


@configclass
class TutorialManagerBasedSceneCfg(InteractiveSceneCfg):
    """Manager-based version of the original drone-navigation scene."""

    robot: ArticulationCfg = HUMMINGBIRD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_cam",
        update_period=0.04,
        height=48,
        width=64,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.4775,
            focus_distance=10.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 15.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -1.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(base_link|rotor_0|rotor_1|rotor_2|rotor_3)",
        update_period=1.0 / 120.0,
        history_length=6,
        debug_vis=False,
    )

    floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(30.0, 30.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -11.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=4000.0, color=(1.0, 1.0, 0.75)),
    )


for maze_index, (position, size) in enumerate(maze_obstacles):
    maze_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/maze_obstacle_{maze_index}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )
    setattr(TutorialManagerBasedSceneCfg, f"maze_obstacle_{maze_index}", maze_cfg)


for obstacle_index, (position, radius, height) in enumerate(obstacle_positions):
    obstacle_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{obstacle_index}",
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )
    setattr(TutorialManagerBasedSceneCfg, f"obstacle_{obstacle_index}", obstacle_cfg)


for u_index, (position, yaw) in enumerate(zip(u_obstacle_positions, u_obstacle_yaws)):
    orientation = (float(math.cos(yaw * 0.5)), 0.0, 0.0, float(math.sin(yaw * 0.5)))
    u_obstacle_cfg = AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/U_Obstacle_{u_index}",
        spawn=sim_utils.UsdFileCfg(usd_path=U_OBSTACLE_USD_PATH),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=orientation),
    )
    setattr(TutorialManagerBasedSceneCfg, f"u_obstacle_{u_index}", u_obstacle_cfg)


@configclass
class ActionsCfg:
    velocity_yaw = mdp.VelocityYawActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        camera = ObsTerm(
            func=mdp.normalized_depth_image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "max_distance": 10.0},
            clip=(-1.0, 1.0),
        )
        robot_state = ObsTerm(
            func=mdp.robot_state,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_navigation = EventTerm(
        func=mdp.reset_robot_goal_and_obstacles,
        mode="reset",
        params={"obstacle_count": len(obstacle_positions), "grid_size": OBSTACLE_GRID_SIZE},
    )


@configclass
class RewardsCfg:
    goal_velocity = RewTerm(
        func=mdp.velocity_towards_goal,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_smoothness = RewTerm(func=mdp.action_smoothness, weight=-0.1)
    contact_force = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-200.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "saturation_force": 50.0,
            "ramp_fraction": 0.2,
        },
    )
    goal_reached = RewTerm(
        func=mdp.goal_reached,
        weight=300.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.4},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.tutorial_time_out, time_out=True)
    collision = DoneTerm(
        func=mdp.collision_termination,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 50.0},
    )
    goal_reached = DoneTerm(
        func=mdp.goal_reached_termination,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.4},
    )


@configclass
class TutorialManagerBasedEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based drone navigation environment."""

    # Aggregate transitions across all parallel environments. Training scripts
    # overwrite this from the active agent configuration.
    contact_force_penalty_max_steps: int = 70_000_000

    scene: TutorialManagerBasedSceneCfg = TutorialManagerBasedSceneCfg(
        num_envs=4,
        env_spacing=30.0,
        replicate_physics=True,
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 5
        self.episode_length_s = 40.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = False
        mdp.set_direct_tutorial_reward_weights(self.rewards, self.decimation * self.sim.dt)
        self.viewer.eye = (25.0, 25.0, 20.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)
