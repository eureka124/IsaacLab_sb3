"""Manager-based multi-drone environment for the six training mazes."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from tutorial.assets.hummingbird import HUMMINGBIRD_CFG

from ..tutorial.alignment import TUTORIAL_DEPTH_MAX_DISTANCE
from . import mdp
from .isolation import CAMERA_MAX_DISTANCE, DRONE_XY_LIMIT, ENV_SPACING, validate_isolation_settings
from .layouts import (
    ARENA_HALF_EXTENT,
    ARENA_HEIGHT,
    INNER_WALL_LENGTH,
    RANDOM_CYLINDER_SPECS,
    WALL_THICKNESS,
)

# Per-policy-step weights. ``set_direct_tutorial_reward_weights`` compensates
# for RewardManager's automatic multiplication by the policy step duration.
TRAINING_MAZE_REWARD_WEIGHTS = {
    "goal_velocity": 0.1,
    "toa_progress": 5.0,
    "contact_force": -100.0,
}


def _fixed_cuboid(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )


def _fixed_cylinder(radius: float, height: float) -> sim_utils.CylinderCfg:
    return sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.43, 0.55)),
    )


@configclass
class TrainingMazesSceneCfg(InteractiveSceneCfg):
    """One independently simulated drone and maze per cloned environment."""

    robot: ArticulationCfg = HUMMINGBIRD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=_fixed_cuboid((2.0 * ARENA_HALF_EXTENT, 2.0 * ARENA_HALF_EXTENT, 0.1)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )

    boundary_north = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundaryNorth",
        spawn=_fixed_cuboid((2.0 * ARENA_HALF_EXTENT, WALL_THICKNESS, ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, ARENA_HALF_EXTENT, ARENA_HEIGHT * 0.5)),
    )
    boundary_south = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundarySouth",
        spawn=_fixed_cuboid((2.0 * ARENA_HALF_EXTENT, WALL_THICKNESS, ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -ARENA_HALF_EXTENT, ARENA_HEIGHT * 0.5)),
    )
    boundary_east = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundaryEast",
        spawn=_fixed_cuboid((WALL_THICKNESS, 2.0 * ARENA_HALF_EXTENT, ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(ARENA_HALF_EXTENT, 0.0, ARENA_HEIGHT * 0.5)),
    )
    boundary_west = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundaryWest",
        spawn=_fixed_cuboid((WALL_THICKNESS, 2.0 * ARENA_HALF_EXTENT, ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-ARENA_HALF_EXTENT, 0.0, ARENA_HEIGHT * 0.5)),
    )

    # Every maze uses the same three cuboids.  The reset event changes their
    # poses and parks unused slots below the floor, preserving homogeneous
    # cloned physics for high environment counts.
    inner_wall_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InnerWall_0",
        spawn=_fixed_cuboid((INNER_WALL_LENGTH, WALL_THICKNESS, ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    inner_wall_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InnerWall_1",
        spawn=_fixed_cuboid((INNER_WALL_LENGTH, WALL_THICKNESS, ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    inner_wall_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InnerWall_2",
        spawn=_fixed_cuboid((INNER_WALL_LENGTH, WALL_THICKNESS, ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )

    # Cylinder geometry stays fixed for the lifetime of the simulation.  The
    # reset event randomizes only XY poses, which preserves physics replication.
    random_cylinders = RigidObjectCollectionCfg(
        rigid_objects={
            f"cylinder_{index:02d}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/RandomCylinder_{index:02d}",
                spawn=_fixed_cylinder(radius, height),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
            )
            for index, (radius, height) in enumerate(RANDOM_CYLINDER_SPECS)
        }
    )

    camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.04,
        mesh_prim_paths=[
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr=f"{{ENV_REGEX_NS}}/{name}",
                is_shared=True,
                track_mesh_transforms=True,
            )
            for name in (
                "Floor",
                "BoundaryNorth",
                "BoundarySouth",
                "BoundaryEast",
                "BoundaryWest",
                "InnerWall_.*",
                "RandomCylinder_.*",
            )
        ],
        data_types=["distance_to_image_plane"],
        max_distance=CAMERA_MAX_DISTANCE,
        depth_clipping_behavior="max",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=10.4775,
            horizontal_aperture=20.955,
            height=48,
            width=64,
        ),
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
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

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 0.75)),
    )


@configclass
class ActionsCfg:
    velocity_yaw = mdp.VelocityYawActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        camera = ObsTerm(
            func=mdp.normalized_depth_image,
            # TutorialEnv clips/normalizes depth at 10 m. The camera keeps its
            # wider isolation-safe clipping range, but values beyond 10 m are
            # represented identically to the direct environment.
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "max_distance": TUTORIAL_DEPTH_MAX_DISTANCE,
            },
            clip=(-1.0, 1.0),
        )
        robot_state = ObsTerm(func=mdp.robot_state, params={"asset_cfg": SceneEntityCfg("robot")})
        critic_toa = ObsTerm(
            func=mdp.privileged_toa_map,
            params={"asset_cfg": SceneEntityCfg("robot"), "crop_size": 16},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_training_maze = EventTerm(func=mdp.reset_training_maze, mode="reset")
    reset_navigation_markers = EventTerm(func=mdp.reset_navigation_markers, mode="reset")
    update_navigation_markers = EventTerm(
        func=mdp.update_navigation_markers,
        mode="interval",
        interval_range_s=(5.0 / 120.0, 5.0 / 120.0),
        is_global_time=False,
    )


@configclass
class RewardsCfg:
    goal_velocity = RewTerm(
        func=mdp.velocity_towards_goal,
        weight=TRAINING_MAZE_REWARD_WEIGHTS["goal_velocity"],
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    toa_progress = RewTerm(
        func=mdp.toa_progress,
        weight=TRAINING_MAZE_REWARD_WEIGHTS["toa_progress"],
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale": 1.0,
            "clip": 0.25,
        },
    )
    action_smoothness = RewTerm(func=mdp.action_smoothness, weight=-0.1)
    contact_force = RewTerm(
        func=mdp.contact_force_penalty,
        weight=TRAINING_MAZE_REWARD_WEIGHTS["contact_force"],
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "saturation_force": 50.0,
            # Reach full strength at 40% of the active training horizon.
            "ramp_fraction": 0.4,
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
    isolation_boundary = DoneTerm(
        func=mdp.outside_isolated_arena,
        params={"asset_cfg": SceneEntityCfg("robot"), "xy_limit": DRONE_XY_LIMIT},
    )


@configclass
class TrainingMazesEnvCfg(ManagerBasedRLEnvCfg):
    """Six-maze training task with a runtime-configurable number of drones."""

    # Aggregate transitions across all parallel environments. Training scripts
    # overwrite this from the active agent configuration.
    contact_force_penalty_max_steps: int = 70_000_000
    training_curriculum_max_steps: int = 70_000_000
    training_curriculum_stage_fractions: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75)
    training_curriculum_maze_counts: tuple[int, ...] = (1, 6, 6, 6, 6)
    training_curriculum_cylinder_counts: tuple[int, ...] = (0, 0, 10, 30, 60)

    # Save one bundle/preview per maze type (six files of each kind), never per cloned drone.
    save_global_toa_maps: bool = True
    toa_map_output_dir: str = "outputs/training_maze_toa"
    enable_navigation_markers: bool = True
    navigation_marker_max_envs: int = 16
    navigation_marker_trail_length: int = 64
    navigation_marker_trail_stride: int = 4
    navigation_marker_velocity_scale: float = 1.0
    navigation_marker_velocity_max_length: float = 4.0
    navigation_marker_velocity_height_offset: float = 0.8
    navigation_marker_prim_path: str = "/Visuals/TrainingMazesNavigation"

    scene: TrainingMazesSceneCfg = TrainingMazesSceneCfg(
        num_envs=4,
        env_spacing=ENV_SPACING,
        replicate_physics=True,
        filter_collisions=True,
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
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**23
        mdp.set_direct_tutorial_reward_weights(
            self.rewards,
            self.decimation * self.sim.dt,
            overrides=TRAINING_MAZE_REWARD_WEIGHTS,
        )
        self.viewer.eye = (28.0, 28.0, 24.0)
        self.viewer.lookat = (0.0, 0.0, 1.5)
        validate_isolation_settings(self.scene.env_spacing, CAMERA_MAX_DISTANCE, DRONE_XY_LIMIT)
