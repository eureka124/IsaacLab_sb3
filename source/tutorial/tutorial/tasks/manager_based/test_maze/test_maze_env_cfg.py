"""ManagerBased evaluation environment reconstructed from ``测试环境.jpg``."""

from __future__ import annotations

import math

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
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from tutorial.assets.hummingbird import HUMMINGBIRD_CFG

from ..tutorial.alignment import TUTORIAL_DEPTH_MAX_DISTANCE
from . import mdp
from .isolation import CAMERA_MAX_DISTANCE, ENV_SPACING, validate_isolation_settings
from .layout import (
    TEST_ARENA_HALF_X,
    TEST_ARENA_HALF_Y,
    TEST_ARENA_HEIGHT,
    TEST_MOVING_OBSTACLES,
    TEST_STATIC_OBSTACLES,
    TEST_WALL_THICKNESS,
)


def _fixed_cuboid(
    size: tuple[float, float, float],
    color: tuple[float, float, float] = (0.72, 0.74, 0.78),
) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
    )


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _obstacle_cfg(
    name: str,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    yaw: float,
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=_fixed_cuboid((*size_xy, TEST_ARENA_HEIGHT), color),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(*center_xy, TEST_ARENA_HEIGHT * 0.5),
            rot=_yaw_quaternion(yaw),
        ),
    )


@configclass
class TestMazeSceneCfg(InteractiveSceneCfg):
    """One drone and one independent copy of the mixed test maze per clone."""

    robot: ArticulationCfg = HUMMINGBIRD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=_fixed_cuboid((2.0 * TEST_ARENA_HALF_X, 2.0 * TEST_ARENA_HALF_Y, 0.1), (0.86, 0.86, 0.86)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    boundary_north = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundaryNorth",
        spawn=_fixed_cuboid((2.0 * TEST_ARENA_HALF_X, TEST_WALL_THICKNESS, TEST_ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, TEST_ARENA_HALF_Y, TEST_ARENA_HEIGHT * 0.5)),
    )
    boundary_south = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundarySouth",
        spawn=_fixed_cuboid((2.0 * TEST_ARENA_HALF_X, TEST_WALL_THICKNESS, TEST_ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -TEST_ARENA_HALF_Y, TEST_ARENA_HEIGHT * 0.5)),
    )
    boundary_east = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundaryEast",
        spawn=_fixed_cuboid((TEST_WALL_THICKNESS, 2.0 * TEST_ARENA_HALF_Y, TEST_ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(TEST_ARENA_HALF_X, 0.0, TEST_ARENA_HEIGHT * 0.5)),
    )
    boundary_west = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoundaryWest",
        spawn=_fixed_cuboid((TEST_WALL_THICKNESS, 2.0 * TEST_ARENA_HALF_Y, TEST_ARENA_HEIGHT)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-TEST_ARENA_HALF_X, 0.0, TEST_ARENA_HEIGHT * 0.5)),
    )

    static_obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            f"static_obstacle_{index}": _obstacle_cfg(
                f"StaticObstacle_{index}",
                obstacle.center_xy,
                obstacle.size_xy,
                obstacle.yaw,
                (0.72, 0.74, 0.78),
            )
            for index, obstacle in enumerate(TEST_STATIC_OBSTACLES)
        }
    )
    moving_obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            f"moving_obstacle_{index}": _obstacle_cfg(
                f"MovingObstacle_{index}",
                obstacle.center_xy,
                obstacle.size_xy,
                obstacle.yaw,
                (0.85, 0.15, 0.12),
            )
            for index, obstacle in enumerate(TEST_MOVING_OBSTACLES)
        }
    )

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
            clipping_range=(0.1, CAMERA_MAX_DISTANCE),
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
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=4000.0, color=(1.0, 1.0, 0.85)),
    )


@configclass
class TestActionsCfg:
    velocity_yaw = mdp.VelocityYawActionCfg(asset_name="robot")


@configclass
class TestObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        camera = ObsTerm(
            func=mdp.normalized_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "max_distance": TUTORIAL_DEPTH_MAX_DISTANCE,
            },
            clip=(-1.0, 1.0),
        )
        robot_state = ObsTerm(func=mdp.robot_state, params={"asset_cfg": SceneEntityCfg("robot")})
        critic_toa = ObsTerm(
            func=mdp.privileged_test_toa_map,
            params={"asset_cfg": SceneEntityCfg("robot"), "crop_size": 16},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TestEventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_test_maze = EventTerm(func=mdp.reset_test_maze, mode="reset")
    reset_navigation_markers = EventTerm(func=mdp.reset_navigation_markers, mode="reset")
    move_obstacles = EventTerm(
        func=mdp.move_test_obstacles,
        mode="interval",
        interval_range_s=(5.0 / 120.0, 5.0 / 120.0),
        is_global_time=False,
    )
    update_navigation_markers = EventTerm(
        func=mdp.update_navigation_markers,
        mode="interval",
        interval_range_s=(5.0 / 120.0, 5.0 / 120.0),
        is_global_time=False,
    )


@configclass
class TestRewardsCfg:
    # Keep evaluation rewards identical to training for comparable metrics.
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
class TestTerminationsCfg:
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
        func=mdp.outside_test_arena,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TestMazeEnvCfg(ManagerBasedRLEnvCfg):
    """Unseen mixed maze with two routes and two moving obstacles."""

    # Aggregate transitions across all parallel environments. Training scripts
    # overwrite this from the active agent configuration.
    contact_force_penalty_max_steps: int = 70_000_000

    save_global_toa_maps: bool = True
    toa_map_output_dir: str = "outputs/test_maze_toa"
    enable_navigation_markers: bool = True
    navigation_marker_max_envs: int = 16
    navigation_marker_trail_length: int = 64
    navigation_marker_trail_stride: int = 4
    navigation_marker_velocity_scale: float = 1.0
    navigation_marker_velocity_max_length: float = 4.0
    navigation_marker_velocity_height_offset: float = 0.8
    navigation_marker_prim_path: str = "/Visuals/TestMazeNavigation"

    scene: TestMazeSceneCfg = TestMazeSceneCfg(
        num_envs=4,
        env_spacing=ENV_SPACING,
        replicate_physics=True,
        filter_collisions=True,
    )
    actions: TestActionsCfg = TestActionsCfg()
    observations: TestObservationsCfg = TestObservationsCfg()
    events: TestEventCfg = TestEventCfg()
    rewards: TestRewardsCfg = TestRewardsCfg()
    terminations: TestTerminationsCfg = TestTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 5
        self.episode_length_s = 40.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = False
        mdp.set_direct_tutorial_reward_weights(self.rewards, self.decimation * self.sim.dt)
        self.viewer.eye = (32.0, 30.0, 28.0)
        self.viewer.lookat = (0.0, 0.0, 1.5)
        validate_isolation_settings(
            self.scene.env_spacing,
            CAMERA_MAX_DISTANCE,
            TEST_ARENA_HALF_X - 0.5,
            TEST_ARENA_HALF_Y - 0.5,
        )
