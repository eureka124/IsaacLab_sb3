# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from tutorial.assets.hummingbird import HUMMINGBIRD_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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


# 定义迷宫障碍物
def create_maze_obstacles(
    opening_size=8.0,  # 实际净开口宽度
    wall_offset=12.0,  # 墙体中心线距离原点
    wall_thickness=1.0,
    wall_height=4.0,
):
    """
    创建四面带开口的方形围墙。
    """
    a = opening_size
    b = wall_thickness
    c = wall_offset

    wall_length = c - a / 2.0 - b / 2.0

    half_wall_height = wall_height / 2.0
    wall_length_position = c / 2.0 + a / 4.0 - b / 4.0

    maze_obstacles = [
        # 第一象限
        ((wall_length_position, wall_offset, half_wall_height), (wall_length, wall_thickness, wall_height)),
        ((wall_offset, wall_length_position, half_wall_height), (wall_thickness, wall_length, wall_height)),
        # 第二象限
        ((-wall_length_position, wall_offset, half_wall_height), (wall_length, wall_thickness, wall_height)),
        ((-wall_offset, wall_length_position, half_wall_height), (wall_thickness, wall_length, wall_height)),
        # 第三象限
        ((-wall_length_position, -wall_offset, half_wall_height), (wall_length, wall_thickness, wall_height)),
        ((-wall_offset, -wall_length_position, half_wall_height), (wall_thickness, wall_length, wall_height)),
        # 第四象限
        ((wall_length_position, -wall_offset, half_wall_height), (wall_length, wall_thickness, wall_height)),
        ((wall_offset, -wall_length_position, half_wall_height), (wall_thickness, wall_length, wall_height)),
    ]
    print("=============================maze obstacles information================================")
    print(maze_obstacles)
    return maze_obstacles


maze_obstacles = create_maze_obstacles(
    opening_size=8.0,
    wall_offset=12.0,
    wall_thickness=1.0,
    wall_height=4.0,
)

for i, (pos, size) in enumerate(maze_obstacles):
    obstacle_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/maze_Obstacle_{i}",
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
    setattr(MySceneCfg, f"maze_Obstacle_{i}", obstacle_cfg)


# 从 USD 文件加载 U 形障碍物。XY 和朝向在每次启动时随机生成，Z 坐标固定。
NUM_U_OBSTACLES = 20
U_OBSTACLE_XY_RANGE = (-10.0, 10.0)
U_OBSTACLE_ORIGIN_EXCLUSION_HALF_SIZE = 2.0
U_OBSTACLE_Z = 0.0
U_OBSTACLE_YAWS = (np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
U_OBSTACLE_USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets/usd/u_obstacle.usd"))
# 与 USD 模型的俯视轮廓一致：左右侧墙和后墙，开口朝局部 -Y。
U_OBSTACLE_FOOTPRINT_RECTS = (
    ((-0.95, 0.0), (0.1, 0.8)),
    ((0.95, 0.0), (0.1, 0.8)),
    ((0.0, 0.45), (2.0, 0.1)),
)

u_obstacle_rng = np.random.default_rng()
u_obstacle_positions = []
u_obstacle_yaws = []
for i in range(NUM_U_OBSTACLES):
    while True:
        x, y = u_obstacle_rng.uniform(*U_OBSTACLE_XY_RANGE, size=2)
        if not (abs(x) <= U_OBSTACLE_ORIGIN_EXCLUSION_HALF_SIZE and abs(y) <= U_OBSTACLE_ORIGIN_EXCLUSION_HALF_SIZE):
            break
    position = (float(x), float(y), U_OBSTACLE_Z)
    yaw = float(u_obstacle_rng.choice(U_OBSTACLE_YAWS))
    orientation = (float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0)))
    u_obstacle_positions.append(position)
    u_obstacle_yaws.append(yaw)
    u_obstacle_cfg = AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/U_Obstacle_{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=U_OBSTACLE_USD_PATH),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=orientation),
    )
    setattr(MySceneCfg, f"U_Obstacle_{i}", u_obstacle_cfg)


def create_u_obstacle_toa_rects(position, yaw):
    """Create the three axis-aligned TOA rectangles for a quarter-turn U obstacle."""
    cos_yaw = int(round(np.cos(yaw)))
    sin_yaw = int(round(np.sin(yaw)))
    swap_size_axes = abs(sin_yaw) == 1
    rectangles = []

    for offset_xy, size_xy in U_OBSTACLE_FOOTPRINT_RECTS:
        rotated_offset_x = cos_yaw * offset_xy[0] - sin_yaw * offset_xy[1]
        rotated_offset_y = sin_yaw * offset_xy[0] + cos_yaw * offset_xy[1]
        rotated_size_xy = (size_xy[1], size_xy[0]) if swap_size_axes else size_xy
        rectangles.append(
            (
                (position[0] + rotated_offset_x, position[1] + rotated_offset_y, U_OBSTACLE_Z),
                (rotated_size_xy[0], rotated_size_xy[1], 4.0),
            )
        )

    return rectangles


# TOA 使用轴对齐矩形，将每个 U 形模型拆成三个与其墙体对应的矩形。
u_obstacle_toa_obstacles = [
    rectangle
    for position, yaw in zip(u_obstacle_positions, u_obstacle_yaws)
    for rectangle in create_u_obstacle_toa_rects(position, yaw)
]


# 随机生成静止圆柱障碍物；数量范围为 0 到 (OBSTACLE_GRID_SIZE ** 2 - 2)。
OBSTACLE_GRID_SIZE = 10
NUM_OBSTACLES = 80
OBSTACLE_RANDOM_SEED = 42  # 设为 None 可在每次启动时生成不同布局


def generate_obstacle_positions(
    num_obstacles: int,
    seed: Optional[int] = None,
    xy_range: tuple[float, float] = (-10.0, 10.0),
    radius_range: tuple[float, float] = (0.2, 0.5),
    height_range: tuple[float, float] = (3.0, 4.0),
    min_clearance: float = 0.1,
    origin_clearance: float = 3.0,
):
    """Generate non-overlapping cylinder parameters as ``(position, radius, height)``."""
    max_obstacles = OBSTACLE_GRID_SIZE**2 - 2  # reserve grid cells for the robot and goal
    if not 0 <= num_obstacles <= max_obstacles:
        raise ValueError(f"num_obstacles must be between 0 and {max_obstacles}, got {num_obstacles}.")
    if xy_range[0] >= xy_range[1] or radius_range[0] <= 0.0 or radius_range[0] > radius_range[1]:
        raise ValueError("Invalid obstacle position or radius range.")
    if height_range[0] <= 0.0 or height_range[0] > height_range[1]:
        raise ValueError("Invalid obstacle height range.")

    rng = np.random.default_rng(seed)
    generated = []
    max_attempts_per_obstacle = 10_000

    for obstacle_index in range(num_obstacles):
        radius = float(rng.uniform(*radius_range))
        height = float(rng.uniform(*height_range))

        for _ in range(max_attempts_per_obstacle):
            x = float(rng.uniform(*xy_range))
            y = float(rng.uniform(*xy_range))

            if np.hypot(x, y) < origin_clearance + radius:
                continue
            if any(
                np.hypot(x - position[0], y - position[1]) < radius + other_radius + min_clearance
                for position, other_radius, _ in generated
            ):
                continue

            generated.append(((x, y, height / 2.0), radius, height))
            break
        else:
            raise RuntimeError(f"Unable to place obstacle {obstacle_index}; reduce NUM_OBSTACLES or min_clearance.")

    return generated


# obstacle_positions 是场景创建和 TutorialEnv 障碍物处理的唯一数量来源。
obstacle_positions = generate_obstacle_positions(NUM_OBSTACLES, seed=OBSTACLE_RANDOM_SEED)

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
    # critic TOA crop size (pixels)
    critic_toa_crop_size: int = 16

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
        "critic-toa": spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, critic_toa_crop_size, critic_toa_crop_size),
            dtype=np.float32,
        ),
    }
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = MySceneCfg(num_envs=4, env_spacing=30.0, replicate_physics=True)
