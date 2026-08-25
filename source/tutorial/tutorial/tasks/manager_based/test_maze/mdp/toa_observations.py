"""Static privileged TOA field for the unseen mixed test maze."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from tutorial.tasks.direct.tutorial.TOA import build_toa_map
from tutorial.tasks.manager_based.toa_preview import toa_to_rgb

from ..layout import (
    TEST_ARENA_HALF_X,
    TEST_ARENA_HALF_Y,
    TEST_ARENA_HEIGHT,
    TEST_START_GOAL_PAIRS,
    TEST_STATIC_OBSTACLES,
    TEST_WALL_THICKNESS,
    BoxObstacle,
)
TEST_TOA_GRID_SPACING = 0.1
TEST_TOA_WIDTH = int(round(2.0 * TEST_ARENA_HALF_X / TEST_TOA_GRID_SPACING)) + 1
TEST_TOA_HEIGHT = int(round(2.0 * TEST_ARENA_HALF_Y / TEST_TOA_GRID_SPACING)) + 1
TOA_CROP_SIZE = 16
TOA_ROBOT_RADIUS = 0.4
TOA_SAFE_DISTANCE = 2.0
TOA_SLOW_SPEED = 0.2
TOA_NORMALIZATION_MAX = 255.0


def _normalize_toa(values: torch.Tensor, normalization_max: float) -> torch.Tensor:
    """Normalize finite TOA values and obstacle sentinels to ``[-1, 1]``."""

    values = torch.nan_to_num(values, nan=normalization_max, posinf=normalization_max, neginf=0.0)
    values = values.clamp(0.0, normalization_max)
    return values / normalization_max * 2.0 - 1.0


def _boundary_obstacles() -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    return [
        (
            (0.0, TEST_ARENA_HALF_Y, TEST_ARENA_HEIGHT * 0.5),
            (2.0 * TEST_ARENA_HALF_X, TEST_WALL_THICKNESS, TEST_ARENA_HEIGHT),
        ),
        (
            (0.0, -TEST_ARENA_HALF_Y, TEST_ARENA_HEIGHT * 0.5),
            (2.0 * TEST_ARENA_HALF_X, TEST_WALL_THICKNESS, TEST_ARENA_HEIGHT),
        ),
        (
            (TEST_ARENA_HALF_X, 0.0, TEST_ARENA_HEIGHT * 0.5),
            (TEST_WALL_THICKNESS, 2.0 * TEST_ARENA_HALF_Y, TEST_ARENA_HEIGHT),
        ),
        (
            (-TEST_ARENA_HALF_X, 0.0, TEST_ARENA_HEIGHT * 0.5),
            (TEST_WALL_THICKNESS, 2.0 * TEST_ARENA_HALF_Y, TEST_ARENA_HEIGHT),
        ),
    ]


def _toa_rectangles(obstacle: BoxObstacle) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Conservatively approximate an oriented cuboid using axis-aligned cells."""

    yaw_quarter_turn = round(obstacle.yaw / (0.5 * math.pi))
    if abs(obstacle.yaw - yaw_quarter_turn * 0.5 * math.pi) < 1.0e-6:
        size_xy = obstacle.size_xy
        if yaw_quarter_turn % 2:
            size_xy = (size_xy[1], size_xy[0])
        return [
            (
                (obstacle.center_xy[0], obstacle.center_xy[1], TEST_ARENA_HEIGHT * 0.5),
                (size_xy[0], size_xy[1], TEST_ARENA_HEIGHT),
            )
        ]

    length, width = obstacle.size_xy
    cell_count = max(2, int(math.ceil(length / max(width, TEST_TOA_GRID_SPACING))))
    cell_length = length / cell_count
    cell_size = max(width, cell_length) * 1.15
    direction = (math.cos(obstacle.yaw), math.sin(obstacle.yaw))
    rectangles = []
    for cell_index in range(cell_count):
        along = -0.5 * length + (cell_index + 0.5) * cell_length
        rectangles.append(
            (
                (
                    obstacle.center_xy[0] + along * direction[0],
                    obstacle.center_xy[1] + along * direction[1],
                    TEST_ARENA_HEIGHT * 0.5,
                ),
                (cell_size, cell_size, TEST_ARENA_HEIGHT),
            )
        )
    return rectangles


def _build_test_toa_bank(device: torch.device) -> tuple[torch.Tensor, float]:
    grid_xs = np.linspace(
        -TEST_ARENA_HALF_X, TEST_ARENA_HALF_X, TEST_TOA_WIDTH, dtype=np.float32
    )
    grid_ys = np.linspace(
        -TEST_ARENA_HALF_Y, TEST_ARENA_HALF_Y, TEST_TOA_HEIGHT, dtype=np.float32
    )
    obstacles = _boundary_obstacles()
    for obstacle in TEST_STATIC_OBSTACLES:
        obstacles.extend(_toa_rectangles(obstacle))

    maps = [
        build_toa_map(
            goal_xy=np.asarray(goal, dtype=np.float32),
            grid_xs=grid_xs,
            grid_ys=grid_ys,
            obstacles=obstacles,
            robot_radius=TOA_ROBOT_RADIUS,
            safe_distance=TOA_SAFE_DISTANCE,
            slow_speed=TOA_SLOW_SPEED,
        )
        for _, goal in TEST_START_GOAL_PAIRS
    ]
    return torch.from_numpy(np.stack(maps)).to(device=device, dtype=torch.float32), TEST_TOA_GRID_SPACING


def _save_test_toa_bank(map_bank: torch.Tensor, output_dir: str) -> None:
    """Save one bundle and one two-panel preview for this environment type."""

    from PIL import Image, ImageDraw

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    maps = map_bank.detach().cpu().numpy()
    goals = np.asarray([goal for _, goal in TEST_START_GOAL_PAIRS], dtype=np.float32)
    np.savez_compressed(
        output_path / "test_maze_toa.npz",
        maps=maps,
        goals=goals,
        variant_names=np.asarray(("blue_route", "yellow_route")),
        extent=np.asarray(
            [-TEST_ARENA_HALF_X, TEST_ARENA_HALF_X, -TEST_ARENA_HALF_Y, TEST_ARENA_HALF_Y],
            dtype=np.float32,
        ),
        dynamic_obstacles_included=np.asarray(False),
    )

    preview = Image.new("RGB", (TEST_TOA_WIDTH * 2, TEST_TOA_HEIGHT), color=(255, 255, 255))
    for route_id, title in enumerate(("blue_route", "yellow_route")):
        rgb, display_max = toa_to_rgb(maps[route_id])
        panel = Image.fromarray(np.flipud(rgb), mode="RGB")
        draw = ImageDraw.Draw(panel)
        goal_x = int(
            round(
                (goals[route_id, 0] + TEST_ARENA_HALF_X)
                / (2.0 * TEST_ARENA_HALF_X)
                * (TEST_TOA_WIDTH - 1)
            )
        )
        goal_y = (TEST_TOA_HEIGHT - 1) - int(
            round((goals[route_id, 1] + TEST_ARENA_HALF_Y) / (2.0 * TEST_ARENA_HALF_Y) * (TEST_TOA_HEIGHT - 1))
        )
        draw.ellipse((goal_x - 4, goal_y - 4, goal_x + 4, goal_y + 4), fill=(255, 0, 0))
        label = f"{title} max={display_max:.1f}"
        draw.rectangle((0, 0, 132, 13), fill=(255, 255, 255))
        draw.text((2, 1), label, fill=(0, 0, 0))
        preview.paste(panel, (route_id * TEST_TOA_WIDTH, 0))
    preview.save(output_path / "test_maze_toa.png")


def _get_test_toa_cache(env, crop_size: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    cache_name = f"_test_toa_cache_{crop_size}"
    if hasattr(env, cache_name):
        return getattr(env, cache_name)

    map_bank, grid_spacing = _build_test_toa_bank(env.device)
    cfg = getattr(env, "cfg", None)
    if bool(getattr(cfg, "save_global_toa_maps", False)):
        _save_test_toa_bank(map_bank, getattr(cfg, "toa_map_output_dir", "outputs/test_maze_toa"))

    half_extent = crop_size * grid_spacing * 0.5
    coordinates = torch.linspace(
        -half_extent + grid_spacing * 0.5,
        half_extent - grid_spacing * 0.5,
        steps=crop_size,
        device=env.device,
    )
    grid_x, grid_y = torch.meshgrid(coordinates, coordinates, indexing="xy")
    crop_grid_body = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
    cached = (map_bank, crop_grid_body, grid_spacing)
    setattr(env, cache_name, cached)
    return cached


def _sample_test_map_bank(
    map_bank: torch.Tensor,
    map_ids: torch.Tensor,
    points_xy: torch.Tensor,
    grid_spacing: float,
) -> torch.Tensor:
    height, width = map_bank.shape[-2:]
    fractional_x = ((points_xy[..., 0] + TEST_ARENA_HALF_X) / grid_spacing).clamp(0.0, width - 1.0)
    fractional_y = ((points_xy[..., 1] + TEST_ARENA_HALF_Y) / grid_spacing).clamp(0.0, height - 1.0)
    x0 = torch.floor(fractional_x).long()
    y0 = torch.floor(fractional_y).long()
    x1 = torch.clamp(x0 + 1, max=width - 1)
    y1 = torch.clamp(y0 + 1, max=height - 1)
    weight_x = fractional_x - x0
    weight_y = fractional_y - y0
    flattened_bank = map_bank.reshape(-1)
    map_offsets = map_ids.view(-1, 1) * (height * width)

    def gather(y_indices: torch.Tensor, x_indices: torch.Tensor) -> torch.Tensor:
        return flattened_bank[map_offsets + y_indices * width + x_indices]

    value_00 = gather(y0, x0)
    value_10 = gather(y0, x1)
    value_01 = gather(y1, x0)
    value_11 = gather(y1, x1)
    value_0 = value_00 + (value_10 - value_00) * weight_x
    value_1 = value_01 + (value_11 - value_01) * weight_x
    return value_0 + (value_1 - value_0) * weight_y


def privileged_test_toa_map(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    crop_size: int = TOA_CROP_SIZE,
    normalization_max: float = TOA_NORMALIZATION_MAX,
) -> torch.Tensor:
    """Return a robot-aligned local crop from the static test-scene TOA map."""

    robot: Articulation = env.scene[asset_cfg.name]
    map_ids = getattr(env, "toa_map_ids", torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
    map_bank, crop_grid_body, grid_spacing = _get_test_toa_cache(env, crop_size)
    robot_position_local = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    w, x, y, z = torch.unbind(robot.data.root_quat_w, dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = torch.cos(yaw).unsqueeze(-1)
    sin_yaw = torch.sin(yaw).unsqueeze(-1)
    body_x = crop_grid_body[:, 0].unsqueeze(0)
    body_y = crop_grid_body[:, 1].unsqueeze(0)
    sample_x = body_x * cos_yaw - body_y * sin_yaw + robot_position_local[:, 0:1]
    sample_y = body_x * sin_yaw + body_y * cos_yaw + robot_position_local[:, 1:2]
    sampled = _sample_test_map_bank(
        map_bank, map_ids, torch.stack((sample_x, sample_y), dim=-1), grid_spacing
    )
    return _normalize_toa(sampled, normalization_max).reshape(env.num_envs, 1, crop_size, crop_size)
