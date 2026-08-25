"""Privileged TOA observations for the six training mazes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from tutorial.tasks.direct.tutorial.TOA import build_toa_map
from tutorial.tasks.manager_based.toa_preview import toa_to_rgb

from ..layouts import (
    ARENA_HALF_EXTENT,
    ARENA_HEIGHT,
    INNER_WALL_LENGTH,
    MAZE_LAYOUTS,
    WALL_THICKNESS,
)


TOA_GRID_SIZE = 401
TOA_CROP_SIZE = 16
TOA_ROBOT_RADIUS = 0.4
TOA_SAFE_DISTANCE = 2.0
TOA_SLOW_SPEED = 0.2
TOA_NORMALIZATION_MAX = 255.0
TOA_MAPS_PER_MAZE = 4  # two pairs, each with normal and reversed direction


def _boundary_obstacles() -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Return axis-aligned rectangles for the four arena walls."""

    diameter = 2.0 * ARENA_HALF_EXTENT
    return [
        ((0.0, ARENA_HALF_EXTENT, ARENA_HEIGHT * 0.5), (diameter, WALL_THICKNESS, ARENA_HEIGHT)),
        ((0.0, -ARENA_HALF_EXTENT, ARENA_HEIGHT * 0.5), (diameter, WALL_THICKNESS, ARENA_HEIGHT)),
        ((ARENA_HALF_EXTENT, 0.0, ARENA_HEIGHT * 0.5), (WALL_THICKNESS, diameter, ARENA_HEIGHT)),
        ((-ARENA_HALF_EXTENT, 0.0, ARENA_HEIGHT * 0.5), (WALL_THICKNESS, diameter, ARENA_HEIGHT)),
    ]


def _layout_obstacles(layout) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Convert a maze layout's quarter-turn walls to TOA rectangles."""

    obstacles = _boundary_obstacles()
    for wall in layout.walls:
        vertical = abs(abs(wall.yaw) - 0.5 * np.pi) < 1.0e-6
        size_xy = (
            (WALL_THICKNESS, INNER_WALL_LENGTH)
            if vertical
            else (INNER_WALL_LENGTH, WALL_THICKNESS)
        )
        obstacles.append(
            (
                (wall.center_xy[0], wall.center_xy[1], ARENA_HEIGHT * 0.5),
                (size_xy[0], size_xy[1], ARENA_HEIGHT),
            )
        )
    return obstacles


def _build_toa_bank(device: torch.device) -> tuple[torch.Tensor, float]:
    """Precompute the 24 static global maps and copy the small bank to the simulation device."""

    grid_coordinates = np.linspace(
        -ARENA_HALF_EXTENT,
        ARENA_HALF_EXTENT,
        TOA_GRID_SIZE,
        dtype=np.float32,
    )
    maps = []
    for layout in MAZE_LAYOUTS:
        obstacles = _layout_obstacles(layout)
        for start, goal in layout.start_goal_pairs:
            for goal_xy in (goal, start):
                maps.append(
                    build_toa_map(
                        goal_xy=np.asarray(goal_xy, dtype=np.float32),
                        grid_xs=grid_coordinates,
                        grid_ys=grid_coordinates,
                        obstacles=obstacles,
                        robot_radius=TOA_ROBOT_RADIUS,
                        safe_distance=TOA_SAFE_DISTANCE,
                        slow_speed=TOA_SLOW_SPEED,
                    )
                )

    expected_map_count = len(MAZE_LAYOUTS) * TOA_MAPS_PER_MAZE
    if len(maps) != expected_map_count:
        raise RuntimeError(f"Expected {expected_map_count} TOA maps, built {len(maps)}.")
    map_bank = torch.from_numpy(np.stack(maps, axis=0)).to(device=device, dtype=torch.float32)
    grid_spacing = float(grid_coordinates[1] - grid_coordinates[0])
    return map_bank, grid_spacing


def _save_toa_bank_by_maze(map_bank: torch.Tensor, output_dir: str) -> None:
    """Save exactly one data bundle and one preview image for each maze type."""

    from PIL import Image, ImageDraw

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    maps_cpu = map_bank.detach().cpu().numpy()
    variant_titles = ("pair_1_goal", "pair_1_reversed", "pair_2_goal", "pair_2_reversed")

    for maze_id, layout in enumerate(MAZE_LAYOUTS):
        first_map = maze_id * TOA_MAPS_PER_MAZE
        maze_maps = maps_cpu[first_map : first_map + TOA_MAPS_PER_MAZE]
        goals = np.asarray(
            [
                layout.start_goal_pairs[0][1],
                layout.start_goal_pairs[0][0],
                layout.start_goal_pairs[1][1],
                layout.start_goal_pairs[1][0],
            ],
            dtype=np.float32,
        )
        np.savez_compressed(
            output_path / f"{layout.name}_toa.npz",
            maps=maze_maps,
            goals=goals,
            variant_names=np.asarray(variant_titles),
            extent=np.asarray(
                [-ARENA_HALF_EXTENT, ARENA_HALF_EXTENT, -ARENA_HALF_EXTENT, ARENA_HALF_EXTENT],
                dtype=np.float32,
            ),
        )

        panel_size = TOA_GRID_SIZE
        panel_last_index = panel_size - 1
        preview = Image.new("RGB", (panel_size * 2, panel_size * 2), color=(255, 255, 255))
        for variant_id, title in enumerate(variant_titles):
            rgb, display_max = toa_to_rgb(maze_maps[variant_id])
            panel = Image.fromarray(np.flipud(rgb), mode="RGB")
            draw = ImageDraw.Draw(panel)
            goal_x = int(
                round(
                    (goals[variant_id, 0] + ARENA_HALF_EXTENT)
                    / (2.0 * ARENA_HALF_EXTENT)
                    * panel_last_index
                )
            )
            goal_y = panel_last_index - int(
                round(
                    (goals[variant_id, 1] + ARENA_HALF_EXTENT)
                    / (2.0 * ARENA_HALF_EXTENT)
                    * panel_last_index
                )
            )
            draw.ellipse((goal_x - 4, goal_y - 4, goal_x + 4, goal_y + 4), fill=(255, 0, 0))
            label = f"{title} max={display_max:.1f}"
            draw.rectangle((0, 0, 148, 13), fill=(255, 255, 255))
            draw.text((2, 1), label, fill=(0, 0, 0))
            preview.paste(panel, ((variant_id % 2) * panel_size, (variant_id // 2) * panel_size))
        preview.save(output_path / f"{layout.name}_toa.png")


def _get_toa_bank_cache(env) -> tuple[torch.Tensor, float]:
    """Return the map bank shared by all cloned environments."""

    cache_name = "_training_toa_bank_cache"
    if not hasattr(env, cache_name):
        map_bank_cache = _build_toa_bank(env.device)
        setattr(env, cache_name, map_bank_cache)
        cfg = getattr(env, "cfg", None)
        if bool(getattr(cfg, "save_global_toa_maps", False)):
            _save_toa_bank_by_maze(
                map_bank_cache[0],
                getattr(cfg, "toa_map_output_dir", "outputs/training_maze_toa"),
            )
    return getattr(env, cache_name)


def _get_toa_cache(env, crop_size: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the cached global map bank and body-frame crop grid."""

    cache_name = f"_training_toa_cache_{crop_size}"
    if hasattr(env, cache_name):
        return getattr(env, cache_name)

    map_bank, grid_spacing = _get_toa_bank_cache(env)
    half_extent = crop_size * grid_spacing * 0.5
    coordinates = torch.linspace(
        -half_extent + grid_spacing * 0.5,
        half_extent - grid_spacing * 0.5,
        steps=crop_size,
        device=env.device,
        dtype=torch.float32,
    )
    grid_x, grid_y = torch.meshgrid(coordinates, coordinates, indexing="xy")
    crop_grid_body = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
    cached = (map_bank, crop_grid_body, grid_spacing)
    setattr(env, cache_name, cached)
    return cached


def _bilinear_sample_map_bank(
    map_bank: torch.Tensor,
    map_ids: torch.Tensor,
    points_xy: torch.Tensor,
    grid_spacing: float,
) -> torch.Tensor:
    """Sample different maps per environment without materializing a map per drone."""

    grid_size = map_bank.shape[-1]
    fractional_x = (points_xy[..., 0] + ARENA_HALF_EXTENT) / grid_spacing
    fractional_y = (points_xy[..., 1] + ARENA_HALF_EXTENT) / grid_spacing
    fractional_x = fractional_x.clamp(0.0, grid_size - 1.0)
    fractional_y = fractional_y.clamp(0.0, grid_size - 1.0)

    x0 = torch.floor(fractional_x).long()
    y0 = torch.floor(fractional_y).long()
    x1 = torch.clamp(x0 + 1, max=grid_size - 1)
    y1 = torch.clamp(y0 + 1, max=grid_size - 1)
    weight_x = fractional_x - x0
    weight_y = fractional_y - y0

    flattened_bank = map_bank.reshape(-1)
    map_offsets = map_ids.view(-1, 1) * (grid_size * grid_size)

    def gather(y_indices: torch.Tensor, x_indices: torch.Tensor) -> torch.Tensor:
        return flattened_bank[map_offsets + y_indices * grid_size + x_indices]

    value_00 = gather(y0, x0)
    value_10 = gather(y0, x1)
    value_01 = gather(y1, x0)
    value_11 = gather(y1, x1)
    value_0 = value_00 + (value_10 - value_00) * weight_x
    value_1 = value_01 + (value_11 - value_01) * weight_x
    return value_0 + (value_1 - value_0) * weight_y


def _normalize_toa(values: torch.Tensor, normalization_max: float) -> torch.Tensor:
    """Normalize finite TOA values and obstacle sentinels to ``[-1, 1]``."""

    values = torch.nan_to_num(values, nan=normalization_max, posinf=normalization_max, neginf=0.0)
    values = values.clamp(0.0, normalization_max)
    return values / normalization_max * 2.0 - 1.0


def privileged_toa_map(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    crop_size: int = TOA_CROP_SIZE,
    normalization_max: float = TOA_NORMALIZATION_MAX,
) -> torch.Tensor:
    """Return a robot-aligned local TOA crop exclusively consumed by the critic.

    The result has shape ``(num_envs, 1, crop_size, crop_size)`` and is
    normalized to ``[-1, 1]``. Map selection follows the maze, pair, and
    direction sampled by :func:`reset_training_maze`.
    """

    robot: Articulation = env.scene[asset_cfg.name]
    if hasattr(env, "toa_map_ids"):
        map_ids = env.toa_map_ids
    else:
        map_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    map_bank, crop_grid_body, grid_spacing = _get_toa_cache(env, crop_size)

    robot_position_local = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    quaternion = robot.data.root_quat_w
    w, x, y, z = torch.unbind(quaternion, dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = torch.cos(yaw).unsqueeze(-1)
    sin_yaw = torch.sin(yaw).unsqueeze(-1)

    body_x = crop_grid_body[:, 0].unsqueeze(0)
    body_y = crop_grid_body[:, 1].unsqueeze(0)
    sample_x = body_x * cos_yaw - body_y * sin_yaw + robot_position_local[:, 0:1]
    sample_y = body_x * sin_yaw + body_y * cos_yaw + robot_position_local[:, 1:2]
    sample_points = torch.stack((sample_x, sample_y), dim=-1)

    sampled = _bilinear_sample_map_bank(map_bank, map_ids, sample_points, grid_spacing)
    normalized = _normalize_toa(sampled, normalization_max)
    return normalized.reshape(env.num_envs, 1, crop_size, crop_size)
