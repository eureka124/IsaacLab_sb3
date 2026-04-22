import torch
import torch.nn.functional as F
import numpy as np
import skfmm


def _rectangle_sdf(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    inflation: float,
) -> np.ndarray:
    half_x = 0.5 * size_xy[0] + inflation
    half_y = 0.5 * size_xy[1] + inflation
    dx = np.abs(grid_x - center_xy[0]) - half_x
    dy = np.abs(grid_y - center_xy[1]) - half_y
    outside_dx = np.maximum(dx, 0.0)
    outside_dy = np.maximum(dy, 0.0)
    outside_dist = np.sqrt(outside_dx * outside_dx + outside_dy * outside_dy)
    inside_dist = np.minimum(np.maximum(dx, dy), 0.0)
    return outside_dist + inside_dist


def build_toa_map(
    goal_xy: np.ndarray,
    grid_xs: np.ndarray,
    grid_ys: np.ndarray,
    obstacles: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    robot_radius: float,
    safe_distance: float,
    slow_speed: float,
) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(grid_xs, grid_ys, indexing="xy")
    obstacle_sdf = np.full_like(grid_x, 1e6, dtype=np.float32)

    for obstacle_pos, obstacle_size in obstacles:
        obstacle_sdf = np.minimum(
            obstacle_sdf,
            _rectangle_sdf(
                grid_x,
                grid_y,
                obstacle_pos[:2],
                obstacle_size[:2],
                robot_radius,
            ).astype(np.float32),
        )

    obstacle_mask = obstacle_sdf <= 0.0
    clearance = np.maximum(obstacle_sdf, 0.0)
    speed = np.ones_like(clearance, dtype=np.float32)
    if safe_distance > 1e-6:
        slow_region = clearance <= safe_distance
        speed[slow_region] = slow_speed + (1.0 - slow_speed) * (
            clearance[slow_region] / safe_distance
        )
    speed = np.clip(speed, 1e-3, 1.0)
    speed = np.ma.array(speed, mask=obstacle_mask)

    grid_dx = float(grid_xs[1] - grid_xs[0])
    goal_radius = max(grid_dx * 1.5, 1e-3)
    phi = np.sqrt((grid_x - goal_xy[0]) ** 2 + (grid_y - goal_xy[1]) ** 2) - goal_radius
    toa_map = skfmm.travel_time(phi, speed, dx=grid_dx)

    if np.ma.isMaskedArray(toa_map):
        # toa_map.fill_value should be 0 or small for goal, but skfmm returns distance from interface
        # The masked areas are obstacles, they should have high TOA values
        toa_map = toa_map.filled(fill_value=1e6)

    return np.asarray(toa_map, dtype=np.float32)


def circle_obstacle_to_rect(
    center_xy: tuple[float, float],
    radius: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    diameter = 2.0 * float(radius)
    return (float(center_xy[0]), float(center_xy[1]), 0.0), (diameter, diameter, 1.0)


def sample_toa_from_maps(
    toa_maps: torch.Tensor,
    positions_xy: torch.Tensor,
    origins_xy: torch.Tensor,
    toa_x_min: float,
    toa_x_max: float,
    toa_y_min: float,
    toa_y_max: float,
) -> torch.Tensor:
    local_xy = positions_xy - origins_xy
    x_norm = 2.0 * (local_xy[:, 0] - toa_x_min) / (toa_x_max - toa_x_min) - 1.0
    y_norm = 2.0 * (local_xy[:, 1] - toa_y_min) / (toa_y_max - toa_y_min) - 1.0
    sample_grid = torch.stack((x_norm, y_norm), dim=-1).view(-1, 1, 1, 2)
    sampled_toa = F.grid_sample(
        toa_maps.unsqueeze(1),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled_toa[:, 0, 0, 0]


def compute_toa_gradient(
    toa_maps: torch.Tensor,
    positions_xy: torch.Tensor,
    origins_xy: torch.Tensor,
    toa_grid_xs: np.ndarray,
    toa_x_min: float,
    toa_x_max: float,
    toa_y_min: float,
    toa_y_max: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute TOA map gradient in world XY at given positions."""
    dx = float(toa_grid_xs[1] - toa_grid_xs[0])
    delta = torch.tensor([dx, 0.0], device=device, dtype=positions_xy.dtype)
    delta_y = torch.tensor([0.0, dx], device=device, dtype=positions_xy.dtype)

    toa_xp = sample_toa_from_maps(
        toa_maps,
        positions_xy + delta,
        origins_xy,
        toa_x_min,
        toa_x_max,
        toa_y_min,
        toa_y_max,
    )
    toa_xm = sample_toa_from_maps(
        toa_maps,
        positions_xy - delta,
        origins_xy,
        toa_x_min,
        toa_x_max,
        toa_y_min,
        toa_y_max,
    )
    toa_yp = sample_toa_from_maps(
        toa_maps,
        positions_xy + delta_y,
        origins_xy,
        toa_x_min,
        toa_x_max,
        toa_y_min,
        toa_y_max,
    )
    toa_ym = sample_toa_from_maps(
        toa_maps,
        positions_xy - delta_y,
        origins_xy,
        toa_x_min,
        toa_x_max,
        toa_y_min,
        toa_y_max,
    )

    grad_x = (toa_xp - toa_xm) / (2.0 * dx)
    grad_y = (toa_yp - toa_ym) / (2.0 * dx)
    return torch.stack([grad_x, grad_y], dim=-1)
