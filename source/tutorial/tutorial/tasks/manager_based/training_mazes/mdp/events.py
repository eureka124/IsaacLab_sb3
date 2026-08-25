"""Reset events for the six-way vectorized training-maze task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection

from ..layouts import (
    ARENA_HALF_EXTENT,
    ARENA_HEIGHT,
    INNER_WALL_LENGTH,
    MAX_INNER_WALLS,
    MAZE_LAYOUTS,
    RANDOM_CYLINDER_CLEARANCE,
    RANDOM_CYLINDER_COUNT,
    RANDOM_CYLINDER_MAX_ATTEMPTS,
    RANDOM_CYLINDER_SPECS,
    RANDOM_CYLINDER_START_GOAL_CLEARANCE,
    WALL_THICKNESS,
)
from .toa_observations import TOA_MAPS_PER_MAZE


def _yaw_to_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    quaternion = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    quaternion[:, 0] = torch.cos(yaw * 0.5)
    quaternion[:, 3] = torch.sin(yaw * 0.5)
    return quaternion


def _get_layout_tensors(env) -> tuple[torch.Tensor, ...]:
    """Create device tensors once and cache them on the environment."""

    cache_name = "_training_maze_layout_tensors"
    if hasattr(env, cache_name):
        return getattr(env, cache_name)

    num_layouts = len(MAZE_LAYOUTS)
    wall_xy = torch.zeros((num_layouts, MAX_INNER_WALLS, 2), device=env.device)
    wall_yaw = torch.zeros((num_layouts, MAX_INNER_WALLS), device=env.device)
    wall_active = torch.zeros((num_layouts, MAX_INNER_WALLS), dtype=torch.bool, device=env.device)
    starts = torch.zeros((num_layouts, 2, 2), device=env.device)
    goals = torch.zeros((num_layouts, 2, 2), device=env.device)

    for layout_index, layout in enumerate(MAZE_LAYOUTS):
        for wall_index, wall in enumerate(layout.walls):
            wall_xy[layout_index, wall_index] = torch.tensor(wall.center_xy, device=env.device)
            wall_yaw[layout_index, wall_index] = wall.yaw
            wall_active[layout_index, wall_index] = True
        for pair_index, (start, goal) in enumerate(layout.start_goal_pairs):
            starts[layout_index, pair_index] = torch.tensor(start, device=env.device)
            goals[layout_index, pair_index] = torch.tensor(goal, device=env.device)

    cached = (wall_xy, wall_yaw, wall_active, starts, goals)
    setattr(env, cache_name, cached)
    return cached


def _get_random_cylinder_specs(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed radius and height of each cylinder slot."""

    cache_name = "_training_random_cylinder_specs"
    if not hasattr(env, cache_name):
        radii = torch.tensor([radius for radius, _ in RANDOM_CYLINDER_SPECS], device=env.device)
        heights = torch.tensor([height for _, height in RANDOM_CYLINDER_SPECS], device=env.device)
        setattr(env, cache_name, (radii, heights))
    return getattr(env, cache_name)


def _sample_random_cylinder_positions(
    env,
    env_ids: torch.Tensor,
    start_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    selected_wall_xy: torch.Tensor,
    selected_wall_yaw: torch.Tensor,
    selected_wall_active: torch.Tensor,
) -> torch.Tensor:
    """Sample non-overlapping local XY positions for every fixed cylinder slot."""

    radii, _ = _get_random_cylinder_specs(env)
    num_resets = env_ids.numel()
    positions = torch.empty(
        (num_resets, RANDOM_CYLINDER_COUNT, 2), device=env.device, dtype=selected_wall_xy.dtype
    )

    vertical = torch.isclose(
        selected_wall_yaw.abs(),
        torch.full_like(selected_wall_yaw, 0.5 * torch.pi),
        atol=1.0e-6,
        rtol=0.0,
    )
    wall_half_x = torch.where(
        vertical,
        torch.full_like(selected_wall_yaw, WALL_THICKNESS * 0.5),
        torch.full_like(selected_wall_yaw, INNER_WALL_LENGTH * 0.5),
    )
    wall_half_y = torch.where(
        vertical,
        torch.full_like(selected_wall_yaw, INNER_WALL_LENGTH * 0.5),
        torch.full_like(selected_wall_yaw, WALL_THICKNESS * 0.5),
    )

    for cylinder_index in range(RANDOM_CYLINDER_COUNT):
        radius = radii[cylinder_index]
        center_limit = ARENA_HALF_EXTENT - WALL_THICKNESS * 0.5 - radius - RANDOM_CYLINDER_CLEARANCE
        unresolved = torch.ones(num_resets, dtype=torch.bool, device=env.device)

        for _ in range(RANDOM_CYLINDER_MAX_ATTEMPTS):
            if not torch.any(unresolved):
                break

            candidate = (
                torch.rand((num_resets, 2), device=env.device, dtype=positions.dtype) * 2.0 - 1.0
            ) * center_limit

            delta_to_wall = (candidate[:, None, :] - selected_wall_xy).abs()
            wall_dx = torch.clamp(delta_to_wall[..., 0] - wall_half_x, min=0.0)
            wall_dy = torch.clamp(delta_to_wall[..., 1] - wall_half_y, min=0.0)
            wall_distance_sq = wall_dx.square() + wall_dy.square()
            valid = torch.all(
                (~selected_wall_active)
                | (wall_distance_sq >= (radius + RANDOM_CYLINDER_CLEARANCE).square()),
                dim=1,
            )

            point_clearance_sq = (radius + RANDOM_CYLINDER_START_GOAL_CLEARANCE).square()
            valid &= torch.sum((candidate - start_xy).square(), dim=-1) >= point_clearance_sq
            valid &= torch.sum((candidate - goal_xy).square(), dim=-1) >= point_clearance_sq

            if cylinder_index > 0:
                previous_delta = candidate[:, None, :] - positions[:, :cylinder_index]
                minimum_distance = (
                    radius + radii[:cylinder_index] + RANDOM_CYLINDER_CLEARANCE
                )
                valid &= torch.all(
                    torch.sum(previous_delta.square(), dim=-1) >= minimum_distance.square(),
                    dim=1,
                )

            accepted = unresolved & valid
            positions[accepted, cylinder_index] = candidate[accepted]
            unresolved &= ~accepted

        if torch.any(unresolved):
            failed_env_ids = env_ids[unresolved][:8].detach().cpu().tolist()
            raise RuntimeError(
                f"Unable to place random cylinder {cylinder_index} without overlap after "
                f"{RANDOM_CYLINDER_MAX_ATTEMPTS} attempts; sample env ids: {failed_env_ids}."
            )

    return positions


def _write_random_cylinders(
    env,
    env_ids: torch.Tensor,
    origins: torch.Tensor,
    local_positions: torch.Tensor,
) -> None:
    """Write randomized positions while keeping every cylinder's geometry fixed."""

    cylinders: RigidObjectCollection = env.scene["random_cylinders"]
    expected_names = [f"cylinder_{index:02d}" for index in range(RANDOM_CYLINDER_COUNT)]
    if cylinders.object_names != expected_names:
        raise RuntimeError(
            "Random-cylinder object order does not match the fixed geometry specification: "
            f"{cylinders.object_names}."
        )

    _, heights = _get_random_cylinder_specs(env)
    pose = cylinders.data.default_object_state[env_ids, :, :7].clone()
    pose[:, :, :2] = origins[:, None, :2] + local_positions
    pose[:, :, 2] = heights.unsqueeze(0) * 0.5
    cylinders.write_object_pose_to_sim(pose, env_ids=env_ids)
    cylinders.write_object_velocity_to_sim(
        torch.zeros(
            (env_ids.numel(), RANDOM_CYLINDER_COUNT, 6),
            device=env.device,
            dtype=pose.dtype,
        ),
        env_ids=env_ids,
    )


def reset_training_maze(env, env_ids: torch.Tensor) -> None:
    """Reset drones and assign maze ``env_id % 6`` with randomized route direction.

    Each cloned environment owns one drone.  A reset samples one of the two
    start/goal pairs and swaps its direction with probability 0.5.
    """

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene["robot"]
    wall_xy, wall_yaw, wall_active, starts, goals = _get_layout_tensors(env)
    maze_ids = torch.remainder(env_ids, len(MAZE_LAYOUTS))
    pair_ids = torch.randint(0, 2, (env_ids.numel(),), device=env.device)
    reverse = torch.rand(env_ids.numel(), device=env.device) < 0.5

    selected_starts = starts[maze_ids, pair_ids]
    selected_goals = goals[maze_ids, pair_ids]
    start_xy = torch.where(reverse.unsqueeze(-1), selected_goals, selected_starts)
    goal_xy = torch.where(reverse.unsqueeze(-1), selected_starts, selected_goals)
    origins = env.scene.env_origins[env_ids]

    if not hasattr(env, "goal_positions"):
        env.goal_positions = torch.zeros((env.num_envs, 3), device=env.device)
    if not hasattr(env, "maze_ids"):
        env.maze_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "start_goal_pair_ids"):
        env.start_goal_pair_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "direction_reversed"):
        env.direction_reversed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "toa_map_ids"):
        env.toa_map_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env.maze_ids[env_ids] = maze_ids
    env.start_goal_pair_ids[env_ids] = pair_ids
    env.direction_reversed[env_ids] = reverse
    env.toa_map_ids[env_ids] = maze_ids * TOA_MAPS_PER_MAZE + pair_ids * 2 + reverse.long()
    env.goal_positions[env_ids, :2] = origins[:, :2] + goal_xy
    env.goal_positions[env_ids, 2] = 2.0

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += origins
    root_state[:, :2] = origins[:, :2] + start_xy
    root_state[:, 2] = 2.0
    direction = goal_xy - start_xy
    root_state[:, 3:7] = _yaw_to_quaternion(torch.atan2(direction[:, 1], direction[:, 0]))
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    selected_wall_xy = wall_xy[maze_ids]
    selected_wall_yaw = wall_yaw[maze_ids]
    selected_wall_active = wall_active[maze_ids]
    cylinder_positions = _sample_random_cylinder_positions(
        env,
        env_ids,
        start_xy,
        goal_xy,
        selected_wall_xy,
        selected_wall_yaw,
        selected_wall_active,
    )
    for wall_index in range(MAX_INNER_WALLS):
        wall: RigidObject = env.scene[f"inner_wall_{wall_index}"]
        wall_pose = wall.data.default_root_state[env_ids, :7].clone()
        wall_pose[:, :2] = origins[:, :2] + selected_wall_xy[:, wall_index]
        wall_pose[:, 2] = torch.where(
            selected_wall_active[:, wall_index],
            torch.full((env_ids.numel(),), ARENA_HEIGHT * 0.5, device=env.device),
            torch.full((env_ids.numel(),), -10.0, device=env.device),
        )
        wall_pose[:, 3:7] = _yaw_to_quaternion(selected_wall_yaw[:, wall_index])
        wall.write_root_pose_to_sim(wall_pose, env_ids)

    _write_random_cylinders(env, env_ids, origins, cylinder_positions)
