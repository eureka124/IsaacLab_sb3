"""Reset events for the six-way vectorized training-maze task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject

from ..layouts import ARENA_HEIGHT, MAX_INNER_WALLS, MAZE_LAYOUTS
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
