from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject


def _yaw_to_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    quaternion = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    quaternion[:, 0] = torch.cos(yaw * 0.5)
    quaternion[:, 3] = torch.sin(yaw * 0.5)
    return quaternion


def reset_robot_goal_and_obstacles(
    env,
    env_ids: torch.Tensor,
    obstacle_count: int,
    grid_size: int = 10,
    cell_extent: float = 20.0,
) -> None:
    """Reset the drone at the origin, sample a corner goal, and randomize cylinders on a grid."""

    robot: Articulation = env.scene["robot"]
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    num_resets = env_ids.numel()
    if num_resets == 0:
        return

    if not hasattr(env, "goal_positions"):
        env.goal_positions = torch.zeros(env.num_envs, 3, device=env.device)

    default_root_state = robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env.scene.env_origins[env_ids]
    goals_xy = torch.tensor(
        ((15.0, 15.0), (-15.0, 15.0), (-15.0, -15.0), (15.0, -15.0)),
        device=env.device,
        dtype=default_root_state.dtype,
    )
    goal_xy = goals_xy[torch.randint(0, 4, (num_resets,), device=env.device)]
    default_root_state[:, 0:2] = env.scene.env_origins[env_ids, 0:2]
    default_root_state[:, 2] = 2.0
    env.goal_positions[env_ids, :2] = goal_xy + env.scene.env_origins[env_ids, :2]
    env.goal_positions[env_ids, 2] = 2.0

    direction = env.goal_positions[env_ids, :2] - default_root_state[:, :2]
    yaw = torch.atan2(direction[:, 1], direction[:, 0])
    default_root_state[:, 3:7] = _yaw_to_quaternion(yaw)
    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

    num_cells = grid_size**2
    if obstacle_count > num_cells:
        raise ValueError(f"Cannot place {obstacle_count} obstacles in a {grid_size}x{grid_size} grid.")
    cell_size = cell_extent / grid_size
    permutations = torch.argsort(torch.rand(num_resets, num_cells, device=env.device), dim=1)

    for obstacle_index in range(obstacle_count):
        obstacle: RigidObject = env.scene[f"obstacle_{obstacle_index}"]
        grid_index = permutations[:, obstacle_index]
        row = grid_index // grid_size
        column = grid_index % grid_size
        center_x = (row - (grid_size / 2.0 - 0.5)) * cell_size
        center_y = (column - (grid_size / 2.0 - 0.5)) * cell_size
        radius = float(obstacle.cfg.spawn.radius)
        max_offset = max(cell_size * 0.5 - radius, 0.0)
        obstacle_x = center_x + (torch.rand(num_resets, device=env.device) * 2.0 - 1.0) * max_offset
        obstacle_y = center_y + (torch.rand(num_resets, device=env.device) * 2.0 - 1.0) * max_offset

        obstacle_state = obstacle.data.default_root_state[env_ids].clone()
        obstacle_state[:, 0] = obstacle_x + env.scene.env_origins[env_ids, 0]
        obstacle_state[:, 1] = obstacle_y + env.scene.env_origins[env_ids, 1]
        obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
        obstacle.write_root_velocity_to_sim(obstacle_state[:, 7:], env_ids)
