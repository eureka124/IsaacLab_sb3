"""Reset and moving-obstacle events for the unseen mixed test maze."""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation, RigidObjectCollection

from ..layout import TEST_MOVING_OBSTACLES, TEST_START_GOAL_PAIRS


def _yaw_to_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    quaternion = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    quaternion[:, 0] = torch.cos(yaw * 0.5)
    quaternion[:, 3] = torch.sin(yaw * 0.5)
    return quaternion


def _moving_tensors(env) -> tuple[torch.Tensor, ...]:
    cache_name = "_test_moving_obstacle_tensors"
    if hasattr(env, cache_name):
        return getattr(env, cache_name)

    centers = torch.tensor([item.center_xy for item in TEST_MOVING_OBSTACLES], device=env.device)
    axes = torch.tensor([item.motion_axis_xy for item in TEST_MOVING_OBSTACLES], device=env.device)
    amplitudes = torch.tensor([item.amplitude for item in TEST_MOVING_OBSTACLES], device=env.device)
    angular_frequencies = torch.tensor(
        [2.0 * math.pi / item.period_s for item in TEST_MOVING_OBSTACLES], device=env.device
    )
    cached = (centers, axes, amplitudes, angular_frequencies)
    setattr(env, cache_name, cached)
    return cached


def _write_moving_obstacles(env, env_ids: torch.Tensor, elapsed_s: torch.Tensor) -> None:
    centers, axes, amplitudes, angular_frequencies = _moving_tensors(env)
    origins = env.scene.env_origins[env_ids]
    phases = env.test_moving_phase_offsets[env_ids] + elapsed_s.unsqueeze(-1) * angular_frequencies
    offsets = torch.sin(phases) * amplitudes
    speeds = torch.cos(phases) * amplitudes * angular_frequencies
    obstacle_collection: RigidObjectCollection = env.scene["moving_obstacles"]
    pose = obstacle_collection.data.default_object_state[env_ids, :, :7].clone()
    local_xy = centers.unsqueeze(0) + offsets.unsqueeze(-1) * axes.unsqueeze(0)
    pose[:, :, :2] = origins[:, None, :2] + local_xy
    obstacle_collection.write_object_pose_to_sim(pose, env_ids=env_ids)

    velocity = torch.zeros(
        (env_ids.numel(), len(TEST_MOVING_OBSTACLES), 6), device=env.device, dtype=pose.dtype
    )
    velocity[:, :, :2] = speeds.unsqueeze(-1) * axes.unsqueeze(0)
    obstacle_collection.write_object_velocity_to_sim(velocity, env_ids=env_ids)


def reset_test_maze(env, env_ids: torch.Tensor) -> None:
    """Reset one drone onto one of the two routes shown in the test figure."""

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene["robot"]
    route_ids = torch.remainder(env_ids, len(TEST_START_GOAL_PAIRS))
    routes = torch.tensor(TEST_START_GOAL_PAIRS, device=env.device, dtype=torch.float32)
    start_xy = routes[route_ids, 0]
    goal_xy = routes[route_ids, 1]
    origins = env.scene.env_origins[env_ids]

    if not hasattr(env, "goal_positions"):
        env.goal_positions = torch.zeros((env.num_envs, 3), device=env.device)
    if not hasattr(env, "test_route_ids"):
        env.test_route_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "toa_map_ids"):
        env.toa_map_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "test_moving_phase_offsets"):
        env.test_moving_phase_offsets = torch.zeros(
            (env.num_envs, len(TEST_MOVING_OBSTACLES)), device=env.device
        )

    env.test_route_ids[env_ids] = route_ids
    env.toa_map_ids[env_ids] = route_ids
    env.goal_positions[env_ids, :2] = origins[:, :2] + goal_xy
    env.goal_positions[env_ids, 2] = 2.0
    env.test_moving_phase_offsets[env_ids] = 2.0 * math.pi * torch.rand(
        (env_ids.numel(), len(TEST_MOVING_OBSTACLES)), device=env.device
    )

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += origins
    root_state[:, :2] = origins[:, :2] + start_xy
    root_state[:, 2] = 2.0
    direction = goal_xy - start_xy
    root_state[:, 3:7] = _yaw_to_quaternion(torch.atan2(direction[:, 1], direction[:, 0]))
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    _write_moving_obstacles(env, env_ids, torch.zeros(env_ids.numel(), device=env.device))


def move_test_obstacles(env, env_ids: torch.Tensor) -> None:
    """Advance both kinematic obstacles along smooth sinusoidal trajectories."""

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0 or not hasattr(env, "test_moving_phase_offsets"):
        return
    elapsed_s = env.episode_length_buf[env_ids].to(dtype=torch.float32) * env.step_dt
    _write_moving_obstacles(env, env_ids, elapsed_s)
