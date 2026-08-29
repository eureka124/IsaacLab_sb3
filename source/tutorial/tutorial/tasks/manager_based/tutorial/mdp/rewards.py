from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ..alignment import reward_manager_weights


def set_direct_tutorial_reward_weights(rewards_cfg, step_dt: float) -> None:
    """Compensate RewardManager dt scaling to reproduce TutorialEnv rewards."""

    for term_name, manager_weight in reward_manager_weights(step_dt).items():
        term_cfg = getattr(rewards_cfg, term_name, None)
        if term_cfg is not None:
            term_cfg.weight = manager_weight


def velocity_towards_goal(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward planar velocity projected towards the current goal."""

    robot: Articulation = env.scene[asset_cfg.name]
    direction = env.goal_positions[:, :2] - robot.data.root_pos_w[:, :2]
    direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1.0e-6)
    return torch.sum(robot.data.root_lin_vel_w[:, :2] * direction, dim=-1)


def action_smoothness(env) -> torch.Tensor:
    """Penalize changes between consecutive physical commands."""

    action_term = env.action_manager.get_term("velocity_yaw")
    return torch.norm(action_term.processed_actions - action_term.previous_processed_actions, dim=-1)


def goal_reached(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.4,
) -> torch.Tensor:
    """Return one when the robot reaches the planar goal."""

    robot: Articulation = env.scene[asset_cfg.name]
    distance = torch.norm(env.goal_positions[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    return (distance <= threshold).float()


def collision_detected(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return one when any monitored robot body exceeds the contact-force threshold."""

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    return torch.any(torch.norm(sensor.data.net_forces_w, dim=-1) > threshold, dim=1).float()


def goal_reached_termination(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.4,
) -> torch.Tensor:
    """Terminate when the planar goal is reached."""

    return goal_reached(env, asset_cfg, threshold).bool()


def collision_termination(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when the robot collides with the scene."""

    return collision_detected(env, sensor_cfg, threshold).bool()
