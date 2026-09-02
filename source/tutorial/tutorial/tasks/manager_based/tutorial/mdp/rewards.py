from __future__ import annotations

from collections.abc import Mapping

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ..alignment import reward_manager_weights


def set_direct_tutorial_reward_weights(
    rewards_cfg,
    step_dt: float,
    overrides: Mapping[str, float] | None = None,
) -> None:
    """Compensate RewardManager dt scaling, with optional task-specific weights."""

    for term_name, manager_weight in reward_manager_weights(step_dt, overrides).items():
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
    threshold: float = 50.0,
) -> torch.Tensor:
    """Return one when any monitored robot body reaches the contact-force threshold."""

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    return torch.any(torch.norm(sensor.data.net_forces_w, dim=-1) >= threshold, dim=1).float()


def contact_force_penalty(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    saturation_force: float = 50.0,
    ramp_fraction: float = 0.2,
) -> torch.Tensor:
    """Return a force-proportional penalty that ramps up during early training.

    The unweighted value is capped at one once the maximum contact force on any
    monitored body reaches ``saturation_force``.  Its scale increases linearly
    from zero to one over ``ramp_fraction`` of the configured training steps.
    """

    if saturation_force <= 0.0:
        raise ValueError(f"saturation_force must be positive, got {saturation_force}")
    if not 0.0 < ramp_fraction <= 1.0:
        raise ValueError(f"ramp_fraction must be in (0, 1], got {ramp_fraction}")

    max_training_steps = int(getattr(env.cfg, "contact_force_penalty_max_steps", 0))
    if max_training_steps <= 0:
        raise ValueError(
            "env.cfg.contact_force_penalty_max_steps must be positive; "
            f"got {max_training_steps}"
        )

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    max_contact_force = torch.norm(sensor.data.net_forces_w, dim=-1).amax(dim=1)
    force_scale = (max_contact_force / saturation_force).clamp(max=1.0)

    # common_step_counter counts vector-environment steps. Convert it to the
    # aggregate transition count used by the training configuration.
    elapsed_steps = float(env.common_step_counter * env.num_envs)
    ramp_steps = max_training_steps * ramp_fraction
    curriculum_scale = min(elapsed_steps / ramp_steps, 1.0)
    return force_scale * curriculum_scale


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
    threshold: float = 50.0,
) -> torch.Tensor:
    """Terminate when the robot collides with the scene."""

    return collision_detected(env, sensor_cfg, threshold).bool()
