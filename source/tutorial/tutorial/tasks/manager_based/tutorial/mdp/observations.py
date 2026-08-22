from __future__ import annotations

import torch
import torch.nn.functional as F

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import quat_apply_inverse


def normalized_depth_image(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    max_distance: float = 10.0,
) -> torch.Tensor:
    """Return pooled depth images in NCHW format normalized to ``[-1, 1]``."""

    camera: TiledCamera = env.scene[sensor_cfg.name]
    depth = camera.data.output["distance_to_image_plane"].clone()
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)
    depth = depth.permute(0, 3, 1, 2)
    depth = -F.max_pool2d(-depth, kernel_size=4, stride=4)
    depth = torch.nan_to_num(depth, nan=max_distance, posinf=max_distance, neginf=0.0)
    depth = torch.clamp(depth, min=0.0, max=max_distance)
    return depth / max_distance * 2.0 - 1.0


def robot_state(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return goal, previous action, velocity, and yaw rate in the robot frame."""

    robot: Articulation = env.scene[asset_cfg.name]
    default_goal_positions = env.scene.env_origins + torch.tensor([0.0, 0.0, 2.0], device=env.device)
    goal_positions = getattr(env, "goal_positions", default_goal_positions)
    relative_goal_body = quat_apply_inverse(robot.data.root_quat_w, goal_positions - robot.data.root_pos_w)
    relative_goal_xy = relative_goal_body[:, :2].clamp(-5.0, 5.0)
    linear_velocity_body = quat_apply_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w)
    angular_velocity_body = quat_apply_inverse(robot.data.root_quat_w, robot.data.root_ang_vel_w)
    return torch.cat(
        (
            relative_goal_xy,
            env.action_manager.action,
            linear_velocity_body[:, :2],
            angular_velocity_body[:, 2:3],
        ),
        dim=-1,
    )
