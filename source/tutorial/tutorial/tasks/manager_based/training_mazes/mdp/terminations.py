"""Task-specific termination terms."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ..isolation import DRONE_XY_LIMIT


def outside_isolated_arena(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_limit: float = DRONE_XY_LIMIT,
    min_height: float = 0.25,
    max_height: float = 3.75,
) -> torch.Tensor:
    """Terminate before a drone can leave its camera-isolated arena."""

    robot: Articulation = env.scene[asset_cfg.name]
    local_position = robot.data.root_pos_w - env.scene.env_origins
    outside_xy = torch.any(torch.abs(local_position[:, :2]) > xy_limit, dim=-1)
    outside_z = (local_position[:, 2] < min_height) | (local_position[:, 2] > max_height)
    return outside_xy | outside_z
