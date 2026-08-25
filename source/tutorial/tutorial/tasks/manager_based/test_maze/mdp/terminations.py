"""Termination terms for the rectangular mixed test arena."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ..layout import TEST_ARENA_HALF_X, TEST_ARENA_HALF_Y


def outside_test_arena(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_limit: float = TEST_ARENA_HALF_X - 0.5,
    y_limit: float = TEST_ARENA_HALF_Y - 0.5,
    min_height: float = 0.25,
    max_height: float = 3.75,
) -> torch.Tensor:
    """Terminate before a drone can leave its isolated rectangular arena."""

    robot: Articulation = env.scene[asset_cfg.name]
    local_position = robot.data.root_pos_w - env.scene.env_origins
    outside_xy = (torch.abs(local_position[:, 0]) > x_limit) | (torch.abs(local_position[:, 1]) > y_limit)
    outside_z = (local_position[:, 2] < min_height) | (local_position[:, 2] > max_height)
    return outside_xy | outside_z
