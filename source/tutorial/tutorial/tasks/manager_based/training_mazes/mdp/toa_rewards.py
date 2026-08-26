"""Reward terms based on temporal changes in the training-maze TOA field."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .toa_observations import _bilinear_sample_map_bank, _get_toa_bank_cache


def toa_progress(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
    clip: float = 0.0,
) -> torch.Tensor:
    """Reward decrease in the TOA value between consecutive policy steps.

    The returned progress is ``previous_toa - current_toa``.  Therefore,
    moving toward the goal (lower TOA) produces a positive reward.  The first
    step after reset returns zero because there is no previous frame from the
    new episode.  The TOA field is the same static field used by the critic and
    intentionally does not include the randomized cylinders.

    Args:
        env: Manager-based RL environment.
        asset_cfg: Robot entity whose position is sampled from the TOA field.
        scale: Optional multiplier applied after computing the difference.
        clip: Symmetric clipping bound.  A non-positive value disables clipping.
    """

    robot: Articulation = env.scene[asset_cfg.name]
    map_bank, grid_spacing = _get_toa_bank_cache(env)
    map_ids = getattr(
        env,
        "toa_map_ids",
        torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
    )
    local_position = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    current_toa = _bilinear_sample_map_bank(
        map_bank,
        map_ids,
        local_position.unsqueeze(1),
        grid_spacing,
    ).squeeze(-1)

    previous_toa = getattr(env, "_toa_reward_previous", None)
    if previous_toa is None or previous_toa.shape != current_toa.shape:
        previous_toa = torch.full_like(current_toa, torch.nan)
        setattr(env, "_toa_reward_previous", previous_toa)

    valid = torch.isfinite(previous_toa) & torch.isfinite(current_toa)
    progress = torch.where(valid, previous_toa - current_toa, torch.zeros_like(current_toa))
    if clip > 0.0:
        progress = progress.clamp(min=-clip, max=clip)

    # Keep the cache on the simulation device and avoid retaining autograd state.
    previous_toa.copy_(current_toa.detach())
    return progress * scale


__all__ = ["toa_progress"]
