"""Termination terms aligned with the direct tutorial environment."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def tutorial_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Use TutorialEnv's ``max_episode_length - 1`` timeout boundary."""

    return env.episode_length_buf >= env.max_episode_length - 1
