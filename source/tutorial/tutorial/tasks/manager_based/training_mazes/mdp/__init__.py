"""MDP terms for the training-maze task."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from tutorial.tasks.manager_based.tutorial.mdp.actions import VelocityYawActionCfg
from tutorial.tasks.manager_based.tutorial.mdp.observations import normalized_depth_image, robot_state
from tutorial.tasks.manager_based.tutorial.mdp.rewards import (
    action_smoothness,
    collision_detected,
    collision_termination,
    goal_reached,
    goal_reached_termination,
    set_direct_tutorial_reward_weights,
    velocity_towards_goal,
)
from tutorial.tasks.manager_based.tutorial.mdp.terminations import tutorial_time_out

from ...navigation_markers import reset_navigation_markers, update_navigation_markers
from .events import reset_training_maze
from .terminations import outside_isolated_arena
from .toa_observations import privileged_toa_map

__all__ = [
    "VelocityYawActionCfg",
    "action_smoothness",
    "collision_detected",
    "collision_termination",
    "goal_reached",
    "goal_reached_termination",
    "normalized_depth_image",
    "outside_isolated_arena",
    "privileged_toa_map",
    "reset_navigation_markers",
    "reset_training_maze",
    "robot_state",
    "set_direct_tutorial_reward_weights",
    "tutorial_time_out",
    "update_navigation_markers",
    "velocity_towards_goal",
]
