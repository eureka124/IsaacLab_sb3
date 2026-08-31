"""Pure-Python contract copied from the direct tutorial environment.

Keep manager-based policy-facing constants here so action, observation, TOA,
and reward code cannot silently drift to different scales.
"""

from __future__ import annotations

import math


TUTORIAL_ACTION_LOW = (-0.1, -0.5, -math.pi / 3.0)
TUTORIAL_ACTION_HIGH = (2.0, 0.5, math.pi / 3.0)
TUTORIAL_DEPTH_MAX_DISTANCE = 10.0
TUTORIAL_TOA_CROP_GRID_SPACING = 30.0 / 200.0
TUTORIAL_REWARD_WEIGHTS = {
    "goal_velocity": 0.5,
    "action_smoothness": -0.1,
    "contact_force": -200.0,
    "goal_reached": 300.0,
}


def reward_manager_weights(step_dt: float) -> dict[str, float]:
    """Return weights that undo RewardManager's automatic ``step_dt`` factor."""

    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    return {name: weight / step_dt for name, weight in TUTORIAL_REWARD_WEIGHTS.items()}
