"""Pure-Python contract copied from the direct tutorial environment.

Keep manager-based policy-facing constants here so action, observation, TOA,
and reward code cannot silently drift to different scales.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


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


def reward_manager_weights(
    step_dt: float,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return direct-style weights, optionally overridden for a specific task."""

    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    direct_weights = dict(TUTORIAL_REWARD_WEIGHTS)
    if overrides is not None:
        direct_weights.update(overrides)
    return {name: weight / step_dt for name, weight in direct_weights.items()}
