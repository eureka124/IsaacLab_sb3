"""Geometry for the independent unseen test maze shown in ``测试环境.jpg``.

The source figure is perspective-only and does not provide metric dimensions.
The coordinates below therefore preserve its topology and relative placement
inside a 30 m x 20 m arena instead of claiming an exact reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


TEST_ARENA_HALF_X = 15.0
TEST_ARENA_HALF_Y = 10.0
TEST_ARENA_HEIGHT = 4.0
TEST_WALL_THICKNESS = 0.35


@dataclass(frozen=True)
class BoxObstacle:
    """An oriented cuboid obstacle expressed in local arena coordinates."""

    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    yaw: float = 0.0


@dataclass(frozen=True)
class MovingBoxObstacle(BoxObstacle):
    """A cuboid following sinusoidal motion along one planar axis."""

    motion_axis_xy: tuple[float, float] = (1.0, 0.0)
    amplitude: float = 1.5
    period_s: float = 6.0


# Static structures, ordered approximately from upper-left to lower-right in
# the reference image. Compound U/L/X shapes are assembled from cuboids.
TEST_STATIC_OBSTACLES = (
    BoxObstacle((-12.0, 7.3), (4.8, TEST_WALL_THICKNESS)),
    BoxObstacle((-9.8, 6.2), (TEST_WALL_THICKNESS, 2.3)),
    BoxObstacle((-7.0, 7.2), (TEST_WALL_THICKNESS, 1.7)),
    BoxObstacle((-5.3, 4.7), (2.5, TEST_WALL_THICKNESS), math.pi / 4.0),
    BoxObstacle((-5.5, 4.5), (2.3, TEST_WALL_THICKNESS), -math.pi / 4.0),
    BoxObstacle((-4.3, 4.8), (1.8, TEST_WALL_THICKNESS), math.pi / 2.0),
    BoxObstacle((-1.8, 6.2), (2.8, TEST_WALL_THICKNESS), -math.pi / 6.0),
    BoxObstacle((-0.5, 5.3), (2.0, TEST_WALL_THICKNESS), math.pi / 2.0),
    BoxObstacle((5.2, 4.2), (3.6, 0.55)),
    BoxObstacle((9.3, 2.4), (3.0, 0.55)),
    BoxObstacle((5.4, -0.2), (3.8, 0.55)),
    BoxObstacle((-12.4, -1.8), (0.55, 2.0)),
    # U-shaped enclosure containing route 2's start point.
    BoxObstacle((-7.0, -1.8), (4.4, TEST_WALL_THICKNESS)),
    BoxObstacle((-9.0, -3.5), (TEST_WALL_THICKNESS, 3.4)),
    BoxObstacle((-5.0, -3.5), (TEST_WALL_THICKNESS, 3.4)),
    # Central X obstacle.
    BoxObstacle((-1.2, -4.2), (3.3, TEST_WALL_THICKNESS), math.pi / 4.0),
    BoxObstacle((-1.2, -4.2), (3.3, TEST_WALL_THICKNESS), -math.pi / 4.0),
    # Lower-right channel and enclosure.
    BoxObstacle((2.8, -5.0), (TEST_WALL_THICKNESS, 4.0)),
    BoxObstacle((6.0, -3.2), (6.4, TEST_WALL_THICKNESS)),
    BoxObstacle((10.0, -6.2), (6.0, TEST_WALL_THICKNESS)),
    BoxObstacle((12.8, -4.7), (TEST_WALL_THICKNESS, 3.2)),
)


# The two red-boxed obstacles in the figure. They are deliberately omitted
# from the global TOA field because their positions change during an episode.
TEST_MOVING_OBSTACLES = (
    MovingBoxObstacle((-0.8, 2.4), (0.65, 1.5), motion_axis_xy=(1.0, 0.0), amplitude=1.6, period_s=5.0),
    MovingBoxObstacle((0.0, 0.5), (0.85, 1.25), motion_axis_xy=(1.0, 0.0), amplitude=1.8, period_s=6.5),
)


# Purple markers are starts and green markers are goals in the source figure.
# One drone is used per cloned environment; env_id % 2 selects the route.
TEST_START_GOAL_PAIRS = (
    ((8.2, -8.0), (-10.7, 8.0)),
    ((-7.0, -3.8), (-12.8, 8.1)),
)


def validate_test_layout(clearance: float = 0.6) -> None:
    """Validate arena containment and basic start/goal clearance."""

    for obstacle in (*TEST_STATIC_OBSTACLES, *TEST_MOVING_OBSTACLES):
        half_diagonal = 0.5 * math.hypot(*obstacle.size_xy)
        if abs(obstacle.center_xy[0]) + half_diagonal >= TEST_ARENA_HALF_X:
            raise ValueError(f"Obstacle exceeds test arena X bounds: {obstacle}.")
        if abs(obstacle.center_xy[1]) + half_diagonal >= TEST_ARENA_HALF_Y:
            raise ValueError(f"Obstacle exceeds test arena Y bounds: {obstacle}.")

    for start, goal in TEST_START_GOAL_PAIRS:
        for label, point in (("start", start), ("goal", goal)):
            if abs(point[0]) >= TEST_ARENA_HALF_X - clearance:
                raise ValueError(f"Test {label} {point} is too close to the X boundary.")
            if abs(point[1]) >= TEST_ARENA_HALF_Y - clearance:
                raise ValueError(f"Test {label} {point} is too close to the Y boundary.")


validate_test_layout()
