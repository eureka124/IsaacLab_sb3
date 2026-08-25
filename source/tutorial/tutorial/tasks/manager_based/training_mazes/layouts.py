"""Geometry and navigation pairs for the six mazes in ``训练环境.jpg``.

Coordinates in the source layout are scaled into one 40 m x 40 m cloned environment.  All inner
walls deliberately share one primitive size; layouts only change poses.  This
keeps physics replication enabled when thousands of environments are used.
"""

from __future__ import annotations

from dataclasses import dataclass


MAZE_PLANAR_SCALE = 2.0
ARENA_HALF_EXTENT = 10.0 * MAZE_PLANAR_SCALE
ARENA_HEIGHT = 4.0
WALL_THICKNESS = 0.35 * MAZE_PLANAR_SCALE
INNER_WALL_LENGTH = 6.0 * MAZE_PLANAR_SCALE
MAX_INNER_WALLS = 3

RANDOM_CYLINDER_COUNT = 60
RANDOM_CYLINDER_VARIANTS = ((0.30, 2.5), (0.45, 3.2), (0.60, 4.0))
RANDOM_CYLINDER_SPECS = tuple(
    RANDOM_CYLINDER_VARIANTS[index % len(RANDOM_CYLINDER_VARIANTS)]
    for index in range(RANDOM_CYLINDER_COUNT)
)
RANDOM_CYLINDER_CLEARANCE = 0.35
RANDOM_CYLINDER_START_GOAL_CLEARANCE = 1.0
RANDOM_CYLINDER_MAX_ATTEMPTS = 128


@dataclass(frozen=True)
class InnerWall:
    """Pose of one fixed-size inner wall in local XY coordinates."""

    center_xy: tuple[float, float]
    yaw: float = 0.0


@dataclass(frozen=True)
class MazeLayout:
    """Inner walls and the two start/goal pairs of one training maze."""

    name: str
    walls: tuple[InnerWall, ...]
    start_goal_pairs: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


# Numbered clockwise from the upper-left, as specified by the source figure.
# A yaw of pi/2 turns the common horizontal cuboid into a vertical wall.
_BASE_MAZE_LAYOUTS = (
    MazeLayout(
        name="maze_01",
        walls=(InnerWall((0.0, 0.5)),),
        start_goal_pairs=(
            ((-1.2, -5.5), (-1.2, 5.5)),
            ((1.2, -5.5), (1.2, 5.5)),
        ),
    ),
    MazeLayout(
        name="maze_02",
        walls=(InnerWall((3.2, 3.0)), InnerWall((-3.2, -2.0))),
        start_goal_pairs=(
            ((-0.8, -6.0), (-1.8, 6.0)),
            ((1.2, -5.5), (0.8, 6.2)),
        ),
    ),
    MazeLayout(
        name="maze_03",
        walls=(InnerWall((-3.2, 3.6)), InnerWall((3.4, 0.4)), InnerWall((0.8, -3.5))),
        start_goal_pairs=(
            ((-1.3, -6.5), (-1.2, 6.2)),
            ((1.8, -6.3), (1.0, 6.0)),
        ),
    ),
    MazeLayout(
        name="maze_04",
        walls=(
            InnerWall((0.0, 1.8)),
            InnerWall((-3.0, -1.2), 1.5707963267948966),
            InnerWall((3.0, -1.2), 1.5707963267948966),
        ),
        start_goal_pairs=(
            ((-1.2, -1.2), (-1.2, 6.0)),
            ((1.2, -1.2), (1.2, 6.0)),
        ),
    ),
    MazeLayout(
        name="maze_05",
        walls=(InnerWall((-3.2, 3.2)), InnerWall((3.2, -3.2))),
        start_goal_pairs=(
            ((1.0, -6.3), (-5.0, 6.0)),
            ((4.5, -6.0), (-2.5, 6.2)),
        ),
    ),
    MazeLayout(
        name="maze_06",
        walls=(InnerWall((0.0, 1.8)), InnerWall((0.0, -1.2), 1.5707963267948966)),
        start_goal_pairs=(
            ((-3.0, -4.5), (-2.0, 5.5)),
            ((3.0, -4.5), (2.0, 5.5)),
        ),
    ),
)


MAZE_LAYOUTS = tuple(
    MazeLayout(
        name=layout.name,
        walls=tuple(
            InnerWall(
                (wall.center_xy[0] * MAZE_PLANAR_SCALE, wall.center_xy[1] * MAZE_PLANAR_SCALE),
                wall.yaw,
            )
            for wall in layout.walls
        ),
        start_goal_pairs=tuple(
            (
                (start[0] * MAZE_PLANAR_SCALE, start[1] * MAZE_PLANAR_SCALE),
                (goal[0] * MAZE_PLANAR_SCALE, goal[1] * MAZE_PLANAR_SCALE),
            )
            for start, goal in layout.start_goal_pairs
        ),
    )
    for layout in _BASE_MAZE_LAYOUTS
)


def validate_layouts(clearance: float = 0.6) -> None:
    """Raise when a layout violates arena, topology, or spawn-clearance invariants."""

    if len(MAZE_LAYOUTS) != 6:
        raise ValueError(f"Expected six training mazes, got {len(MAZE_LAYOUTS)}.")

    for layout in MAZE_LAYOUTS:
        if len(layout.walls) > MAX_INNER_WALLS:
            raise ValueError(f"{layout.name} has more than {MAX_INNER_WALLS} inner walls.")
        if len(layout.start_goal_pairs) != 2:
            raise ValueError(f"{layout.name} must define exactly two start/goal pairs.")

        for wall in layout.walls:
            vertical = abs(abs(wall.yaw) - 1.5707963267948966) < 1.0e-6
            half_x = WALL_THICKNESS * 0.5 if vertical else INNER_WALL_LENGTH * 0.5
            half_y = INNER_WALL_LENGTH * 0.5 if vertical else WALL_THICKNESS * 0.5
            if abs(wall.center_xy[0]) + half_x >= ARENA_HALF_EXTENT:
                raise ValueError(f"{layout.name} wall extends beyond the arena in X: {wall}.")
            if abs(wall.center_xy[1]) + half_y >= ARENA_HALF_EXTENT:
                raise ValueError(f"{layout.name} wall extends beyond the arena in Y: {wall}.")

        for start, goal in layout.start_goal_pairs:
            for label, point in (("start", start), ("goal", goal)):
                if max(abs(point[0]), abs(point[1])) >= ARENA_HALF_EXTENT - clearance:
                    raise ValueError(f"{layout.name} {label} {point} is too close to the boundary.")
                for wall in layout.walls:
                    vertical = abs(abs(wall.yaw) - 1.5707963267948966) < 1.0e-6
                    half_x = WALL_THICKNESS * 0.5 if vertical else INNER_WALL_LENGTH * 0.5
                    half_y = INNER_WALL_LENGTH * 0.5 if vertical else WALL_THICKNESS * 0.5
                    dx = max(abs(point[0] - wall.center_xy[0]) - half_x, 0.0)
                    dy = max(abs(point[1] - wall.center_xy[1]) - half_y, 0.0)
                    if (dx * dx + dy * dy) ** 0.5 < clearance:
                        raise ValueError(f"{layout.name} {label} {point} is too close to {wall}.")


validate_layouts()
