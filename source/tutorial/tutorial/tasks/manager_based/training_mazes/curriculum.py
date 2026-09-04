"""Dependency-free helpers for the training-maze difficulty curriculum."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrainingMazeCurriculumStage:
    """Active environment complexity at one point in training."""

    index: int
    start_fraction: float
    maze_count: int
    cylinder_count: int


def select_training_maze_curriculum_stage(
    elapsed_steps: int,
    max_steps: int,
    stage_fractions: Sequence[float],
    maze_counts: Sequence[int],
    cylinder_counts: Sequence[int],
) -> TrainingMazeCurriculumStage:
    """Select a curriculum stage from aggregate environment transition counts."""

    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}.")
    if not (len(stage_fractions) == len(maze_counts) == len(cylinder_counts)):
        raise ValueError("Curriculum fractions, maze counts and cylinder counts must have equal lengths.")
    if not stage_fractions:
        raise ValueError("At least one curriculum stage is required.")
    if stage_fractions[0] != 0.0:
        raise ValueError("The first curriculum stage must start at fraction 0.0.")
    if any(current <= previous for previous, current in zip(stage_fractions, stage_fractions[1:])):
        raise ValueError("Curriculum stage fractions must be strictly increasing.")
    if stage_fractions[-1] > 1.0:
        raise ValueError("Curriculum stage fractions cannot exceed 1.0.")
    if any(count <= 0 for count in maze_counts):
        raise ValueError("Every curriculum stage must enable at least one maze.")
    if any(count < 0 for count in cylinder_counts):
        raise ValueError("Curriculum cylinder counts cannot be negative.")

    progress = min(max(float(elapsed_steps) / float(max_steps), 0.0), 1.0)
    stage_index = bisect_right(stage_fractions, progress) - 1
    return TrainingMazeCurriculumStage(
        index=stage_index,
        start_fraction=float(stage_fractions[stage_index]),
        maze_count=int(maze_counts[stage_index]),
        cylinder_count=int(cylinder_counts[stage_index]),
    )


__all__ = ["TrainingMazeCurriculumStage", "select_training_maze_curriculum_stage"]
