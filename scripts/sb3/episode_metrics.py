"""Dependency-free episode outcome aggregation for RL logging callbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeOutcomeWindow:
    """Accumulate terminal outcomes over a configurable episode window."""

    minimum_episodes: int = 100
    episodes: int = 0
    successes: int = 0
    collisions: int = 0
    timeouts: int = 0
    isolation_boundaries: int = 0

    def __post_init__(self) -> None:
        if self.minimum_episodes <= 0:
            raise ValueError("minimum_episodes must be positive.")

    @property
    def ready(self) -> bool:
        return self.episodes >= self.minimum_episodes

    def add(
        self,
        *,
        success: bool,
        collision: bool,
        timeout: bool,
        isolation_boundary: bool = False,
    ) -> None:
        """Add one completed episode to the current window."""

        self.episodes += 1
        self.successes += int(success)
        self.collisions += int(collision)
        self.timeouts += int(timeout)
        self.isolation_boundaries += int(isolation_boundary)

    def consume(self) -> dict[str, float]:
        """Return rates for the current window and clear its counters."""

        if self.episodes == 0:
            raise RuntimeError("Cannot consume an empty episode outcome window.")
        rates = {
            "success_rate": self.successes / self.episodes,
            "collision_rate": self.collisions / self.episodes,
            "timeout_rate": self.timeouts / self.episodes,
            "isolation_boundary_rate": self.isolation_boundaries / self.episodes,
            "episodes": float(self.episodes),
        }
        self.episodes = 0
        self.successes = 0
        self.collisions = 0
        self.timeouts = 0
        self.isolation_boundaries = 0
        return rates
