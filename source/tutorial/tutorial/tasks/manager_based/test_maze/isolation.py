"""Isolation invariants for cloned copies of the rectangular test arena."""

from __future__ import annotations


CAMERA_MAX_DISTANCE = 12.0
ENV_SPACING = 50.0


def minimum_cross_environment_drone_distance(
    env_spacing: float = ENV_SPACING,
    drone_x_limit: float = 14.5,
    drone_y_limit: float = 9.5,
) -> float:
    """Return a conservative separation between drones in adjacent clones."""

    return env_spacing - 2.0 * max(drone_x_limit, drone_y_limit)


def validate_isolation_settings(
    env_spacing: float = ENV_SPACING,
    camera_max_distance: float = CAMERA_MAX_DISTANCE,
    drone_x_limit: float = 14.5,
    drone_y_limit: float = 9.5,
) -> None:
    """Raise unless other clones remain beyond the camera far plane."""

    minimum_separation = minimum_cross_environment_drone_distance(
        env_spacing, drone_x_limit, drone_y_limit
    )
    if minimum_separation <= camera_max_distance:
        raise ValueError(
            "Test environment spacing does not guarantee camera isolation: "
            f"minimum drone separation is {minimum_separation:.2f} m but camera range is "
            f"{camera_max_distance:.2f} m."
        )
