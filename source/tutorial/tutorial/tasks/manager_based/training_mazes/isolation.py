"""Pure configuration invariants for cloned-environment isolation."""

from __future__ import annotations


CAMERA_MAX_DISTANCE = 12.0
DRONE_XY_LIMIT = 19.5
ENV_SPACING = 52.0


def minimum_cross_environment_drone_distance(
    env_spacing: float = ENV_SPACING,
    drone_xy_limit: float = DRONE_XY_LIMIT,
) -> float:
    """Return the conservative distance between drones in adjacent clones."""

    return env_spacing - 2.0 * drone_xy_limit


def validate_isolation_settings(
    env_spacing: float = ENV_SPACING,
    camera_max_distance: float = CAMERA_MAX_DISTANCE,
    drone_xy_limit: float = DRONE_XY_LIMIT,
) -> None:
    """Raise unless another clone is strictly beyond the camera far plane."""

    minimum_separation = minimum_cross_environment_drone_distance(env_spacing, drone_xy_limit)
    if minimum_separation <= camera_max_distance:
        raise ValueError(
            "Environment spacing does not guarantee camera isolation: "
            f"minimum drone separation is {minimum_separation:.2f} m but camera range is "
            f"{camera_max_distance:.2f} m."
        )


validate_isolation_settings()
