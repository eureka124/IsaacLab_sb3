"""Pure NumPy helpers for rendering contrast-preserving color TOA maps."""

from __future__ import annotations

import numpy as np


TOA_OBSTACLE_THRESHOLD = 1.0e5

# Reversed Viridis: low TOA is bright yellow and high TOA is dark purple.
# The palette is perceptually ordered and remains distinguishable for most
# forms of color-vision deficiency.
_VIRIDIS_R_POSITIONS = np.linspace(0.0, 1.0, 5, dtype=np.float32)
_VIRIDIS_R_COLORS = np.asarray(
    (
        (253, 231, 37),
        (94, 201, 98),
        (33, 145, 140),
        (59, 82, 139),
        (68, 1, 84),
    ),
    dtype=np.float32,
)


def toa_to_rgb(toa_map: np.ndarray) -> tuple[np.ndarray, float]:
    """Render one map in reversed Viridis using its largest free-space TOA.

    The goal region is yellow, increasing travel time progresses through green
    and blue to purple, and obstacles remain black. The second return value is
    the maximum free-space TOA used for scaling.
    """

    values = np.asarray(toa_map, dtype=np.float32)
    free_mask = np.isfinite(values) & (values < TOA_OBSTACLE_THRESHOLD)
    if not np.any(free_mask):
        raise ValueError("TOA map does not contain any finite non-obstacle cells.")

    free_max = float(np.max(values[free_mask]))
    scale = max(free_max, np.finfo(np.float32).eps)
    normalized_free = np.clip(values[free_mask] / scale, 0.0, 1.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel][free_mask] = np.rint(
            np.interp(
                normalized_free,
                _VIRIDIS_R_POSITIONS,
                _VIRIDIS_R_COLORS[:, channel],
            )
        ).astype(np.uint8)
    return rgb, free_max
