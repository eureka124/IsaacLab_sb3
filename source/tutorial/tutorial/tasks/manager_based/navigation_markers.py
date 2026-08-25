"""Scalable navigation markers shared by manager-based navigation tasks."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class _NavigationMarkerState:
    """Own point-instancer markers and a bounded trajectory ring buffer."""

    def __init__(self, env):
        cfg = env.cfg
        self.num_envs = min(env.num_envs, int(getattr(cfg, "navigation_marker_max_envs", 16)))
        self.trail_length = max(1, int(getattr(cfg, "navigation_marker_trail_length", 64)))
        self.trail_stride = max(1, int(getattr(cfg, "navigation_marker_trail_stride", 4)))
        self.velocity_scale = float(getattr(cfg, "navigation_marker_velocity_scale", 1.0))
        self.velocity_max_length = float(getattr(cfg, "navigation_marker_velocity_max_length", 4.0))
        self.velocity_height_offset = float(getattr(cfg, "navigation_marker_velocity_height_offset", 0.8))
        prim_path = str(getattr(cfg, "navigation_marker_prim_path", "/Visuals/Navigation"))

        self.goal_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=f"{prim_path}/Goals",
                markers={
                    "goal": sim_utils.SphereCfg(
                        radius=0.25,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.9, 0.2), emissive_color=(0.02, 0.15, 0.03)
                        ),
                    )
                },
            )
        )
        self.velocity_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=f"{prim_path}/Velocities",
                markers={
                    "velocity": sim_utils.ConeCfg(
                        radius=0.10,
                        height=1.0,
                        axis="X",
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.45, 1.0), emissive_color=(0.02, 0.06, 0.2)
                        ),
                    )
                },
            )
        )
        self.trail_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=f"{prim_path}/Trails",
                markers={
                    "trail": sim_utils.SphereCfg(
                        radius=0.065,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.55, 0.05), emissive_color=(0.18, 0.06, 0.0)
                        ),
                    )
                },
            )
        )

        self.trail_positions = torch.zeros(
            (self.num_envs, self.trail_length, 3), device=env.device, dtype=torch.float32
        )
        self.trail_valid = torch.zeros(
            (self.num_envs, self.trail_length), device=env.device, dtype=torch.bool
        )
        self.trail_write_indices = torch.zeros(self.num_envs, device=env.device, dtype=torch.long)
        self.last_recorded_step = -1

    @staticmethod
    def _velocity_quaternions(velocity: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Return WXYZ quaternions rotating the marker's +X axis onto velocity."""

        direction = velocity / speed.clamp_min(1.0e-6).unsqueeze(-1)
        quaternion = torch.stack(
            (
                1.0 + direction[:, 0],
                torch.zeros_like(direction[:, 0]),
                -direction[:, 2],
                direction[:, 1],
            ),
            dim=-1,
        )
        opposite = direction[:, 0] < -0.9999
        quaternion[opposite] = torch.tensor(
            (0.0, 0.0, 0.0, 1.0), device=velocity.device, dtype=velocity.dtype
        )
        stationary = speed < 1.0e-5
        quaternion[stationary] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), device=velocity.device, dtype=velocity.dtype
        )
        return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-6)

    def reset(self, env, env_ids: torch.Tensor) -> None:
        visible_env_ids = env_ids[env_ids < self.num_envs]
        if visible_env_ids.numel() == 0:
            return
        self.trail_positions[visible_env_ids] = 0.0
        self.trail_valid[visible_env_ids] = False
        self.trail_write_indices[visible_env_ids] = 0

        robot: Articulation = env.scene["robot"]
        positions = robot.data.root_pos_w[visible_env_ids]
        self.trail_positions[visible_env_ids, 0] = positions
        self.trail_valid[visible_env_ids, 0] = True
        self.trail_write_indices[visible_env_ids] = 1 % self.trail_length

    def update(self, env) -> None:
        if self.num_envs == 0 or not hasattr(env, "goal_positions"):
            return

        robot: Articulation = env.scene["robot"]
        positions = robot.data.root_pos_w[: self.num_envs]
        velocities = robot.data.root_lin_vel_w[: self.num_envs]
        speed = torch.linalg.vector_norm(velocities, dim=-1)

        self.goal_markers.visualize(translations=env.goal_positions[: self.num_envs])

        velocity_origins = positions.clone()
        velocity_origins[:, 2] += self.velocity_height_offset
        velocity_orientations = self._velocity_quaternions(velocities, speed)
        velocity_lengths = (speed * self.velocity_scale).clamp(0.05, self.velocity_max_length)
        velocity_scales = torch.ones_like(velocity_origins)
        velocity_scales[:, 0] = velocity_lengths
        self.velocity_markers.visualize(
            translations=velocity_origins,
            orientations=velocity_orientations,
            scales=velocity_scales,
        )

        current_step = int(env.common_step_counter)
        if current_step != self.last_recorded_step and current_step % self.trail_stride == 0:
            rows = torch.arange(self.num_envs, device=env.device)
            columns = self.trail_write_indices
            self.trail_positions[rows, columns] = positions
            self.trail_valid[rows, columns] = True
            self.trail_write_indices = torch.remainder(columns + 1, self.trail_length)
            self.last_recorded_step = current_step

        visible_trail = self.trail_positions[self.trail_valid]
        if visible_trail.numel() > 0:
            self.trail_markers.visualize(translations=visible_trail)


def _get_navigation_marker_state(env) -> _NavigationMarkerState | None:
    if not bool(getattr(env.cfg, "enable_navigation_markers", True)):
        return None
    cache_name = "_navigation_marker_state"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, _NavigationMarkerState(env))
    return getattr(env, cache_name)


def reset_navigation_markers(env, env_ids: torch.Tensor) -> None:
    """Clear trails for reset environments and move markers to current state."""

    marker_state = _get_navigation_marker_state(env)
    if marker_state is None:
        return
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    marker_state.reset(env, env_ids)
    marker_state.update(env)


def update_navigation_markers(env, env_ids: torch.Tensor) -> None:
    """Update goal, velocity and bounded trajectory markers."""

    marker_state = _get_navigation_marker_state(env)
    if marker_state is not None:
        marker_state.update(env)
