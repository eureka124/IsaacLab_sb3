from __future__ import annotations

from collections.abc import Sequence

import torch
from omni_drones.controllers import LeePositionController

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class VelocityYawAction(ActionTerm):
    """Track body-frame planar velocity and yaw-rate commands with a Lee controller."""

    _asset: Articulation

    def __init__(self, cfg: VelocityYawActionCfg, env):
        super().__init__(cfg, env)
        if not isinstance(self._asset, Articulation):
            raise TypeError(f"VelocityYawAction requires an Articulation, got {type(self._asset).__name__}.")

        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target_pos_setpoint = self._asset.data.root_pos_w.clone()
        self._target_yaw_setpoint = torch.zeros(self.num_envs, 1, device=self.device)
        self._base_link_ids, body_names = self._asset.find_bodies(cfg.body_name)
        if len(self._base_link_ids) != 1:
            raise RuntimeError(f"Expected one body matching {cfg.body_name!r}, got {body_names}.")

        uav_params = {
            "name": "hummingbird",
            "mass": 0.716,
            "inertia": {"xx": 0.007, "yy": 0.007, "zz": 0.012},
            "rotor_configuration": {
                "rotor_angles": [0.0, 1.57079632679, 3.14159265359, -1.57079632679],
                "arm_lengths": [0.17, 0.17, 0.17, 0.17],
                "force_constants": [8.54858e-06] * 4,
                "moment_constants": [1.3677728816219314e-07] * 4,
                "directions": [-1, 1, -1, 1],
                "max_rotation_velocities": [838] * 4,
            },
        }
        self._controller = LeePositionController(g=9.81, uav_params=uav_params).to(self.device)
        self.reset()

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        normalized_actions = actions.clamp(-1.0, 1.0)
        self._processed_actions[:, 0] = 0.95 + 1.05 * normalized_actions[:, 0]
        self._processed_actions[:, 1] = 0.5 * normalized_actions[:, 1]
        self._processed_actions[:, 2] = (torch.pi / 3.0) * normalized_actions[:, 2]

    def apply_actions(self) -> None:
        position = self._asset.data.root_pos_w
        orientation = self._asset.data.root_quat_w
        linear_velocity = self._asset.data.root_lin_vel_w
        angular_velocity = self._asset.data.root_ang_vel_w

        w, x, y, z = torch.unbind(orientation, dim=-1)
        forward = torch.stack(
            (w * w + x * x - y * y - z * z, 2.0 * (x * y + w * z)),
            dim=-1,
        )
        forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1.0e-6)
        side = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        target_velocity_xy = (
            forward * self._processed_actions[:, 0:1] + side * self._processed_actions[:, 1:2]
        )

        physics_dt = self._env.physics_dt
        self._target_pos_setpoint[:, :2] += target_velocity_xy * physics_dt
        self._target_pos_setpoint[:, 2] = self.cfg.target_height
        self._target_yaw_setpoint += self._processed_actions[:, 2:3] * physics_dt
        self._target_yaw_setpoint[:] = (self._target_yaw_setpoint + torch.pi) % (2.0 * torch.pi) - torch.pi

        target_velocity = torch.cat(
            (target_velocity_xy, torch.zeros(self.num_envs, 1, device=self.device)),
            dim=-1,
        )
        root_state = torch.cat((position, orientation, linear_velocity, angular_velocity), dim=-1)
        command = self._controller.compute(
            root_state=root_state,
            target_pos=self._target_pos_setpoint,
            target_vel=target_velocity,
            target_acc=torch.zeros_like(target_velocity),
            target_yaw=self._target_yaw_setpoint,
        )

        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        forces[:, 0, 2] = command[:, 0]
        torques = command[:, 1:4].unsqueeze(1)
        self._asset.set_external_force_and_torque(
            forces,
            torques,
            body_ids=self._base_link_ids,
            is_global=False,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._target_pos_setpoint[env_ids] = self._asset.data.root_pos_w[env_ids]
        orientation = self._asset.data.root_quat_w[env_ids]
        w, x, y, z = torch.unbind(orientation, dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self._target_yaw_setpoint[env_ids] = yaw.unsqueeze(-1)


@configclass
class VelocityYawActionCfg(ActionTermCfg):
    """Configuration for :class:`VelocityYawAction`."""

    class_type: type[ActionTerm] = VelocityYawAction
    body_name: str = "base_link"
    target_height: float = 2.0
