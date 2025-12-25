# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
from tensordict import TensorDict


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)

    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(2.0 * (w * y - z * x)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )

    return euler_angles


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    )
    matrix = matrix.unflatten(matrix.dim() - 1, (3, 3))
    return matrix


class DSLPIDController(nn.Module):
    def __init__(self, dt: float, g: float, uav_params) -> None:
        super().__init__()
        # 根据用户提供的参数调整:
        # kp_lin = 2.0 (对应 D_COEFF_FOR)
        # kp_ang = 1.2, izz = 0.006 => D_COEFF_TOR[2] = 0.0072
        # 用户在外部计算了 target_vel_z = 4*error_z，因此这里 P_COEFF_FOR 设为 0
        self.P_COEFF_FOR = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.I_COEFF_FOR = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.D_COEFF_FOR = nn.Parameter(torch.tensor([2.0, 2.0, 2.0]))  # 对应 kp_lin

        self.P_COEFF_TOR = nn.Parameter(torch.tensor([0.3, 0.3, 0.01]))
        self.I_COEFF_TOR = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.D_COEFF_TOR = nn.Parameter(torch.tensor([0.1, 0.1, 0.0072]))

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000.0
        self.MAX_PWM = 65535.0
        self.GRAVITY = nn.Parameter(torch.tensor([0, 0, abs(g) * uav_params["mass"]]))
        self.MIXER_MATRIX = nn.Parameter(
            torch.tensor(
                [
                    # [ .5, -.5, -1], # 0
                    # [ .5,  .5,  1], # 1
                    # [-.5,  .5, -1], # 2
                    # [-.5, -.5,  1], # 3
                    [-0.5, -0.5, 1],  # 3
                    [-0.5, 0.5, -1],  # 2
                    [0.5, 0.5, 1],  # 1
                    [0.5, -0.5, -1],  # 0
                ]
            )
        )
        self.KF = 1.0e-8
        self.dt = dt
        self.MAX_RPM = 21714
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        state: torch.Tensor,
        control_target: torch.Tensor,
        controller_state: TensorDict,
    ):
        # parsing input
        pos, quat, vel, angvel = torch.split(state, [3, 4, 3, 3], dim=-1)
        target_pos, target_vel, target_yaw = torch.split(
            control_target, [3, 3, 1], dim=-1
        )
        rpy = quaternion_to_euler(quat)
        rot = quaternion_to_rotation_matrix(quat)
        integral_pos_error = controller_state.get(
            "integral_pos_error", torch.zeros_like(pos)
        )
        integral_rpy_error = controller_state.get(
            "integral_rpy_error", torch.zeros_like(rpy)
        )
        last_rpy = controller_state.get("last_rpy", rpy)

        # position control
        pos_error = target_pos - pos
        vel_error = target_vel - vel
        integral_pos_error = torch.clip(integral_pos_error + pos_error * self.dt, -2, 2)

        target_thrust = (
            self.P_COEFF_FOR * pos_error
            + self.I_COEFF_FOR * integral_pos_error
            + self.D_COEFF_FOR * vel_error
            + self.GRAVITY
        )
        # Batch dot product: (N, 3) * (N, 3) -> (N)
        scalar_thrust = torch.sum(target_thrust * rot[:, :, 2], dim=-1)

        # attitute control
        target_x_c = torch.cat(
            [
                torch.cos(target_yaw),
                torch.sin(target_yaw),
                torch.zeros_like(target_yaw),
            ],
            dim=-1,
        )
        target_z_ax = normalize(target_thrust)
        target_y_ax = normalize(torch.cross(target_z_ax, target_x_c, dim=-1))
        target_x_ax = torch.cross(target_y_ax, target_z_ax, dim=-1)
        target_rot = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=-1)

        # Batch matrix multiplication and transpose
        rot_matrix_error = torch.matmul(
            target_rot.transpose(-1, -2), rot
        ) - torch.matmul(rot.transpose(-1, -2), target_rot)
        rot_error = torch.stack(
            [
                rot_matrix_error[:, 2, 1],
                rot_matrix_error[:, 0, 2],
                rot_matrix_error[:, 1, 0],
            ],
            dim=-1,
        )
        rpy_rates_error = -(rpy - last_rpy) / self.dt
        integral_rpy_error = integral_rpy_error - rot_error * self.dt
        target_torque = (
            -self.P_COEFF_TOR * rot_error
            + self.D_COEFF_TOR * rpy_rates_error
            + self.I_COEFF_TOR * integral_rpy_error
        )

        thrust = (
            torch.sqrt(torch.clamp(scalar_thrust / (4 * self.KF), min=0.0))
            - self.PWM2RPM_CONST
        ) / self.PWM2RPM_SCALE

        # Batch mixer: (N, 3) @ (3, 4) -> (N, 4)
        pwm = torch.clip(
            thrust.unsqueeze(-1) + target_torque @ self.MIXER_MATRIX.T, 0, 65535
        )
        rpms = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        cmd = torch.square(rpms / self.MAX_RPM) * 2 - 1

        controller_state.update(
            {
                "integral_pos_error": integral_pos_error,
                "integral_rpy_error": integral_rpy_error,
                "last_rpy": rpy,
            }
        )
        return cmd, controller_state, scalar_thrust, target_torque
