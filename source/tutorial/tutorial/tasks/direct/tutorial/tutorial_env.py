# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .tutorial_env_cfg import TutorialEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers
from omni_drones.controllers import LeePositionController


class TutorialEnv(DirectRLEnv):
    cfg: TutorialEnvCfg

    def __init__(self, cfg: TutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # print("__init__")
        # 创建可视化目标位置的标记
        self.target_pos = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
        # 目标位置可视化
        marker_cfg.prim_path = "/Visuals/Command/goal_position"
        self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
        self.img_buffers = [DepthImageBuffer() for _ in range(self.cfg.scene.num_envs)]

        # 初始化 LeePositionController
        uav_params = {
            "name": "hummingbird",
            "mass": 0.5,
            "inertia": {"xx": 0.0023, "yy": 0.0023, "zz": 0.004},
            "rotor_configuration": {
                "rotor_angles": [0.785, 3.927, 5.498, 2.356],
                "arm_lengths": [0.15, 0.15, 0.15, 0.15],
                "force_constants": [8.54858e-06, 8.54858e-06, 8.54858e-06, 8.54858e-06],
                "moment_constants": [1.6e-07, 1.6e-07, 1.6e-07, 1.6e-07],
                "directions": [-1, -1, 1, 1],
                "max_rotation_velocities": [838, 838, 838, 838],
            },
        }
        self.controller = LeePositionController(g=9.81, uav_params=uav_params).to(
            self.device
        )
        self.target_pos_setpoint = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_yaw_setpoint = torch.zeros((self.num_envs, 1), device=self.device)

        # [超时次数, 碰撞次数, 到达次数]
        self.success_rate_count = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        self.total_episodes = 0
        self.last_condition_state = False

    def _setup_scene(self):
        self.robot = self.scene["robot_cfg"]
        # 添加地面平面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 克隆环境
        self.scene.clone_environments(copy_from_source=False)
        # 我们需要为 CPU 模拟明确过滤碰撞
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # 添加环境光
        light_cfg = sim_utils.DomeLightCfg(intensity=4000.0, color=(1.0, 1.0, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # # 可动障碍物的定义 (优化后)
        # # 随机生成障碍物位置
        # obstacle_params = []
        # for _ in range(5):
        #     while True:
        #         # 随机生成 x, y 坐标，范围在 -1.0 到 1.0 之间
        #         x = torch.rand(1).item() * 2.0 - 1.0
        #         y = torch.rand(1).item() * 2.0 - 1.0
        #         # 简单的避障：避免在原点(0,0)附近生成，防止与机器人重叠 (半径 > 0.1m)
        #         if x**2 + y**2 > 0.01:
        #             break
        #     z = 1.5  # 固定高度
        #     obstacle_params.append((x, y, z))
        # for i, pos in enumerate(obstacle_params):
        #     # 创建配置
        #     obstacle_cfg = RigidObjectCfg(
        #         prim_path=f"/World/move_obstacle/Move_Obstacle_{i}",
        #         spawn=sim_utils.CylinderCfg(
        #             radius=gauss(0.1, 0.02),
        #             height=3.0,
        #             rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #             mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #             collision_props=sim_utils.CollisionPropertiesCfg(),
        #             # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        #         ),
        #         init_state=RigidObjectCfg.InitialStateCfg(
        #             pos=pos,
        #         ),
        #     )
        #     # 实例化对象
        #     obstacle_obj = RigidObject(cfg=obstacle_cfg)
        #     # 动态设置类属性，相当于 self.Move_Obstacle_0 = ...
        #     setattr(self, f"Move_Obstacle_{i}", obstacle_obj)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # print("_pre_physics_step")
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 1. 获取当前状态
        pos = self.robot.data.root_pos_w
        quat = self.robot.data.root_quat_w
        vel = self.robot.data.root_lin_vel_w
        angvel = self.robot.data.root_ang_vel_w

        # 2. 计算机头朝向 (用于将 vx 动作转为世界系速度)
        w, x, y, z = torch.unbind(quat, dim=-1)
        x1 = w * w + x * x - y * y - z * z
        y1 = 2 * (x * y + w * z)
        horizon_direction = torch.stack((x1, y1, torch.zeros_like(y1)), dim=-1)
        normalized_direction = horizon_direction / (
            torch.norm(horizon_direction, dim=-1, keepdim=True) + 1e-6
        )

        # 3. 构造控制目标
        # actions[:, 0] 是前进速度, actions[:, 1] 是绕Z轴旋转角速度
        target_vel_xy = normalized_direction[:, :2] * self.actions[:, 0:1]
        target_z = 2.0
        error_z = target_z - pos[:, 2:3]
        target_vel_z = 4.0 * error_z

        target_vel = torch.cat([target_vel_xy, target_vel_z], dim=-1)
        target_yaw_rate = self.actions[:, 1:2]

        # 积分更新目标位置和偏航角
        self.target_pos_setpoint[:, 0:2] += target_vel_xy * self.cfg.sim.dt
        self.target_pos_setpoint[:, 2] = target_z
        self.target_yaw_setpoint += target_yaw_rate * self.cfg.sim.dt

        # 4. 调用 LeePositionController
        # Lee 控制器期望: root_state (13), target_pos (3), target_vel (3), target_acc (3), target_yaw (1)
        target_acc = torch.zeros_like(target_vel)
        state = torch.cat([pos, quat, vel, angvel], dim=-1)

        # compute 返回的是归一化的电机指令 [-1, 1]
        cmd = self.controller.compute(
            root_state=state,
            target_pos=self.target_pos_setpoint,
            target_vel=target_vel,
            target_acc=target_acc,
            target_yaw=self.target_yaw_setpoint,
        )

        # 5. 从电机指令还原物理力和力矩 (用于 set_external_force_and_torque)
        # cmd = (mixer @ [ang_acc, thrust].T).T / max_thrusts * 2 - 1
        # 我们通过逆运算获取 thrust 和 ang_acc
        real_cmd = (cmd + 1) / 2 * self.controller.max_thrusts

        # ang_acc_thrust = (mixer_pinverse @ real_cmd.T).T
        # 注意: LeePositionController 内部使用了 mixer = A.T @ (A @ A.T).inverse() @ I
        # 所以 ang_acc_thrust = [ang_acc, thrust]
        # 我们直接使用伪逆还原
        mixer_pinv = torch.linalg.pinv(self.controller.mixer)
        ang_acc_thrust = (mixer_pinv @ real_cmd.unsqueeze(-1)).squeeze(-1)

        ang_acc = ang_acc_thrust[:, :3]
        thrust = ang_acc_thrust[:, 3]

        # 实际上，我们可以直接应用 thrust 和 torque
        # 为了简化，我们直接从 ang_acc 和 thrust 构造
        forces_local = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces_local[:, 0, 2] = thrust

        # torque = I * ang_acc.
        # 我们从 uav_params 中已知的惯性参数计算
        inertia = torch.tensor([0.0023, 0.0023, 0.004], device=self.device)
        torques_local = (ang_acc * inertia).unsqueeze(1)

        self.robot.set_external_force_and_torque(
            forces_local, torques_local, body_ids=[0], is_global=False
        )

        # 6. 设置螺旋桨转速 (可视化)
        rpms = torch.sqrt(torch.clamp((cmd + 1) / 2, min=0.0)) * 838  # max_rot_vel
        joint_vel_targets = rpms * (2 * torch.pi / 60.0)  # 假设单位转换
        self.robot.write_joint_velocity_to_sim(joint_vel_targets, env_ids=None)

    def _get_norm_depth_image(self) -> torch.Tensor:
        depth_frame = (
            self.scene["camera"].data.output["distance_to_image_plane"].clone()
        )
        # 归一化深度图
        max_vals = 5  # 相机最远探测距离m
        depth_frame = torch.nan_to_num(
            depth_frame,
            nan=5.0,
            posinf=max_vals,  # depth_frame[depth_frame != float('inf')].max(),
            neginf=0,  # depth_frame.min()
        )
        depth_frame[depth_frame >= max_vals] = max_vals
        depth_norm = depth_frame / (max_vals + 1e-8)  # +1e-8防止除零
        # print(depth_norm.shape) # (env_num, 1, 16, 12)
        return depth_norm  # shape:[env_num, 1, 16, 12]

    def _get_observations(self) -> dict:
        depth_norm = self._get_norm_depth_image()
        camera_observation = torch.zeros(
            (self.cfg.scene.num_envs, 3, 16, 12), device=self.device
        )  # shape=[num_envs, 3, 16, 12]
        # print(depth_norm.shape)  # (num_envs, 16, 12, 1)
        for env in range(self.cfg.scene.num_envs):
            camera_observation[env] = self.img_buffers[env].update_buffer(
                depth_norm[env]
            )
            # print(depth_norm[env].shape, camera_observation[env].shape)# (3, 16, 12)
        # print(camera_observation)

        robot_pos = self.robot.data.root_state_w[:, :2]  # 位置(x,y)
        relative_position = (
            self.target_pos[:, :2] - robot_pos
        )  # 目标位置相对于机器人的位置差 (num_envs, 2)
        # 归一化
        relative_position /= torch.norm(relative_position, dim=-1, keepdim=True) + 1e-8
        last_action = self.actions
        obs = torch.cat(
            (relative_position, last_action), dim=-1
        )  # shape: (num_envs, 4)

        observations = {"policy": {"robot-state": obs, "camera": camera_observation}}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # 计算速度奖励(线速度在目标方向的分量)
        # 获取目标位置和机器人位置，计算方向向量
        target_pos = self.target_pos[:, :2]  # 目标位置 (num_envs, 2)
        robot_pos = self.robot.data.root_pos_w[:, :2]  # 机器人位置 (num_envs, 2)
        direction_vector = (
            target_pos - robot_pos
        )  # 目标点相对机器人的位置 (num_envs, 2)
        direction_vector = direction_vector / torch.norm(
            direction_vector, dim=-1, keepdim=True
        )  # 归一化方向向量 (num_envs, 2)
        linear_velocity = self.robot.data.root_lin_vel_w[
            :, :2
        ]  # 线速度 (num_envs, 2) [vx, vy]
        reward_velocity = torch.sum(
            linear_velocity * direction_vector, dim=1
        )  # 速度奖励 (num_envs,)

        # 计算动作平滑惩罚
        if not hasattr(self, "prev_actions"):
            self.prev_actions = torch.zeros_like(self.actions, device=self.device)
        action_diff = self.actions - self.prev_actions  # 动作变化 (num_envs    , 2)
        penalty_smooth = torch.norm(action_diff, p=2, dim=1)  # 平滑惩罚 (num_envs,)
        self.prev_actions = self.actions.clone()  # 更新前一动作
        # print("Penalty Smooth:", penalty_smooth)
        # print("Reward Velocity:", reward_velocity)
        # print("Collided:", self.collided)
        # print("Arrived:", self.arrived)
        total_reward = compute_rewards(
            reward_velocity,
            self.collided,
            penalty_smooth,
            self.arrived,
        )  # print(self.robot.data)

        # 检查 total_reward 是否包含 NaN 值，如果有则中断程序
        if torch.isnan(total_reward).any():
            raise RuntimeError("total_reward contains NaN values, interrupting program")

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # print("_get_dones")
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 1. 计算距离 (忽略 Z 轴差异，仅计算 XY 平面距离)
        pos = self.robot.data.root_pos_w[:, :2]  # 只取 (x, y)
        target = self.target_pos[:, :2]  # 只取 (x, y)

        # 计算欧几里得距离
        distances = torch.norm(target - pos, p=2, dim=-1)  # (num_envs,)
        self.distances = distances  # 保存用于可能的奖励计算

        # 判断是否到达 (阈值 0.1 米)
        arrived = distances <= 0.1
        self.arrived = arrived

        # 2. 判断是否碰撞
        # 获取接触力
        contact_forces = self.scene["contact_forces"].data.net_forces_w
        # 计算合力大小
        contact_magnitudes = torch.norm(contact_forces, dim=-1)
        # 阈值建议提高到 1.0，避免物理引擎噪声导致的误判
        collided = (contact_magnitudes > 1.0).any(dim=1)
        self.collided = collided

        # 3. 统计信息 (确保在 __init__ 中初始化了这些变量)
        num_resets = (
            time_out.sum().item() + collided.sum().item() + arrived.sum().item()
        )
        self.success_rate_count[0] += time_out.sum().item()
        self.success_rate_count[1] += collided.sum().item()
        self.success_rate_count[2] += arrived.sum().item()
        self.total_episodes += num_resets

        current_sum = sum(self.success_rate_count)
        # 每 100 次事件打印一次
        if current_sum > 0 and current_sum % 100 == 0:
            if not self.last_condition_state:  # 防止同一步重复打印
                success_rates = (
                    self.success_rate_count / self.success_rate_count.sum().item() * 100
                )
                print(
                    f"Stats [Timeout, Collision, Arrived]: [{success_rates[0]:.2f}%, {success_rates[1]:.2f}%, {success_rates[2]:.2f}%]"
                )
                self.last_condition_state = True
                self.success_rate_count[:] = 0
        else:
            self.last_condition_state = False

        # 4. 决定重置的环境
        reset_envs = (
            arrived | collided | time_out
        )  # 通常超时也需要重置，除非由外部 runner 处理

        return reset_envs, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 记录成功率到 TensorBoard (通过 extras)
        if (
            len(env_ids) > 0 and self.total_episodes >= 100
        ):  # 至少有100个episode后才记录
            if "log" not in self.extras:
                self.extras["log"] = dict()

            total = self.success_rate_count.sum().item()
            if total > 0:
                # 计算成功率百分比
                timeout_rate = (self.success_rate_count[0] / total * 100).item()
                collision_rate = (self.success_rate_count[1] / total * 100).item()
                success_rate = (self.success_rate_count[2] / total * 100).item()

                # 记录到 extras，这将被写入 TensorBoard
                self.extras["log"]["Success_Rate/timeout"] = timeout_rate
                self.extras["log"]["Success_Rate/collision"] = collision_rate
                self.extras["log"]["Success_Rate/arrived"] = success_rate
                self.extras["log"]["Success_Rate/total_episodes"] = self.total_episodes

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        # print("Resetting envs:", env_ids)
        # 将重置的环境位置偏移到对应环境的原点位置
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 0] += self.cfg.scene.env_spacing / 2.0 + 0.5
        default_root_state[:, 1] += self.cfg.scene.env_spacing / 2.0 + 0.5

        # 设置机器人速度
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # 为重置的环境采样新的目标位置
        len_env_ids = len(env_ids)
        x = (
            torch.zeros(len_env_ids, device=self.device)
            .uniform_(
                -self.cfg.scene.env_spacing / 2.0 + 4.0,
                self.cfg.scene.env_spacing / 2.0 - 4.0,
            )
            .unsqueeze(dim=1)
        )
        y = (
            torch.zeros(len_env_ids, device=self.device)
            .uniform_(
                -self.cfg.scene.env_spacing / 2.0 + 4.0,
                self.cfg.scene.env_spacing / 2.0 - 4.0,
            )
            .unsqueeze(dim=1)
        )
        z = torch.zeros(len_env_ids, device=self.device).uniform_(2, 2).unsqueeze(dim=1)
        xyz = torch.cat((x, y, z), dim=-1)
        # print("Sampled target positions for reset envs:", xyz)

        # 向量化更新：
        # 1. xyz 是局部坐标，加上对应环境的原点坐标 (env_origins[env_ids]) 转换为全局坐标
        # 2. 直接使用索引 [env_ids] 更新 self.target_pos 中对应的行，未重置的环境保持不变
        self.target_pos[env_ids] = xyz + self.scene.env_origins[env_ids]

        # 获取目标点相对机器人位置的yaw角度
        target_yaw = get_target_direction(
            self.target_pos[env_ids, :2], default_root_state[:, :2]
        )  # shape: (num_reset_envs,)
        # 将yaw角度转换为四元数
        target_quat = yaw_to_quaternion(target_yaw)  # shape: (num_reset_envs, 4)
        # print("Target yaw angles for reset envs:", target_yaw)

        # 确保四元数归一化（双重保险）
        quat_norm = torch.norm(target_quat, dim=-1, keepdim=True)
        target_quat = target_quat / (quat_norm + 1e-8)

        # 设置机器人朝向目标点
        default_root_state[:, 3:7] = target_quat

        # 设置机器人位置
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        # create markers if necessary for the first time
        # set their visibility to true
        self.goal_pos_visualizer.set_visibility(True)
        self.goal_pos_visualizer.visualize(self.target_pos)

        # 重置图像缓冲区
        for env_id in env_ids:
            self.img_buffers[env_id].clear_buffer()

        # 初始化控制器的目标点为当前重置后的位置和朝向
        self.target_pos_setpoint[env_ids] = default_root_state[:, :3]
        # 从归一化后的四元数提取 yaw
        w, x, y, z = torch.unbind(target_quat, dim=-1)
        self.target_yaw_setpoint[env_ids] = torch.atan2(
            2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        ).unsqueeze(-1)


class DepthImageBuffer:
    def __init__(self, buffer_size=49):
        """
        初始化深度图缓冲区
        Args:
            buffer_size: 缓冲区大小，默认为49
        """
        self.buffer_size = buffer_size
        self.buffer: torch.Tensor | None = None  # 用于存储所有深度图的张量
        self.current_size = 0  # 当前缓冲区中的图像数量
        self.buffer_full = False  # 缓冲区是否已满

    def update_buffer(self, depth_norm: torch.Tensor) -> torch.Tensor:
        """
        使用torch.roll优化缓冲区更新
        Args:
            depth_norm: 形状为(16,12,1)的深度图张量（GPU张量）
        Returns:
            combined_tensor: 形状为(16,12,3)的PyTorch张量（在GPU上）
        """
        h, w, c = depth_norm.shape

        # 初始化缓冲区
        if self.buffer is None:
            self.buffer = torch.zeros(
                self.buffer_size, h, w, device=depth_norm.device, dtype=depth_norm.dtype
            )

        if self.buffer_full:
            # 缓冲区已满，使用torch.roll滚动更新
            self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
            self.buffer[-1] = depth_norm.squeeze(-1)  # 移除最后一个维度
        else:
            # 缓冲区未满，直接添加
            self.buffer[self.current_size] = depth_norm.squeeze(-1)
            self.current_size += 1
            if self.current_size >= self.buffer_size:
                self.buffer_full = True

        # 选择三个时间点的图像：最新、中间、最旧
        if self.current_size == 1:
            # 只有1张图片
            indices = [0, 0, 0]
        elif self.current_size == 2:
            # 有2张图片
            indices = [1, 0, 0]
        else:
            # 有3张或以上图片
            latest_idx = (
                self.current_size - 1 if not self.buffer_full else self.buffer_size - 1
            )
            middle_idx = (
                self.current_size // 2
                if not self.buffer_full
                else self.buffer_size // 2
            )
            oldest_idx = 0
            indices = [latest_idx, middle_idx, oldest_idx]

        # 从缓冲区提取对应的图像并重新组合
        selected_imgs = [self.buffer[i].unsqueeze(0) for i in indices]
        combined_tensor = torch.cat(selected_imgs, dim=0)

        return combined_tensor

    def get_buffer_state(self) -> dict:
        """获取缓冲区状态信息"""
        return {
            "buffer_shape": self.buffer.shape if self.buffer is not None else None,
            "current_size": self.current_size,
            "buffer_full": self.buffer_full,
        }

    def clear_buffer(self):
        """清空缓冲区"""
        self.buffer = None
        self.current_size = 0
        self.buffer_full = False

    def get_buffer_status(self):
        """获取缓冲区状态"""
        return f"当前缓冲区大小: {len(self.buffer)}/{self.buffer_size}"

    def is_buffer_full(self):
        """
        检查缓冲区是否已满
        Returns:
            bool: 缓冲区已满返回True，否则返回False
        """
        if self.buffer is None:
            return False
        return len(self.buffer) >= self.buffer_size

    def get_current_size(self):
        """
        获取当前缓冲区中的元素数量
        Returns:
            int: 当前缓冲区中的图片数量
        """
        if self.buffer is None:
            return 0
        return len(self.buffer)


@torch.jit.script
def compute_rewards(
    reward_velocity: torch.Tensor,
    collided: torch.Tensor,
    penalty_smooth: torch.Tensor,
    arrived: torch.Tensor,
):
    total_reward = (
        reward_velocity * 1.0 - penalty_smooth * 0.1 - collided * 20.0 + arrived * 5.0
    )
    return total_reward.unsqueeze(-1)  # 返回 (num_envs, 1) 通常更安全


# 偏航角转换为四元数
@torch.jit.script
def yaw_to_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    """
    将yaw角度转换为四元数表示
    Args:
        yaw: 形状为(num_envs,)的yaw角度张量（单位：弧度）
    Returns:
        quaternions: 形状为(num_envs, 4)的四元数张量（归一化）
    """
    half_yaw = yaw * 0.5
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    # 四元数表示 (w, x, y, z)
    quaternions = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    quaternions[:, 0] = cy  # w
    quaternions[:, 1] = 0.0  # x
    quaternions[:, 2] = 0.0  # y
    quaternions[:, 3] = sy  # z

    # 显式归一化以确保数值稳定性
    norm = torch.sqrt(quaternions[:, 0] ** 2 + quaternions[:, 3] ** 2).unsqueeze(-1)
    quaternions = quaternions / (norm + 1e-8)

    return quaternions


# 根据目标位置和机器人位置计算yaw角度
@torch.jit.script
def get_target_direction(target_pos, robot_pos) -> torch.Tensor:
    direction_vector = (
        target_pos - robot_pos
    )  # 目标点相对机器人的位置 (num_reset_envs, 2)[env_id, x, y]
    direction_vector = direction_vector / torch.norm(
        direction_vector, dim=-1, keepdim=True
    )
    yaw = torch.atan2(direction_vector[:, 1], direction_vector[:, 0])  # 计算yaw角度
    return yaw  # shape: (num_reset_envs,)
