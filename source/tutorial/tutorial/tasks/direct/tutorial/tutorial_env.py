# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .tutorial_env_cfg import TutorialEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG, RED_ARROW_X_MARKER_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers
from omni_drones.controllers import LeePositionController
from isaaclab.utils.math import quat_from_matrix
from matplotlib import pyplot as plt


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

        # 速度方向箭头可视化
        arrow_cfg = RED_ARROW_X_MARKER_CFG.copy()
        arrow_cfg.markers["arrow"].scale = (2, 0.4, 0.4)  # 默认缩放
        arrow_cfg.prim_path = "/Visuals/Command/velocity_arrow"
        self.vel_arrow_visualizer = VisualizationMarkers(arrow_cfg)

        self.img_buffer = DepthImageBuffer(self.cfg.scene.num_envs, self.device)

        # 初始化 LeePositionController
        uav_params = {
            "name": "iris",
            "mass": 1.52,
            "inertia": {"xx": 0.0347563, "yy": 0.0458929, "zz": 0.0977},
            "rotor_configuration": {
                "rotor_angles": [-0.533708, 2.565218, 0.533708, -2.565218],
                "arm_lengths": [0.255539, 0.238537, 0.255539, 0.238537],
                "force_constants": [8.54858e-06, 8.54858e-06, 8.54858e-06, 8.54858e-06],
                "moment_constants": [
                    1.3677728816219314e-07,
                    1.3677728816219314e-07,
                    1.3677728816219314e-07,
                    1.3677728816219314e-07,
                ],
                "directions": [1, 1, -1, -1],
                "max_rotation_velocities": [838, 838, 838, 838],
            },
        }
        self.controller = LeePositionController(g=9.81, uav_params=uav_params).to(
            self.device
        )
        self.mixer_pinv = torch.linalg.pinv(self.controller.mixer)
        self.target_pos_setpoint = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_yaw_setpoint = torch.zeros((self.num_envs, 1), device=self.device)
        # [超时次数, 碰撞次数, 到达次数]
        self.success_rate_count = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        self.total_episodes = 0

        # Grid management
        self.env_spacing = self.cfg.scene.env_spacing
        self.grid_size = 10
        self.cell_size = self.env_spacing / self.grid_size

        # Collect obstacles from scene
        self.obstacles = []
        self.obstacle_radii = []
        # We assume Obstacle_0 to Obstacle_59 exist in scene as per config
        for i in range(60):
            name = f"Obstacle_{i}"
            # Check if it exists in the scene dictionary
            if name in self.scene.keys():
                obs = self.scene[name]
                self.obstacles.append(obs)
                # Get radius from config spawn.
                self.obstacle_radii.append(obs.cfg.spawn.radius)

        if self.obstacles:
            self.obstacle_radii = torch.tensor(self.obstacle_radii, device=self.device)

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

        # 2. 计算机头朝向和侧向 (用于将 vx, vy 动作转为世界系速度)
        w, x, y, z = torch.unbind(quat, dim=-1)
        # 前向向量 (机体系 x 轴在世界系下的投影)
        forward_x = w * w + x * x - y * y - z * z
        forward_y = 2 * (x * y + w * z)
        forward_direction = torch.stack(
            (forward_x, forward_y, torch.zeros_like(forward_y)), dim=-1
        )
        forward_direction = forward_direction / (
            torch.norm(forward_direction, dim=-1, keepdim=True) + 1e-6
        )
        # 侧向向量 (机体系 y 轴在世界系下的投影，简化为水平面内与前向垂直)
        side_direction = torch.stack(
            (
                -forward_direction[:, 1],
                forward_direction[:, 0],
                torch.zeros_like(forward_y),
            ),
            dim=-1,
        )

        # 3. 构造控制目标
        # actions[:, 0] 是vx, actions[:, 1] vy, actions[:, 2] 是 yaw_rate
        target_vel_xy = (
            forward_direction[:, :2] * self.actions[:, 0:1]
            + side_direction[:, :2] * self.actions[:, 1:2]
        )

        # 更新偏航角目标
        target_yaw_rate = self.actions[:, 2:3]
        self.target_yaw_setpoint += target_yaw_rate * self.cfg.sim.dt
        # 限制在 -pi 到 pi 之间
        self.target_yaw_setpoint = (self.target_yaw_setpoint + torch.pi) % (
            2 * torch.pi
        ) - torch.pi

        # 高度控制交给 Lee 控制器：设置目标高度为 2.0，目标垂直速度为 0
        target_z = 2.0
        target_vel_z = torch.zeros((self.num_envs, 1), device=self.device)

        target_vel = torch.cat([target_vel_xy, target_vel_z], dim=-1)

        # 积分更新目标位置
        self.target_pos_setpoint[:, 0:2] += target_vel_xy * self.cfg.sim.dt
        self.target_pos_setpoint[:, 2] = target_z

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
        ang_acc_thrust = (self.mixer_pinv @ real_cmd.unsqueeze(-1)).squeeze(-1)

        ang_acc = ang_acc_thrust[:, :3]
        thrust = ang_acc_thrust[:, 3]

        # 实际上，我们可以直接应用 thrust 和 torque
        # 为了简化，我们直接从 ang_acc 和 thrust 构造
        forces_local = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces_local[:, 0, 2] = thrust

        # torque = I * ang_acc.
        # 我们从 uav_params 中已知的惯性参数计算
        inertia = torch.tensor([0.0347563, 0.0458929, 0.0977], device=self.device)
        torques_local = (ang_acc * inertia).unsqueeze(1)

        self.robot.set_external_force_and_torque(
            forces_local, torques_local, body_ids=[0], is_global=False
        )

        # 6. 设置螺旋桨转速 (可视化)
        rpms = torch.sqrt(torch.clamp((cmd + 1) / 2, min=0.0)) * 838  # max_rot_vel
        joint_vel_targets = rpms * (2 * torch.pi / 60.0)  # 单位转换
        self.robot.write_joint_velocity_to_sim(joint_vel_targets, env_ids=None)

    def _update_velocity_arrow(self):
        # 获取当前速度
        vel = self.robot.data.root_lin_vel_w
        speed = torch.norm(vel, dim=-1, keepdim=True)

        # 计算位置：在无人机上方 0.5m
        pos = self.robot.data.root_pos_w + torch.tensor([0, 0, 0.5], device=self.device)

        # 计算朝向：将 X 轴对齐到速度方向
        # 归一化速度向量作为前向向量 (X)
        forward = vel / (speed + 1e-6)
        # 假设世界上方为 Z 轴
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(forward)
        # 计算右向向量 (Y) = Up x Forward
        right = torch.cross(up, forward, dim=-1)
        right = right / (torch.norm(right, dim=-1, keepdim=True) + 1e-6)
        # 重新计算上向向量 (Z) = Forward x Right
        actual_up = torch.cross(forward, right, dim=-1)

        # 构造旋转矩阵 [X, Y, Z]
        rot_mat = torch.stack([forward, right, actual_up], dim=-1)
        # 转换为四元数
        quat = quat_from_matrix(rot_mat)

        # 动态调整箭头长度，使其与速度成正比 (可选)
        # 这里我们让箭头长度 = 速度大小 * 0.5，最小 0.1
        scale_x = speed * 0.5
        scales = torch.cat(
            [scale_x, torch.full_like(scale_x, 0.1), torch.full_like(scale_x, 0.1)],
            dim=-1,
        )

        # 更新可视化
        self.vel_arrow_visualizer.visualize(pos, quat, scales=scales)

    def _get_norm_depth_image(self) -> torch.Tensor:
        depth_frame = (
            self.scene["camera"].data.output["distance_to_image_plane"].clone()
        )
        # 归一化深度图
        max_vals = 10.0  # 相机最远探测距离m
        depth_frame = torch.nan_to_num(
            depth_frame,
            nan=10.0,
            posinf=max_vals,  # depth_frame[depth_frame != float('inf')].max(),
            neginf=0,  # depth_frame.min()
        )
        depth_frame[depth_frame >= max_vals] = max_vals
        depth_frame = depth_frame / (max_vals + 1e-8)  # +1e-8防止除零
        # print(depth_norm.shape) # (env_num, 1, 16, 12)
        return depth_frame  # shape:[env_num, 1, 16, 12]

    def _get_observations(self) -> dict:
        depth_norm = self._get_norm_depth_image()
        camera_observation = self.img_buffer.update_buffer(depth_norm)
        # print(camera_observation)

        # robot_pos = self.robot.data.root_state_w[:, :2]  # 位置(x,y)
        # relative_position = (
        #     self.target_pos[:, :2] - robot_pos
        # )  # 目标位置相对于机器人的位置差 (num_envs, 2)

        # 修改为机体坐标系下的相对位置
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        diff_global = self.target_pos - robot_pos
        diff_body = quat_rotate_inverse(robot_quat, diff_global)
        relative_position = diff_body[:, :2]

        last_action = self.actions

        # 获取机体坐标系下的速度 (vx, vy)
        vel_w = self.robot.data.root_lin_vel_w
        vel_b = quat_rotate_inverse(robot_quat, vel_w)
        vel_xy = vel_b[:, :2]

        obs = torch.cat(
            (relative_position, last_action, vel_xy), dim=-1
        )  # shape: (num_envs, 7)

        # Calculate privileged information (nearest 5 obstacles)
        if len(self.obstacles) > 0:
            # 1. Gather all obstacle positions: (num_envs, num_obstacles, 3)
            all_obs_pos_w = torch.stack(
                [obs.data.root_pos_w for obs in self.obstacles], dim=1
            )
            # 2. Robot position: (num_envs, 1, 3)
            robot_pos_w = self.robot.data.root_pos_w.unsqueeze(1)
            # 3. Relative positions in world frame
            rel_pos_w = all_obs_pos_w - robot_pos_w
            # 4. Filter by distance (XY plane)
            dists = torch.norm(rel_pos_w[..., :2], dim=-1)
            # 5. Get 5 nearest
            k = min(5, len(self.obstacles))
            vals, indices = torch.topk(dists, k=k, largest=False)

            # 6. Gather geometric features
            # Create batch indices for gathering
            batch_indices = (
                torch.arange(self.num_envs, device=self.device)
                .unsqueeze(1)
                .expand(-1, k)
            )
            # Gather relative positions: (num_envs, k, 3)
            selected_rel_pos_w = rel_pos_w[batch_indices, indices]

            # Rotate to body frame (consistent with other obs)
            # Flatten to vector-process rotation
            flat_rel_pos_w = selected_rel_pos_w.reshape(-1, 3)
            flat_quats = self.robot.data.root_quat_w.repeat_interleave(k, dim=0)
            flat_rel_pos_b = quat_rotate_inverse(flat_quats, flat_rel_pos_w)
            selected_rel_pos_b = flat_rel_pos_b.reshape(self.num_envs, k, 3)

            # Extract XY: (num_envs, k, 2)
            feat_xy = selected_rel_pos_b[..., :2]

            # Gather radii: (num_envs, k)
            selected_radii = self.obstacle_radii[indices]

            # Combine: (num_envs, k, 3) -> [x, y, r]
            critic_obs = torch.cat([feat_xy, selected_radii.unsqueeze(-1)], dim=-1)
            # Flatten: (num_envs, k*3)
            critic_obs = critic_obs.reshape(self.num_envs, -1)

            # Pad if less than 5
            if k < 5:
                padding = torch.zeros((self.num_envs, (5 - k) * 3), device=self.device)
                critic_obs = torch.cat([critic_obs, padding], dim=-1)
        else:
            critic_obs = torch.zeros((self.num_envs, 15), device=self.device)

        observations = {
            "policy": {
                "robot-state": obs,
                "camera": camera_observation,
                "critic-state": critic_obs,
            },
        }
        # 更新速度箭头可视化
        self._update_velocity_arrow()
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
        arrived = distances <= 0.4
        self.arrived = arrived

        # 2. 判断是否碰撞
        if len(self.obstacles) > 0:
            # 机器人位置 (num_envs, 2)
            robot_pos = self.robot.data.root_pos_w[:, :2]

            # 障碍物位置 (num_obstacles, num_envs, 2)
            obstacle_pos = torch.stack(
                [obs.data.root_pos_w[:, :2] for obs in self.obstacles], dim=0
            )

            # 计算距离 (num_obstacles, num_envs)
            dists = torch.norm(obstacle_pos - robot_pos.unsqueeze(0), dim=-1)

            # 碰撞阈值 = 障碍物半径 + 机器人半径(安全距离)
            robot_radius = 0.5  # 假设机器人半径为0.5米
            thresholds = self.obstacle_radii.unsqueeze(1) + robot_radius

            # 判断是否发生碰撞
            collided = (dists < thresholds).any(dim=0)
        else:
            collided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        )  # 通常超时也需要重置，除非由外部 runner 处理W

        # 记录到 extras 中，方便 RL 框架读取
        self.extras["success"] = arrived
        self.extras["collided"] = collided
        self.extras["time_out"] = time_out

        return reset_envs, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        # print("Resetting envs:", env_ids)
        # 将重置的环境位置偏移到对应环境的原点位置
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
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
        self.img_buffer.reset_idx(env_ids)

        # 初始化控制器的目标点为当前重置后的位置和朝向
        self.target_pos_setpoint[env_ids] = default_root_state[:, :3]
        # 从归一化后的四元数提取 yaw
        w, x, y, z = torch.unbind(target_quat, dim=-1)
        self.target_yaw_setpoint[env_ids] = torch.atan2(
            2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        ).unsqueeze(-1)


class DepthImageBuffer:
    def __init__(self, num_envs, device, buffer_size=49):
        """
        初始化向量化深度图缓冲区
        Args:
            num_envs: 环境数量
            device: 设备
            buffer_size: 缓冲区大小，默认为49
        """
        self.num_envs = num_envs
        self.device = device
        self.buffer_size = buffer_size
        self.buffer: torch.Tensor | None = None  # (num_envs, buffer_size, h, w)
        self.current_sizes = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.buffer_full = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update_buffer(self, depth_norm: torch.Tensor) -> torch.Tensor:
        """
        向量化更新缓冲区
        Args:
            depth_norm: 形状为(num_envs, 16, 12, 1)的深度图张量
        Returns:
            combined_tensor: 形状为(num_envs, 3, 16, 12)的PyTorch张量
        """
        if self.buffer is None:
            h, w = depth_norm.shape[1], depth_norm.shape[2]
            self.buffer = torch.zeros(
                (self.num_envs, self.buffer_size, h, w),
                device=self.device,
                dtype=depth_norm.dtype,
            )

        new_imgs = depth_norm.squeeze(-1)  # (num_envs, h, w)

        # 处理已满的缓冲区
        full_mask = self.buffer_full
        if full_mask.any():
            self.buffer[full_mask] = torch.roll(
                self.buffer[full_mask], shifts=-1, dims=1
            )
            self.buffer[full_mask, -1] = new_imgs[full_mask]

        # 处理未满的缓冲区
        not_full_mask = ~full_mask
        if not_full_mask.any():
            env_idx = torch.where(not_full_mask)[0]
            size_idx = self.current_sizes[env_idx]
            self.buffer[env_idx, size_idx] = new_imgs[env_idx]
            self.current_sizes[env_idx] += 1
            self.buffer_full = self.buffer_full | (
                self.current_sizes >= self.buffer_size
            )

        # 计算索引
        latest_idx = torch.where(
            self.buffer_full, self.buffer_size - 1, self.current_sizes - 1
        )
        middle_idx = torch.where(
            self.buffer_full, self.buffer_size // 2, self.current_sizes // 2
        )
        oldest_idx = torch.zeros_like(latest_idx)

        indices = torch.stack(
            [latest_idx, middle_idx, oldest_idx], dim=1
        )  # (num_envs, 3)

        # 特殊情况处理
        size_1 = self.current_sizes == 1
        indices[size_1] = torch.tensor([0, 0, 0], device=self.device)

        size_2 = self.current_sizes == 2
        indices[size_2] = torch.tensor([1, 0, 0], device=self.device)

        # 收集图像
        batch_idx = (
            torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, 3)
        )
        return self.buffer[batch_idx, indices]

    def reset_idx(self, env_ids):
        """重置指定环境的缓冲区"""
        if len(env_ids) > 0:
            self.current_sizes[env_ids] = 0
            self.buffer_full[env_ids] = False
            if self.buffer is not None:
                self.buffer[env_ids] = 0


@torch.jit.script
def compute_rewards(
    reward_velocity: torch.Tensor,
    collided: torch.Tensor,
    penalty_smooth: torch.Tensor,
    arrived: torch.Tensor,
):
    total_reward = (
        reward_velocity * 1.0  # 速度在目标方向的分量
        + 1.0  # 存活奖励
        # - penalty_smooth * 0.1  # 平滑度惩罚
        - collided * 20.0  # 碰撞惩罚
        + arrived * 200.0  # 到达奖励
    )
    # print("Total Reward:", total_reward)
    # print("Reward Velocity:", reward_velocity)
    # print("Penalty Smooth:", penalty_smooth)
    # print("Collided:", collided)
    # print("Arrived:", arrived)
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


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.unsqueeze(1), v.unsqueeze(-1)).squeeze(-1) * 2.0
    return a - b + c
