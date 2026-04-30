# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F
import os
import isaaclab.sim as sim_utils
import omni.timeline
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .tutorial_env_cfg import TutorialEnvCfg, maze_obstacles
from . import TOA
from isaaclab.markers import CUBOID_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from omni_drones.controllers import LeePositionController
from isaaclab.utils.math import quat_from_matrix
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np


class TutorialEnv(DirectRLEnv):
    cfg: TutorialEnvCfg

    def __init__(self, cfg: TutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Keep camera observation normalized in [-1, 1] for SB3 image handling.
        if isinstance(self.observation_space, gym.spaces.Dict) and "camera" in self.observation_space.spaces:
            self.observation_space.spaces["camera"] = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.observation_space.spaces["camera"].shape,
                dtype=np.float32,
            )

        # print("__init__")
        # 创建可视化目标位置的标记
        self.target_pos = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (1.0, 1.0, 1.0)
        # 目标位置可视化
        marker_cfg.prim_path = "/Visuals/Command/goal_position"
        self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

        # 速度方向箭头可视化
        arrow_cfg = RED_ARROW_X_MARKER_CFG.copy()
        arrow_cfg.markers["arrow"].scale = (2.5, 2.5, 2.5)  # 默认缩放
        arrow_cfg.prim_path = "/Visuals/Command/velocity_arrow"
        self.vel_arrow_visualizer = VisualizationMarkers(arrow_cfg)

        # self.img_buffer = DepthImageBuffer(self.cfg.scene.num_envs, self.device)

        # 初始化 LeePositionController
        uav_params = {
            "name": "hummingbird",
            "mass": 0.716,
            "inertia": {"xx": 0.007, "yy": 0.007, "zz": 0.012},
            "rotor_configuration": {
                "rotor_angles": [0.0, 1.57079632679, 3.14159265359, -1.57079632679],
                "arm_lengths": [0.17, 0.17, 0.17, 0.17],
                "force_constants": [8.54858e-06, 8.54858e-06, 8.54858e-06, 8.54858e-06],
                "moment_constants": [
                    1.3677728816219314e-07,
                    1.3677728816219314e-07,
                    1.3677728816219314e-07,
                    1.3677728816219314e-07,
                ],
                "directions": [-1, 1, -1, 1],
                "max_rotation_velocities": [838, 838, 838, 838],
            },
        }
        self.controller = LeePositionController(g=9.81, uav_params=uav_params).to(self.device)
        self.mixer_pinv = torch.linalg.pinv(self.controller.mixer)
        self.target_pos_setpoint = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_yaw_setpoint = torch.zeros((self.num_envs, 1), device=self.device)
        # [超时次数, 碰撞次数, 到达次数]
        self.success_rate_count = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.total_episodes = 0

        # Grid management
        self.env_spacing = self.cfg.scene.env_spacing
        self.grid_size = 10
        self.cell_size = 20.0 / self.grid_size

        # ToA map management
        self.toa_grid_size = 201
        half_extent = self.cfg.scene.env_spacing * 0.5
        self.toa_x_min = -half_extent
        self.toa_x_max = half_extent
        self.toa_y_min = -half_extent
        self.toa_y_max = half_extent
        self.toa_grid_xs = np.linspace(
            self.toa_x_min,
            self.toa_x_max,
            self.toa_grid_size,
            dtype=np.float32,
        )
        self.toa_grid_ys = np.linspace(
            self.toa_y_min,
            self.toa_y_max,
            self.toa_grid_size,
            dtype=np.float32,
        )
        self.toa_robot_radius = 0.4
        self.toa_safe_distance = 2.0
        self.toa_slow_speed = 0.2
        # critic TOA crop size from config
        self.critic_toa_crop_size = getattr(self.cfg, "critic_toa_crop_size", 64)
        self.toa_maps = torch.zeros(
            (self.num_envs, self.toa_grid_size, self.toa_grid_size),
            device=self.device,
        )
        self.prev_toa = torch.zeros(self.num_envs, device=self.device)

        # TOA Map Saving
        self.toa_save_interval = 100  # steps
        self.step_count = 0
        self.toa_output_dir = os.path.join(os.getcwd(), "outputs", "toa_maps")
        self.toa_grad_output_dir = os.path.join(os.getcwd(), "outputs", "toa_gradients")
        if not os.path.exists(self.toa_output_dir):
            os.makedirs(self.toa_output_dir)
        if not os.path.exists(self.toa_grad_output_dir):
            os.makedirs(self.toa_grad_output_dir)

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
            self.obstacle_radii_tensor = torch.tensor(self.obstacle_radii, device=self.device)
        else:
            self.obstacle_radii_tensor = torch.zeros((0,), device=self.device)

        # Precompute TOA maps for 4 corners
        self._precompute_corner_toa_maps()

    def _precompute_corner_toa_maps(self) -> None:
        """Precompute TOA maps for the 4 corner goal positions."""
        # Target coordinates local to environment origin: (15,15), (-15,15), (-15,-15), (15,-15)
        self.corner_coords = np.array(
            [
                [15.0, 15.0],
                [-15.0, 15.0],
                [-15.0, -15.0],
                [15.0, -15.0],
            ],
            dtype=np.float32,
        )

        self.precomputed_toa_maps = torch.zeros(
            (4, self.toa_grid_size, self.toa_grid_size),
            device=self.device,
        )

        from .tutorial_env_cfg import maze_obstacles

        for i, goal_xy in enumerate(self.corner_coords):
            toa_map = TOA.build_toa_map(
                goal_xy=goal_xy,
                grid_xs=self.toa_grid_xs,
                grid_ys=self.toa_grid_ys,
                obstacles=maze_obstacles,  # Static maze only
                robot_radius=self.toa_robot_radius,
                safe_distance=self.toa_safe_distance,
                slow_speed=self.toa_slow_speed,
            )
            self.precomputed_toa_maps[i] = torch.from_numpy(toa_map).to(self.device)

    def _sample_toa_from_maps(
        self,
        toa_maps: torch.Tensor,
        positions_xy: torch.Tensor,
        origins_xy: torch.Tensor,
    ) -> torch.Tensor:
        return TOA.sample_toa_from_maps(
            toa_maps,
            positions_xy,
            origins_xy,
            self.toa_x_min,
            self.toa_x_max,
            self.toa_y_min,
            self.toa_y_max,
        )

    def _compute_toa_gradient(self, positions_xy: torch.Tensor) -> torch.Tensor:
        """Compute TOA map gradient in world XY at given positions."""
        return TOA.compute_toa_gradient(
            self.toa_maps,
            positions_xy,
            self.scene.env_origins[:, :2],
            self.toa_grid_xs,
            self.toa_x_min,
            self.toa_x_max,
            self.toa_y_min,
            self.toa_y_max,
            self.device,
        )

    def _build_robot_aligned_critic_toa(self, crop_size: int = 64) -> torch.Tensor:
        """Build a robot-aligned, rotated and cropped TOA tensor for critic.

        Returns tensor shape (num_envs, 1, crop_size, crop_size), values normalized [0,1].
        """
        device = self.device
        dtype = torch.float32

        # world resolution of TOA grid
        dx = float(self.toa_grid_xs[1] - self.toa_grid_xs[0])
        half_extent = (crop_size / 2.0) * dx

        # build local (robot-frame) sampling grid centered at robot (meters)
        xs = torch.linspace(-half_extent + dx / 2.0, half_extent - dx / 2.0, steps=crop_size, device=device, dtype=dtype)
        ys = torch.linspace(-half_extent + dx / 2.0, half_extent - dx / 2.0, steps=crop_size, device=device, dtype=dtype)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        grid_local = torch.stack([X, Y], dim=-1)  # (H, W, 2)
        HW = crop_size * crop_size
        grid_flat = grid_local.view(1, HW, 2)  # (1,HW,2)

        N = self.num_envs
        # allow overriding by instance config
        if hasattr(self, "critic_toa_crop_size"):
            crop_size = int(self.critic_toa_crop_size)

        # robot positions relative to env origins (local coords)
        robot_pos_local = (self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]).to(device=device, dtype=dtype)

        # robot yaw from quaternion
        q = self.robot.data.root_quat_w.to(device=device, dtype=dtype)
        w, x, y, z = torch.unbind(q, dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # (N,)
        cos = torch.cos(yaw)
        sin = torch.sin(yaw)

        # build rotation matrices per env: rot = [[cos, -sin],[sin, cos]]
        rot = torch.zeros((N, 2, 2), device=device, dtype=dtype)
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = -sin
        rot[:, 1, 0] = sin
        rot[:, 1, 1] = cos

        # expand grid and apply rotation
        grid_exp = grid_flat.expand(N, -1, -1)  # (N,HW,2)
        rotated = torch.matmul(grid_exp, rot.transpose(1, 2))  # (N,HW,2)

        # world coordinates (in env-local frame) = robot_local + rotated
        world_pts = rotated + robot_pos_local.unsqueeze(1)  # (N,HW,2)

        # normalize to [-1,1] for grid_sample using toa bounds
        x_world = world_pts[..., 0]
        y_world = world_pts[..., 1]
        x_norm = 2.0 * (x_world - float(self.toa_x_min)) / float(self.toa_x_max - self.toa_x_min) - 1.0
        y_norm = 2.0 * (y_world - float(self.toa_y_min)) / float(self.toa_y_max - self.toa_y_min) - 1.0

        sample_grid = torch.stack([x_norm, y_norm], dim=-1).view(N, crop_size, crop_size, 2)

        # input maps: (N,1,Hmap,Wmap)
        input_maps = self.toa_maps.unsqueeze(1).to(device=device, dtype=dtype)

        sampled = F.grid_sample(
            input_maps,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # clamp and normalize to [-1, 1] for SB3 image handling
        sampled = torch.clamp(sampled, min=0.0, max=255.0) / 255.0
        sampled = sampled * 2.0 - 1.0
        return sampled

    def _update_toa_cache(self, env_ids: torch.Tensor) -> None:
        if env_ids is None or len(env_ids) == 0:
            return

        # Use precomputed maps by finding the closest corner for each target
        target_local = self.target_pos[env_ids, :2] - self.scene.env_origins[env_ids, :2]

        # Compute distance from each target to the 4 corners
        # corner_coords: (4, 2), target_local: (N, 2)
        dist = torch.norm(
            target_local.unsqueeze(1) - torch.from_numpy(self.corner_coords).to(self.device).unsqueeze(0),
            dim=-1,
        )
        corner_indices = torch.argmin(dist, dim=-1)  # (N,)

        self.toa_maps[env_ids] = self.precomputed_toa_maps[corner_indices]

        self.prev_toa[env_ids] = self._sample_toa_from_maps(
            self.toa_maps[env_ids],
            self.robot.data.root_pos_w[env_ids, :2],
            self.scene.env_origins[env_ids, :2],
        )

    def _setup_scene(self):
        self.robot = self.scene["robot_cfg"]
        # 添加地面平面
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
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

        # # Save TOA map periodically
        # self.step_count += 1
        # if self.step_count % self.toa_save_interval == 0:
        #     self._save_first_env_toa()
        #     self._save_first_env_toa_gradient()

    def _save_first_env_toa(self):
        """Save the TOA map of the first environment as an image."""
        if not hasattr(self, "toa_maps"):
            return

        # 获取地图并转置以匹配 (X, Y) 到 (Row, Col) 的 imshow 默认映射
        # 或者使用 origin='lower' 并确保 extent 正确
        toa_map = self.toa_maps[0].detach().cpu().numpy()
        plt.figure(figsize=(8, 6))
        # skfmm 返回的 grid 索引通常是 [y, x]，所以转置一下
        plt.imshow(
            toa_map.T,
            extent=[self.toa_x_min, self.toa_x_max, self.toa_y_min, self.toa_y_max],
            origin="lower",
            cmap="viridis_r",  # 使用反向色图，较小值（目标）显示为深色，障碍物/远端显示为浅色
        )

        # 标记第一个环境的目标位置 (相对于环境原点的局部坐标)
        goal_xy = (self.target_pos[0, :2] - self.scene.env_origins[0, :2]).detach().cpu().numpy()
        plt.plot(goal_xy[0], goal_xy[1], "r*", markersize=15, label="Goal")

        # 标记机器人当前位置
        robot_xy = (self.robot.data.root_pos_w[0, :2] - self.scene.env_origins[0, :2]).detach().cpu().numpy()
        plt.plot(robot_xy[0], robot_xy[1], "bo", markersize=10, label="Robot")

        # 绘制迷宫墙壁 (maze_obstacles)
        from matplotlib.patches import Rectangle

        for pos, size in maze_obstacles:
            left = pos[0] - size[0] / 2.0
            bottom = pos[1] - size[1] / 2.0
            rect = Rectangle(
                (left, bottom),
                size[0],
                size[1],
                linewidth=1,
                edgecolor="black",
                facecolor="gray",
                alpha=0.5,
            )
            plt.gca().add_patch(rect)

        plt.colorbar(label="Time of Arrival")
        plt.title(f"TOA Map - Step {self.step_count}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        # plt.legend()

        save_path = os.path.join(self.toa_output_dir, f"toa_step_{self.step_count:06d}.png")
        plt.savefig(save_path)
        plt.close()
        # print(f"Saved TOA map to {save_path}")

    def _save_first_env_toa_gradient(self):
        """Save the TOA gradient field of the first environment as a quiver plot."""
        if not hasattr(self, "toa_maps"):
            return

        # 1. 准备全局网格点 (世界系 X, Y)
        # 降采样以避免箭头过密，例如每 3 个点取一个
        stride = 3
        grid_xs_sampled = self.toa_grid_xs[::stride]
        grid_ys_sampled = self.toa_grid_ys[::stride]

        # 生成世界坐标网格
        X_local, Y_local = np.meshgrid(grid_xs_sampled, grid_ys_sampled, indexing="xy")
        origin_xy = self.scene.env_origins[0, :2].unsqueeze(0)  # (1, 2)

        # 将局部网格点转换为世界坐标以便传入 _compute_toa_gradient
        X_world = X_local + origin_xy[0, 0].item()
        Y_world = Y_local + origin_xy[0, 1].item()

        points_world = torch.stack(
            [
                torch.from_numpy(X_world).to(self.device).flatten(),
                torch.from_numpy(Y_world).to(self.device).flatten(),
            ],
            dim=-1,
        )  # (N, 2)

        # 2. 计算梯度 (使用现有的类方法，它会调用 TOA.compute_toa_gradient)
        # 注意：这里需要修改 _compute_toa_gradient 或直接调用 TOA 里的函数来支持大批量点
        # 现有的 _compute_toa_gradient 使用 self.toa_maps (全环境) 和 env_origins
        # 我们这里只需要 env 0 的梯度
        # 直接使用 TOA 模块工具计算 env 0 的梯度分布
        # 为了高效，我们只处理 env 0 的地图
        env0_map = self.toa_maps[0:1]  # (1, H, W)
        env0_origin = self.scene.env_origins[0:1, :2]  # (1, 2)

        # 批量重复 env_origins 以匹配 points_world 的数量
        points_batch = points_world  # (N, 2)
        origins_batch = env0_origin.expand(points_world.shape[0], -1)  # (N, 2)

        # 梯度是指向值增加最快的方向，导航通常需要 -grad
        grad = TOA.compute_toa_gradient(
            env0_map.expand(points_world.shape[0], -1, -1),
            points_batch,
            origins_batch,
            self.toa_grid_xs,
            self.toa_x_min,
            self.toa_x_max,
            self.toa_y_min,
            self.toa_y_max,
            self.device,
        )  # (N, 2)

        grad_np = grad.detach().cpu().numpy()
        U = grad_np[:, 0].reshape(X_local.shape)
        V = grad_np[:, 1].reshape(Y_local.shape)

        # 3. 绘图
        plt.figure(figsize=(10, 10))
        # 背景画出 TOA 地图
        toa_map = self.toa_maps[0].detach().cpu().numpy()
        plt.imshow(
            toa_map.T,
            extent=[self.toa_x_min, self.toa_x_max, self.toa_y_min, self.toa_y_max],
            origin="lower",
            cmap="viridis_r",
            alpha=0.3,
        )

        # 画出梯度矢量场 (通常画 -grad，即指向目标的方向)
        plt.quiver(
            X_local,
            Y_local,
            -U,
            -V,
            color="g",
            scale=80,
            width=0.002,
            label="-Gradient (to Goal)",
        )

        # 标记目标和机器人
        goal_xy = (self.target_pos[0, :2] - self.scene.env_origins[0, :2]).detach().cpu().numpy()
        plt.plot(goal_xy[0], goal_xy[1], "r*", markersize=15, label="Goal")

        robot_xy = (self.robot.data.root_pos_w[0, :2] - self.scene.env_origins[0, :2]).detach().cpu().numpy()
        plt.plot(robot_xy[0], robot_xy[1], "bo", markersize=10, label="Robot")

        # 4. 绘制迷宫墙壁 (maze_obstacles)
        from matplotlib.patches import Rectangle

        for pos, size in maze_obstacles:
            # pos 是中心点 (x, y, z)，size 是 (sx, sy, sz)
            # 计算左下角坐标
            left = pos[0] - size[0] / 2.0
            bottom = pos[1] - size[1] / 2.0
            rect = Rectangle(
                (left, bottom),
                size[0],
                size[1],
                linewidth=1,
                edgecolor="black",
                facecolor="gray",
                alpha=0.5,
            )
            plt.gca().add_patch(rect)

        plt.title(f"TOA Gradient Field (-grad) - Step {self.step_count}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        # plt.legend()

        save_path = os.path.join(self.toa_grad_output_dir, f"toa_grad_step_{self.step_count:06d}.png")
        plt.savefig(save_path)
        plt.close()

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
        forward_direction = torch.stack((forward_x, forward_y, torch.zeros_like(forward_y)), dim=-1)
        forward_direction = forward_direction / (torch.norm(forward_direction, dim=-1, keepdim=True) + 1e-6)
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
        target_vel_xy = forward_direction[:, :2] * self.actions[:, 0:1] + side_direction[:, :2] * self.actions[:, 1:2]

        # 更新偏航角目标
        target_yaw_rate = self.actions[:, 2:3]
        self.target_yaw_setpoint += target_yaw_rate * self.cfg.sim.dt
        # 限制在 -pi 到 pi 之间
        self.target_yaw_setpoint = (self.target_yaw_setpoint + torch.pi) % (2 * torch.pi) - torch.pi

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

        # compute 返回的是直接的推力和扭矩: [推力, 扭矩_x, 扭矩_y, 扭矩_z]
        cmd = self.controller.compute(
            root_state=state,
            target_pos=self.target_pos_setpoint,
            target_vel=target_vel,
            target_acc=target_acc,
            target_yaw=self.target_yaw_setpoint,
        )

        # 5. 直接使用控制器输出的推力和扭矩 (用于 set_external_force_and_torque)
        thrust = cmd[:, 0]
        torques = cmd[:, 1:4]

        forces_local = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces_local[:, 0, 2] = thrust

        torques_local = torques.unsqueeze(1)

        self.robot.set_external_force_and_torque(forces_local, torques_local, body_ids=[0], is_global=False)

        self._update_velocity_arrow()

        # # 6. 设置螺旋桨转速 (可视化) - 已关闭
        # # 因为控制器不再输出每一个电机的归一化指令，所以可以用一个稳定的基础转速作为可视化
        # base_rpm = 600.0
        # joint_vel_targets = torch.full((self.num_envs, 4), base_rpm * (2 * torch.pi / 60.0), device=self.device)
        # self.robot.write_joint_velocity_to_sim(joint_vel_targets, env_ids=None)

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
        depth_frame = self.scene["camera"].data.output["distance_to_image_plane"].clone()

        # Normalize camera tensor to NCHW before pooling.
        # Common layouts from sensor output are NHWC (last dim = 1) or NHW.
        if depth_frame.dim() == 4 and depth_frame.shape[-1] == 1:
            depth_frame = depth_frame.permute(0, 3, 1, 2)
        elif depth_frame.dim() == 3:
            depth_frame = depth_frame.unsqueeze(1)
        elif depth_frame.dim() != 4:
            raise RuntimeError(f"Unexpected raw depth tensor shape: {tuple(depth_frame.shape)}")

        # 最小池化降采样 (kernel_size=4, stride=4) 从 (48, 64) 到 (12, 16)
        depth_frame = -torch.nn.functional.max_pool2d(-depth_frame, kernel_size=4, stride=4)

        # 归一化深度图
        max_vals = 10.0  # 相机最远探测距离m
        depth_frame = torch.nan_to_num(
            depth_frame,
            nan=max_vals,
            posinf=max_vals,
            neginf=0.0,
        )
        depth_frame = torch.clamp(depth_frame, min=0.0, max=max_vals)
        depth_frame = depth_frame / max_vals  # 归一化到 [0, 1]
        depth_frame = depth_frame * 2.0 - 1.0  # 映射到 [-1, 1]
        # print(depth_norm.shape) # (env_num, 1, 12, 16)
        return depth_frame  # shape:[env_num, 1, 12, 16]

    def _get_observations(self) -> dict:
        # depth_norm = self._get_norm_depth_image()
        camera_observation = self._get_norm_depth_image()
        # camera_observation = self.img_buffer.update_buffer(depth_norm)
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
        relative_position = diff_body[:, :2].clamp(-5.0, 5.0)  # 目标位置相对于机器人的位置差 (num_envs, 2)，并裁剪到合理范围

        last_action = self.actions

        # 获取机体坐标系下的速度 (vx, vy)
        vel_w = self.robot.data.root_lin_vel_w
        vel_b = quat_rotate_inverse(robot_quat, vel_w)
        vel_xy = vel_b[:, :2]

        # 获取机体坐标系下的角速度 (yaw_rate)
        ang_vel_w = self.robot.data.root_ang_vel_w
        ang_vel_b = quat_rotate_inverse(robot_quat, ang_vel_w)
        yaw_rate = ang_vel_b[:, 2:3]

        obs = torch.cat((relative_position, last_action, vel_xy, yaw_rate), dim=-1)  # shape: (num_envs, 8)

        # Calculate privileged information (nearest 5 obstacles)
        def get_nearest_obstacle_features():
            if len(self.obstacles) > 0:
                # 1. Gather all obstacle positions: (num_envs, num_obstacles, 3)
                all_obs_pos_w = torch.stack([obs.data.root_pos_w for obs in self.obstacles], dim=1)
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
                batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, k)
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
                selected_radii = self.obstacle_radii_tensor[indices]

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
                "critic-toa": self._build_robot_aligned_critic_toa(crop_size=self.critic_toa_crop_size),
            },
        }
        # print("Observations:", {k: v for k, v in observations["policy"].items()})
        # 更新速度箭头可视化
        # self._update_velocity_arrow()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # 计算速度奖励(线速度在目标方向的分量)
        # 获取目标位置和机器人位置，计算方向向量
        target_pos = self.target_pos[:, :2]  # 目标位置 (num_envs, 2)
        robot_pos = self.robot.data.root_pos_w[:, :2]  # 机器人位置 (num_envs, 2)
        direction_vector = target_pos - robot_pos  # 目标点相对机器人的位置 (num_envs, 2)
        direction_vector = direction_vector / (torch.norm(direction_vector, dim=-1, keepdim=True) + 1e-6)  # 归一化方向向量 (num_envs, 2)
        linear_velocity = self.robot.data.root_lin_vel_w[:, :2]  # 线速度 (num_envs, 2) [vx, vy]
        reward_velocity = torch.sum(linear_velocity * direction_vector, dim=1)  # 速度奖励 (num_envs,)

        # 计算动作平滑惩罚
        if not hasattr(self, "prev_actions"):
            self.prev_actions = torch.zeros_like(self.actions, device=self.device)
        action_diff = self.actions - self.prev_actions  # 动作变化 (num_envs    , 2)
        penalty_smooth = torch.norm(action_diff, p=2, dim=1)  # 平滑惩罚 (num_envs,)
        self.prev_actions = self.actions.clone()  # 更新前一动作

        # 计算障碍物距离惩罚
        if len(self.obstacles) > 0:
            obstacle_pos = torch.stack([obs.data.root_pos_w[:, :2] for obs in self.obstacles], dim=0)  # (num_obstacles, num_envs, 2)
            # 计算机器人到障碍物中心的距离
            # robot_pos: (num_envs, 2) -> unsqueeze(0) -> (1, num_envs, 2)
            dists = torch.norm(obstacle_pos - robot_pos.unsqueeze(0), dim=-1)  # (num_obstacles, num_envs)

            # 计算到障碍物表面的距离: 中心距离 - 障碍物半径
            # self.obstacle_radii_tensor: (num_obstacles,) -> unsqueeze(1) -> (num_obstacles, 1)
            dist_to_surface = dists - self.obstacle_radii_tensor.unsqueeze(1)

            # 找到最近的障碍物距离 (num_envs,)
            min_dist_to_surface, _ = torch.min(dist_to_surface, dim=0)

            # 计算惩罚: 距离1m时惩罚为0, 距离0.4m时惩罚为3. 0.4m到1m之间线性变化 (P = 4 * (0.8 - d))
            # clamp min=0 确保距离大于1m时无惩罚
            obstacle_penalty = torch.clamp(4.0 * (0.8 - min_dist_to_surface), min=0.0)
        else:
            obstacle_penalty = torch.zeros(self.num_envs, device=self.device)

        current_toa = self._sample_toa_from_maps(
            self.toa_maps,
            self.robot.data.root_pos_w[:, :2],
            self.scene.env_origins[:, :2],
        )
        reward_toa = self.prev_toa - current_toa
        self.prev_toa = current_toa.detach()

        # Yaw penalty: angle between yaw heading and negative TOA gradient direction
        toa_grad = self._compute_toa_gradient(self.robot.data.root_pos_w[:, :2])
        desired_dir = -toa_grad
        desired_norm = torch.norm(desired_dir, dim=-1, keepdim=True)
        desired_dir = desired_dir / (desired_norm + 1e-6)

        robot_quat = self.robot.data.root_quat_w
        w, x, y, z = torch.unbind(robot_quat, dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        heading_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

        dot = torch.sum(heading_dir * desired_dir, dim=-1).clamp(-1.0, 1.0)
        yaw_diff = torch.acos(dot)
        yaw_penalty = yaw_diff * (desired_norm.squeeze(-1) > 1e-5).float()

        # print("Penalty Smooth:", penalty_smooth)
        # print("Reward Velocity:", reward_velocity)
        # print("Collided:", self.collided)
        # print("Arrived:", self.arrived)
        total_reward = compute_rewards(
            reward_velocity,
            reward_toa,
            self.collided,
            penalty_smooth,
            self.arrived,
            obstacle_penalty,
            yaw_penalty,
        )  # print(self.robot.data)

        # 检查 total_reward 是否包含 NaN 值，如果有则中断程序
        if torch.isnan(total_reward).any():
            if torch.isnan(reward_velocity).any():
                print("NaN detected in reward_velocity")
                print("linear_velocity:", linear_velocity)
                print("direction_vector:", direction_vector)
                print("robot_pos:", robot_pos)
                print("target_pos:", target_pos)
            if torch.isnan(reward_toa).any():
                print("NaN detected in reward_toa")
                print("current_toa:", current_toa)
                print("prev_toa:", self.prev_toa)
            if torch.isnan(penalty_smooth).any():
                print("NaN detected in penalty_smooth")
                print("actions:", self.actions)
                print("prev_actions:", self.prev_actions)
            if torch.isnan(self.collided.float()).any():
                print("NaN detected in collided")
            if torch.isnan(self.arrived.float()).any():
                print("NaN detected in arrived")
            if torch.isnan(obstacle_penalty).any():
                print("NaN detected in obstacle_penalty")
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

        # 2. 判断是否碰撞 (使用接触力传感器)
        # 获取接触力，传感器名称在 SceneCfg 中定义为 "contact_forces"
        # data.net_forces_w shape: (num_envs, num_bodies, 3)
        contact_forces = self.scene["contact_forces"].data.net_forces_w
        # 计算接触力的模长，如果大于某个阈值 (例如 1.0N) 则认为发生碰撞
        # 排除与地面的正常接触 (如果在起飞前有接触，或者这里只关注大于一定阈值的力)
        collided = torch.any(torch.norm(contact_forces, dim=-1) > 1.0, dim=1)
        self.collided = collided
        # 3. 统计信息 (确保在 __init__ 中初始化了这些变量)
        num_resets = time_out.sum().item() + collided.sum().item() + arrived.sum().item()
        self.success_rate_count[0] += time_out.sum().item()
        self.success_rate_count[1] += collided.sum().item()
        self.success_rate_count[2] += arrived.sum().item()
        self.total_episodes += num_resets

        current_sum = sum(self.success_rate_count)
        # 每 100 次事件打印一次
        if current_sum > 0 and current_sum % 1000 == 0:
            if not self.last_condition_state:  # 防止同一步重复打印
                success_rates = self.success_rate_count / self.success_rate_count.sum().item() * 100
                print(f"Stats [Timeout, Collision, Arrived]: [{success_rates[0]:.2f}%, {success_rates[1]:.2f}%, {success_rates[2]:.2f}%, total episodes: {self.success_rate_count.sum().item()}]")
                self.last_condition_state = True
                self.success_rate_count[:] = 0
        else:
            self.last_condition_state = False

        # 4. 决定重置的环境
        reset_envs = arrived | collided | time_out  # 通常超时也需要重置，除非由外部 runner 处理W

        # 记录到 extras 中，方便 RL 框架读取
        self.extras["success"] = arrived
        self.extras["collided"] = collided
        self.extras["time_out"] = time_out

        return reset_envs, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 根据配置选择随机重置或固定重置
        # self._randomize_grid_positions(env_ids)
        # self._fixed_reset_positions(env_ids)
        self._corner_diagonal_reset_positions(env_ids)

        self._update_toa_cache(env_ids)

        # create markers if necessary for the first time
        # set their visibility to true
        self.goal_pos_visualizer.set_visibility(True)
        self.goal_pos_visualizer.visualize(self.target_pos)

        # 重置图像缓冲区
        # self.img_buffer.reset_idx(env_ids)

    def _randomize_grid_positions(self, env_ids):
        """将机器人、目标和障碍物随机分配到网格位置。"""
        len_env_ids = len(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        # 将重置的环境位置偏移到对应环境的原点位置
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Grid Logic
        rand_vals = torch.rand((len_env_ids, 100), device=self.device)
        perm = torch.argsort(rand_vals, dim=1)  # (len_env_ids, 100)

        obs_grid_indices = perm[:, :60]  # (len_env_ids, 60)
        start_grid_indices = perm[:, 60]  # (len_env_ids,)
        goal_grid_indices = perm[:, 61]  # (len_env_ids,)

        def get_grid_coords_batch(indices):
            row = indices // self.grid_size
            col = indices % self.grid_size
            # -4.5 to 4.5
            x_center = (row - (self.grid_size / 2 - 0.5)) * self.cell_size
            y_center = (col - (self.grid_size / 2 - 0.5)) * self.cell_size
            return x_center, y_center

        # Place Obstacles
        if len(self.obstacles) > 0:
            for k in range(len(self.obstacles)):
                if k >= 60:
                    break

                grid_idx = obs_grid_indices[:, k]
                cx, cy = get_grid_coords_batch(grid_idx)
                r = self.obstacle_radii_tensor[k]

                max_offset = (self.cell_size / 2.0) - r
                max_offset = torch.clamp(max_offset, min=0.0)

                offset_x = (torch.rand(len_env_ids, device=self.device) * 2 - 1) * max_offset
                offset_y = (torch.rand(len_env_ids, device=self.device) * 2 - 1) * max_offset

                obs_x = cx + offset_x
                obs_y = cy + offset_y

                obs_defaults = self.obstacles[k].data.default_root_state[env_ids].clone()
                obs_defaults[:, 0] = obs_x + self.scene.env_origins[env_ids, 0]
                obs_defaults[:, 1] = obs_y + self.scene.env_origins[env_ids, 1]

                self.obstacles[k].write_root_pose_to_sim(obs_defaults[:, :7], env_ids)
                self.obstacles[k].write_root_velocity_to_sim(obs_defaults[:, 7:], env_ids)

        # Place Robot (Start)
        cx_s, cy_s = get_grid_coords_batch(start_grid_indices)
        offset_x_s = (torch.rand(len_env_ids, device=self.device) * 2 - 1) * (self.cell_size / 2 - 0.5)
        offset_y_s = (torch.rand(len_env_ids, device=self.device) * 2 - 1) * (self.cell_size / 2 - 0.5)
        start_x = cx_s + offset_x_s
        start_y = cy_s + offset_y_s

        # Overwrite Robot X, Y
        default_root_state[:, 0] = start_x + self.scene.env_origins[env_ids, 0]
        default_root_state[:, 1] = start_y + self.scene.env_origins[env_ids, 1]

        # 设置机器人速度
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Place Target (Goal)
        cx_g, cy_g = get_grid_coords_batch(goal_grid_indices)
        offset_x_g = (torch.rand(len_env_ids, device=self.device) * 2 - 1) * (self.cell_size / 2 - 0.2)
        offset_y_g = (torch.rand(len_env_ids, device=self.device) * 2 - 1) * (self.cell_size / 2 - 0.2)
        target_x = cx_g + offset_x_g
        target_y = cy_g + offset_y_g
        target_z = torch.ones_like(target_x) * 2.0

        xyz = torch.stack((target_x, target_y, target_z), dim=-1)
        self.target_pos[env_ids] = xyz + self.scene.env_origins[env_ids]

        # 获取目标点相对机器人位置的yaw角度
        target_yaw = get_target_direction(self.target_pos[env_ids, :2], default_root_state[:, :2])
        target_quat = yaw_to_quaternion(target_yaw)

        # 确保四元数归一化
        quat_norm = torch.norm(target_quat, dim=-1, keepdim=True)
        target_quat = target_quat / (quat_norm + 1e-8)

        # 设置机器人朝向目标点
        default_root_state[:, 3:7] = target_quat

        # 设置机器人位置
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        # 初始化控制器的目标点为当前重置后的位置和朝向
        self.target_pos_setpoint[env_ids] = default_root_state[:, :3]
        # 从归一化后的四元数提取 yaw
        w, x, y, z = torch.unbind(target_quat, dim=-1)
        self.target_yaw_setpoint[env_ids] = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).unsqueeze(-1)

    def _fixed_reset_positions(self, env_ids):
        """保持障碍物在原始位置，无人机位于原点，目标点在边界随机采样。"""
        len_env_ids = len(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        # 将重置的环境位置偏移到对应环境的原点位置
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 1. 障碍物保持在 config 定义的初始位置 (default_root_state)
        if len(self.obstacles) > 0:
            for k in range(len(self.obstacles)):
                obs_defaults = self.obstacles[k].data.default_root_state[env_ids].clone()
                obs_defaults[:, :3] += self.scene.env_origins[env_ids]
                self.obstacles[k].write_root_pose_to_sim(obs_defaults[:, :7], env_ids)
                self.obstacles[k].write_root_velocity_to_sim(obs_defaults[:, 7:], env_ids)

        # 2. 无人机位于各环境原点 (0, 0, 2)
        default_root_state[:, 0:2] = self.scene.env_origins[env_ids, 0:2]
        default_root_state[:, 2] = 2.0
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # 3. 目标点在环境边界上（25x25范围，即 [-12.5, 12.5]）随机采样
        # 边界范围
        limit = 11.5  # 略小于 12.5 以防墙体遮挡

        # 随机选择四条边中的一条
        side = torch.randint(0, 4, (len_env_ids,), device=self.device)
        pos_on_side = (torch.rand(len_env_ids, device=self.device) * 2 - 1) * limit

        target_x = torch.zeros(len_env_ids, device=self.device)
        target_y = torch.zeros(len_env_ids, device=self.device)

        # Side 0: North (y = limit), Side 1: South (y = -limit), Side 2: East (x = limit), Side 3: West (x = -limit)
        mask0 = side == 0
        target_x[mask0] = pos_on_side[mask0]
        target_y[mask0] = limit

        mask1 = side == 1
        target_x[mask1] = pos_on_side[mask1]
        target_y[mask1] = -limit

        mask2 = side == 2
        target_x[mask2] = limit
        target_y[mask2] = pos_on_side[mask2]

        mask3 = side == 3
        target_x[mask3] = -limit
        target_y[mask3] = pos_on_side[mask3]

        target_z = torch.ones_like(target_x) * 2.0
        xyz = torch.stack((target_x, target_y, target_z), dim=-1)
        self.target_pos[env_ids] = xyz + self.scene.env_origins[env_ids]

        # 4. 设置机器人朝向目标点
        target_yaw = get_target_direction(self.target_pos[env_ids, :2], default_root_state[:, :2])
        target_quat = yaw_to_quaternion(target_yaw)
        default_root_state[:, 3:7] = target_quat

        # 设置机器人位置
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        # 5. 初始化控制器状态
        self.target_pos_setpoint[env_ids] = default_root_state[:, :3]
        w, x, y, z = torch.unbind(target_quat, dim=-1)
        self.target_yaw_setpoint[env_ids] = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).unsqueeze(-1)

    def _corner_diagonal_reset_positions(self, env_ids):
        """障碍物随机网格重置；无人机起点为四角之一，目标为对角角点。"""
        len_env_ids = len(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        # 将重置的环境位置偏移到对应环境的原点位置
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 1. 障碍物重置方式与 _randomize_grid_positions 一致
        rand_vals = torch.rand((len_env_ids, 100), device=self.device)
        perm = torch.argsort(rand_vals, dim=1)  # (len_env_ids, 100)
        obs_grid_indices = perm[:, :60]  # (len_env_ids, 60)

        def get_grid_coords_batch(indices):
            row = indices // self.grid_size
            col = indices % self.grid_size
            x_center = (row - (self.grid_size / 2 - 0.5)) * self.cell_size
            y_center = (col - (self.grid_size / 2 - 0.5)) * self.cell_size
            return x_center, y_center

        if len(self.obstacles) > 0:
            for k in range(len(self.obstacles)):
                if k >= 60:
                    break

                grid_idx = obs_grid_indices[:, k]
                cx, cy = get_grid_coords_batch(grid_idx)
                r = self.obstacle_radii_tensor[k]

                max_offset = (self.cell_size / 2.0) - r
                max_offset = torch.clamp(max_offset, min=0.0)

                offset_x = (torch.rand(len_env_ids, device=self.device) * 2 - 1) * max_offset
                offset_y = (torch.rand(len_env_ids, device=self.device) * 2 - 1) * max_offset

                obs_x = cx + offset_x
                obs_y = cy + offset_y

                # 确保中心点 3m 内没有障碍物 (x^2 + y^2 < 3^2)
                dist_sq = obs_x**2 + obs_y**2
                safe_mask = dist_sq < 3.0**2

                obs_defaults = self.obstacles[k].data.default_root_state[env_ids].clone()
                obs_defaults[:, 0] = obs_x + self.scene.env_origins[env_ids, 0]
                obs_defaults[:, 1] = obs_y + self.scene.env_origins[env_ids, 1]

                # 如果落在 3m 内，将 Z 坐标设为 -10.0（挪到地面以下）
                if safe_mask.any():
                    obs_defaults[safe_mask, 2] = -10.0

                self.obstacles[k].write_root_pose_to_sim(obs_defaults[:, :7], env_ids)
                self.obstacles[k].write_root_velocity_to_sim(obs_defaults[:, 7:], env_ids)

        # 2. 无人机中心点 (0,0)，目标为 (15,15), (-15,15), (-15,-15), (15,-15) 之一
        goals_xy = torch.tensor(
            [
                [15.0, 15.0],
                [-15.0, 15.0],
                [-15.0, -15.0],
                [15.0, -15.0],
            ],
            device=self.device,
            dtype=default_root_state.dtype,
        )
        goal_idx = torch.randint(0, 4, (len_env_ids,), device=self.device)
        goal_xy = goals_xy[goal_idx]

        start_xy = torch.zeros((len_env_ids, 2), device=self.device, dtype=default_root_state.dtype)

        default_root_state[:, 0] = start_xy[:, 0] + self.scene.env_origins[env_ids, 0]
        default_root_state[:, 1] = start_xy[:, 1] + self.scene.env_origins[env_ids, 1]
        default_root_state[:, 2] = 2.0

        target_z = torch.full((len_env_ids,), 2.0, device=self.device)
        xyz = torch.cat([goal_xy, target_z.unsqueeze(-1)], dim=-1)
        self.target_pos[env_ids] = xyz + self.scene.env_origins[env_ids]

        # 3. 朝向目标并写入无人机状态
        target_yaw = get_target_direction(self.target_pos[env_ids, :2], default_root_state[:, :2])
        target_quat = yaw_to_quaternion(target_yaw)
        quat_norm = torch.norm(target_quat, dim=-1, keepdim=True)
        target_quat = target_quat / (quat_norm + 1e-8)
        default_root_state[:, 3:7] = target_quat

        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        self.target_pos_setpoint[env_ids] = default_root_state[:, :3]
        w, x, y, z = torch.unbind(target_quat, dim=-1)
        self.target_yaw_setpoint[env_ids] = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).unsqueeze(-1)


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
            depth_norm: 形状为(num_envs, 12, 16, 1)的深度图张量
        Returns:
            combined_tensor: 形状为(num_envs, 3, 12, 16)的PyTorch张量
        """
        if self.buffer is None:
            h, w = depth_norm.shape[1], depth_norm.shape[2]
            self.buffer = torch.zeros(
                (self.num_envs, self.buffer_size, h, w),
                device=self.device,
                dtype=depth_norm.dtype,
            )
            # 缓存常数张量，避免每步出现 CPU 列表到 GPU Tensor 的同步拷贝
            self.idx_000 = torch.tensor([0, 0, 0], device=self.device)
            self.idx_100 = torch.tensor([1, 0, 0], device=self.device)
            self._batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, 3)

        new_imgs = depth_norm.squeeze(-1)  # (num_envs, h, w)

        # 处理已满的缓冲区
        full_mask = self.buffer_full
        if full_mask.any():
            self.buffer[full_mask] = torch.roll(self.buffer[full_mask], shifts=-1, dims=1)
            self.buffer[full_mask, -1] = new_imgs[full_mask]

        # 处理未满的缓冲区
        not_full_mask = ~full_mask
        if not_full_mask.any():
            env_idx = torch.where(not_full_mask)[0]
            size_idx = self.current_sizes[env_idx]
            self.buffer[env_idx, size_idx] = new_imgs[env_idx]
            self.current_sizes[env_idx] += 1
            self.buffer_full = self.buffer_full | (self.current_sizes >= self.buffer_size)

        # 计算索引
        latest_idx = torch.where(self.buffer_full, self.buffer_size - 1, self.current_sizes - 1)
        middle_idx = torch.where(self.buffer_full, self.buffer_size // 2, self.current_sizes // 2)
        oldest_idx = torch.zeros_like(latest_idx)

        indices = torch.stack([latest_idx, middle_idx, oldest_idx], dim=1)  # (num_envs, 3)

        # 特殊情况处理
        size_1 = self.current_sizes == 1
        indices[size_1] = self.idx_000

        size_2 = self.current_sizes == 2
        indices[size_2] = self.idx_100

        # 收集图像
        return self.buffer[self._batch_idx, indices]

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
    reward_toa: torch.Tensor,
    collided: torch.Tensor,
    penalty_smooth: torch.Tensor,
    arrived: torch.Tensor,
    penalty_obstacle: torch.Tensor,
    penalty_yaw: torch.Tensor,
):
    total_reward = reward_toa * 1.0
    total_reward = total_reward - penalty_yaw * 1.0
    # total_reward = total_reward + reward_velocity * 0.5
    # total_reward = total_reward - penalty_obstacle
    total_reward = total_reward - penalty_smooth * 0.1
    total_reward = total_reward - collided * 200.0
    total_reward = total_reward + arrived * 300.0

    # print("Reward Velocity:", reward_velocity)
    # print("Reward TOA:", reward_toa)
    # print("Penalty Smooth:", penalty_smooth)
    # print("Collided:", collided)
    # print("Arrived:", arrived)
    # print("Penalty Obstacle:", penalty_obstacle)
    # print("Penalty Yaw:", penalty_yaw)
    return total_reward  # 返回 (num_envs,) 以匹配 SB3 预期


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
    direction_vector = target_pos - robot_pos  # 目标点相对机器人的位置 (num_reset_envs, 2)[env_id, x, y]
    direction_vector = direction_vector / torch.norm(direction_vector, dim=-1, keepdim=True)
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
