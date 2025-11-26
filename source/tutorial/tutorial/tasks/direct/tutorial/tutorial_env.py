# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import collections

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .tutorial_env_cfg import TutorialEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers


class TutorialEnv(DirectRLEnv):
    cfg: TutorialEnvCfg

    def __init__(self, cfg: TutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 创建可视化目标位置的标记
        self.target_pos = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
        # 目标位置可视化
        marker_cfg.prim_path = "/Visuals/Command/goal_position"
        self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
        self.img_buffers = [DepthImageBuffer() for _ in range(self.cfg.scene.num_envs)]

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
        light_cfg = sim_utils.DomeLightCfg(intensity=4000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        now_v = torch.zeros((self.num_envs, 6), device=self.device)
        now_v[:, 0:2] = self.actions

        self.target_z = 2
        now_pos = self.robot.data.root_pos_w.clone()
        now_z = now_pos[:, 2]
        error_z = self.target_z - now_z
        linel_v_z = now_v[:, 2]  # 形状 (num_envs,1)
        linel_v_z.copy_(error_z + 0.1)  #

        self.robot.write_root_velocity_to_sim(now_v, env_ids=None)

    def get_norm_depth_image(self) -> torch.Tensor:
        depth_frame = self.scene["camera"].data.output["distance_to_image_plane"].clone()
        # 归一化深度图
        depth_frame = torch.nan_to_num(
            depth_frame,
            nan=0.0,
            posinf=10,  # depth_frame[depth_frame != float('inf')].max(),
            neginf=0,  # depth_frame.min()
        )
        max_vals = 10  # 相机最远探测距离m
        depth_frame[depth_frame >= max_vals] = max_vals
        depth_norm = depth_frame / (max_vals + 1e-8)  # +1e-8防止除零
        # print(depth_norm.shape) # (16, 16, 1)
        return depth_norm # shape:[env_num, 16, 16, 1]

    def _get_observations(self) -> dict:
        depth_norm = self.get_norm_depth_image()
        camera_observation = torch.zeros((self.cfg.scene.num_envs, 3, 16, 16))  # shape=[num_envs, 3, 16, 16]
        for env in range(self.cfg.scene.num_envs):
            camera_observation[env] = self.img_buffers[env].update_buffer(depth_norm[env])
            # print(depth_norm[env].shape, camera_observation[env].shape)# (3, 16, 16)

        obs1 = self.robot.data.root_state_w[:, :2]
        obs2 = self.target_pos[:, :2]

        observations = {"robot-state": torch.cat((obs1, obs2), dim=-1),
                        "camera": camera_observation
                        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.robot.data.root_pos_w[:, :2],
            self.target_pos[:, :2]
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = time_out

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        len_env_ids = len(env_ids)
        x = (
            torch.zeros(len_env_ids, device=self.device)
            .uniform_(-1, 1)
            .unsqueeze(dim=1)
        )
        y = (
            torch.zeros(len_env_ids, device=self.device)
            .uniform_(-1, 1)
            .unsqueeze(dim=1)
        )
        z = torch.zeros(len_env_ids, device=self.device).uniform_(2, 2).unsqueeze(dim=1)
        xyz = torch.cat((x, y, z), dim=-1)

        result = []
        xyz_indes = 0
        for i in range(0, self.cfg.scene.num_envs):
            if i in env_ids:
                result.append(xyz[xyz_indes: xyz_indes + 1, :])
                xyz_indes += 1
            else:
                result.append(self.target_pos[i: i + 1, :])

        self.target_pos = torch.cat(result, dim=0) + self.scene.env_origins
        # create markers if necessary for the first time
        # set their visibility to true
        self.goal_pos_visualizer.set_visibility(True)
        self.goal_pos_visualizer.visualize(self.target_pos)


@torch.jit.script
def compute_rewards(
        robot_pos: torch.Tensor,
        target_pos: torch.Tensor,
):
    distance = torch.linalg.norm(robot_pos - target_pos, dim=-1)
    # print("distance:", distance)
    arrived = (distance <= 0.05).squeeze(dim=-1)
    total_reward = -torch.linalg.norm(robot_pos - target_pos, dim=-1) + arrived * 5.0
    return total_reward


class DepthImageBuffer:
    def __init__(self, buffer_size=49):
        """
        初始化深度图缓冲区
        Args:
            buffer_size: 缓冲区大小，默认为49
        """
        self.buffer_size = buffer_size
        self.buffer = None  # 用于存储所有深度图的张量
        self.current_size = 0  # 当前缓冲区中的图像数量
        self.buffer_full = False  # 缓冲区是否已满

    def update_buffer(self, depth_norm):
        """
        使用torch.roll优化缓冲区更新
        Args:
            depth_norm: 形状为(16,16,1)的深度图张量（GPU张量）
        Returns:
            combined_tensor: 形状为(16,16,3)的PyTorch张量（在GPU上）
        """
        h, w, c = depth_norm.shape

        # 初始化缓冲区
        if self.buffer is None:
            self.buffer = torch.zeros(self.buffer_size, h, w,
                                      device=depth_norm.device,
                                      dtype=depth_norm.dtype)

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
            latest_idx = self.current_size - 1 if not self.buffer_full else self.buffer_size - 1
            middle_idx = self.current_size // 2 if not self.buffer_full else self.buffer_size // 2
            oldest_idx = 0
            indices = [latest_idx, middle_idx, oldest_idx]

        # 从缓冲区提取对应的图像并重新组合
        selected_imgs = [self.buffer[i].unsqueeze(0) for i in indices]
        combined_tensor = torch.cat(selected_imgs, dim=0)

        return combined_tensor

    def get_buffer_state(self):
        """获取缓冲区状态信息"""
        return {
            "buffer_shape": self.buffer.shape if self.buffer is not None else None,
            "current_size": self.current_size,
            "buffer_full": self.buffer_full
        }

    def clear_buffer(self):
        """
        清空缓冲区，移除所有存储的深度图[6,8](@ref)
        """
        self.buffer.clear()

    def get_buffer_status(self):
        """获取缓冲区状态"""
        return f"当前缓冲区大小: {len(self.buffer)}/{self.buffer_size}"

    def is_buffer_full(self):
        """
        检查缓冲区是否已满
        Returns:
            bool: 缓冲区已满返回True，否则返回False
        """
        return len(self.buffer) >= self.buffer_size

    def get_current_size(self):
        """
        获取当前缓冲区中的元素数量
        Returns:
            int: 当前缓冲区中的图片数量
        """
        return len(self.buffer)
