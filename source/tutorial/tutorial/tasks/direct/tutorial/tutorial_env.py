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

    # 根据目标位置和机器人位置计算yaw角度
    def _get_target_direction(self, target_pos, robot_pos) -> torch.Tensor:
        pass

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
        now_v[:, 0:2] = self.actions  # vx, vy

        self.target_z = 2
        now_pos = self.robot.data.root_pos_w.clone()
        now_z = now_pos[:, 2]
        error_z = self.target_z - now_z
        linel_v_z = now_v[:, 2]  # 形状 (num_envs,1)
        linel_v_z.copy_(error_z + 0.1)  #

        self.robot.write_root_velocity_to_sim(now_v, env_ids=None)

    def _get_norm_depth_image(self) -> torch.Tensor:
        depth_frame = self.scene["camera"].data.output["distance_to_image_plane"].clone()
        # 归一化深度图
        max_vals = 5  # 相机最远探测距离m
        depth_frame = torch.nan_to_num(
            depth_frame,
            nan=0.0,
            posinf=max_vals,  # depth_frame[depth_frame != float('inf')].max(),
            neginf=0,  # depth_frame.min()
        )
        
        depth_frame[depth_frame >= max_vals] = max_vals
        depth_norm = depth_frame / (max_vals + 1e-8)  # +1e-8防止除零
        # print(depth_norm.shape) # (env_num, 1, 16, 16)
        return depth_norm  # shape:[env_num, 1, 16, 16]

    def _get_observations(self) -> dict:
        depth_norm = self._get_norm_depth_image()
        camera_observation = torch.zeros((self.cfg.scene.num_envs, 3, 16, 16), device=self.device)  # shape=[num_envs, 3, 16, 16]
        for env in range(self.cfg.scene.num_envs):
            camera_observation[env] = self.img_buffers[env].update_buffer(depth_norm[env])
            # print(depth_norm[env].shape, camera_observation[env].shape)# (3, 16, 16)
        # print(camera_observation)

        robot_pos = self.robot.data.root_state_w[:, :2]  # 位置(x,y)
        relative_position = self.target_pos[:, :2] - robot_pos  # 目标位置相对于机器人的位置差 (num_envs, 2)

        observations = {
            "policy":
            {
                "relative-position": relative_position,
                "camera": camera_observation
            }
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.distances,
            self.arrived,
        )
        # print(self.robot.data)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 计算是否到达目标位置
        pos = self.robot.data.root_pos_w.clone()  # 机器人位置 (num_envs, 3)
        # print(pos.shape)  # (num_envs, 3)
        direction_vector = self.target_pos - pos  # 目标点相对机器人的位置 (num_envs, 3)  计算方向向量
        # print(direction_vector.shape)  # (num_envs, 3)
        direction_vector = torch.cat(
            (direction_vector[:, 0:2], torch.zeros(direction_vector.shape[0], 1, device=direction_vector.device)),
            dim=-1,
        )  # 将z轴设为0
        # print(direction_vector.shape)  # (num_envs, 3)
        distances = torch.norm(direction_vector, p=2, dim=1, keepdim=True)  # 计算距离
        # print(distances.shape)  # (num_envs, 1)
        direction_vector = direction_vector / torch.norm(
            direction_vector, dim=-1, keepdim=True
        )
        self.distances = distances.squeeze(dim=1)  # (num_envs,)

        arrived = (distances <= 0.05).squeeze(dim=1)
        self.arrived = arrived

        return arrived, time_out

    def _reset_idx(self, env_ids):
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
        print("Sampled target positions for reset envs:", xyz)

        # 向量化更新：
        # 1. xyz 是局部坐标，加上对应环境的原点坐标 (env_origins[env_ids]) 转换为全局坐标
        # 2. 直接使用索引 [env_ids] 更新 self.target_pos 中对应的行，未重置的环境保持不变
        self.target_pos[env_ids] = xyz + self.scene.env_origins[env_ids]

        print("环境原点", self.scene.env_origins)
        print("目标位置", self.target_pos)
        # create markers if necessary for the first time
        # set their visibility to true
        self.goal_pos_visualizer.set_visibility(True)
        self.goal_pos_visualizer.visualize(self.target_pos)


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

    def get_buffer_state(self) -> dict:
        """获取缓冲区状态信息"""
        return {
            "buffer_shape": self.buffer.shape if self.buffer is not None else None,
            "current_size": self.current_size,
            "buffer_full": self.buffer_full
        }

    def clear_buffer(self):
        """
        清空缓冲区，移除所有存储的深度图
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


@torch.jit.script
def compute_rewards(
    distance: torch.Tensor,
    arrived: torch.Tensor,
):
    total_reward = -distance + arrived * 5.0
    return total_reward


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为欧拉角（以弧度为单位）
    Args:
        quat: 形状为 (..., 4) 的四元数张量，格式为 (w, x, y, z)
    Returns:
        euler: 形状为 (..., 3) 的欧拉角张量，格式为 (roll, pitch, yaw)
    """
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    # 计算滚转角 (roll)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    # 计算俯仰角 (pitch)
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    # 计算偏航角 (yaw)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    euler = torch.stack((roll, pitch, yaw), dim=-1)
    return euler
