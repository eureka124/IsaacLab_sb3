# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .tutorial_env_cfg import TutorialEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers


class DepthImageBuffer:
    """
    向量化的深度图缓冲区，支持并行处理所有环境
    """

    def __init__(self, num_envs, buffer_size=49, height=16, width=16, device="cuda:0"):
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.device = device
        self.height = height
        self.width = width

        # 初始化缓冲区张量 (num_envs, buffer_size, H, W)
        # 使用 zeros 初始化，相当于预填充了0
        self.buffer = torch.zeros((num_envs, buffer_size, height, width), device=device, dtype=torch.float32)

        # 记录每个环境当前存储的帧数
        self.counts = torch.zeros(num_envs, device=device, dtype=torch.long)


    def update_buffer(self, new_frames):
        """
        更新缓冲区并返回堆叠后的帧
        Args:
            new_frames: (num_envs, H, W) or (num_envs, H, W, 1)
        Returns:
            combined: (num_envs, H, W, 3)
        """
        # 确保输入维度正确 (num_envs, H, W)
        if new_frames.dim() == 4:
            new_frames = new_frames.squeeze(-1)

        # 1. 缓冲区移位：丢弃最旧的，腾出位置给新的
        # 这种操作在GPU上很快，比循环快得多
        self.buffer[:, :-1] = self.buffer[:, 1:].clone()
        self.buffer[:, -1] = new_frames

        # 2. 更新计数
        self.counts = torch.clamp(self.counts + 1, max=self.buffer_size)

        # 3. 计算索引以提取三帧 (最新, 中间, 最旧)
        # 有效数据范围在 [buffer_size - count, buffer_size - 1]

        # 最新帧索引总是最后一个
        idx_newest = torch.full((self.num_envs,), self.buffer_size - 1, device=self.device, dtype=torch.long)

        # 最旧帧索引
        idx_oldest = self.buffer_size - self.counts

        # 中间帧索引 (相对于最旧帧的偏移)
        idx_middle = idx_oldest + (self.counts // 2)

        # 处理计数较少时的特殊情况 (模仿原代码逻辑)
        # 如果 count=1: new=old=middle
        # 如果 count=2: 原代码是 [new, old, old]，这里简化为 [new, middle, old] 逻辑通用

        # 4. 收集帧
        # 构建索引张量 (num_envs, 3)
        frame_indices = torch.stack([idx_newest, idx_middle, idx_oldest], dim=1)

        # 使用 gather 或高级索引提取
        # batch_indices: [[0,0,0], [1,1,1], ...]
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, 3)

        # 提取数据 (num_envs, 3, H, W)
        selected_frames = self.buffer[batch_indices, frame_indices]

        # 调整维度为 (num_envs, H, W, 3) 以匹配后续处理
        return selected_frames.permute(0, 2, 3, 1)

    def reset(self, env_ids):
        """重置指定环境的缓冲区"""
        if len(env_ids) > 0:
            self.buffer[env_ids] = 0
            self.counts[env_ids] = 0


class TutorialEnv(DirectRLEnv):
    cfg: TutorialEnvCfg

    def __init__(self, cfg: TutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # print("__init__")
        # 创建可视化目标位置的标记
        self.target_pos = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
        # 目标位置可视化
        marker_cfg.prim_path = "/Visuals/Command/goal_position"
        self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
        self.depth_buffer: DepthImageBuffer = DepthImageBuffer(self.cfg.scene.num_envs)
        self.arrived = torch.zeros(self.cfg.scene.num_envs, device=self.device, dtype=torch.bool)

    def _setup_scene(self):
        # print("_setup_scene")
        self.robot1 = self.scene["robot1_cfg"]
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
        # print("_pre_physics_step")
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # print("_apply_action")
        now_v = torch.zeros((self.num_envs, 6), device=self.device)
        now_v[:, 0:2] = self.actions

        self.target_z = 2
        now_pos = self.robot1.data.root_pos_w.clone()
        now_z = now_pos[:, 2]
        error_z = self.target_z - now_z
        linel_v_z = now_v[:, 2]  # 形状 (num_envs,1)
        linel_v_z.copy_(error_z + 0.1)  #

        self.robot1.write_root_velocity_to_sim(now_v, env_ids=None)

    def _get_observations(self) -> dict:
        # print("_get_observations")
        depth_frame = (
            self.scene["camera"].data.output["distance_to_image_plane"].clone()
        )
        depth_frame = torch.nan_to_num(
            depth_frame,
            nan=0.0,
            posinf=10,  # depth_frame[depth_frame != float('inf')].max(),
            neginf=0,  # depth_frame.min()
        )
        self.depth_imge_for_reward = depth_frame

        max_vals = 10  # 相机最远探测距离m
        depth_frame[depth_frame >= max_vals] = max_vals
        depth_norm = depth_frame / (max_vals + 1e-8)  # +1e-8防止除零
        combined_tensor = self.depth_buffer.update_buffer(depth_norm)

        obs1 = self.robot1.data.root_state_w[:, :2]
        obs2 = self.target_pos[:, :2]

        observations = {
            "policy": torch.cat((obs1, obs2), dim=-1),
            # "camera": combined_tensor
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # print("_get_rewards")
        robot_pos = self.robot1.data.root_pos_w.clone()
        target_pos = self.target_pos.clone()
        distance = torch.linalg.norm(robot_pos - target_pos, dim=-1)
        # print("distance:", distance)
        arrived = (distance <= 0.2).squeeze(dim=-1)
        print(distance)
        total_reward = -distance + arrived * 5.0
        # print(total_reward)
        self.arrived = arrived
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # print("_get_dones")
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        arrived = self.arrived
        out_of_bounds = time_out
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        print("_reset_idx")

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot1.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        print(self.arrived)
        self.robot1.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot1.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.arrived[env_ids] = False
        print(self.arrived)

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
                result.append(xyz[xyz_indes : xyz_indes + 1, :])
                xyz_indes += 1
            else:
                result.append(self.target_pos[i : i + 1, :])

        self.target_pos = torch.cat(result, dim=0) + self.scene.env_origins
        # create markers if necessary for the first time
        # set their visibility to true
        self.goal_pos_visualizer.set_visibility(True)
        self.goal_pos_visualizer.visualize(self.target_pos)
