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

    def _setup_scene(self):
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
        self.actions = actions.clone()

    def _apply_action(self) -> None:
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
        obs1 = self.robot1.data.root_state_w[:, :2]
        obs2 = self.target_pos[:, :2]
        observations = {"policy": torch.cat((obs1, obs2), dim=-1)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.robot1.data.root_pos_w[:, :2],
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

        default_root_state = self.robot1.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot1.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot1.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

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
