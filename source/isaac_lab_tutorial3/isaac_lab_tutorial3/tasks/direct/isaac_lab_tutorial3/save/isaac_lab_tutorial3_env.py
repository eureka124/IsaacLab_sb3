# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections.abc import Sequence
import collections
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation ,RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .isaac_lab_tutorial3_env_cfg import IsaacLabTutorial3EnvCfg

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

#可视化
def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "drone_0": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "drone_1": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "drone_2": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
                "drone_3": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
                ),
            "cube": sim_utils.CuboidCfg(

                size=(1.0, 1.0, 1.0),

                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),

            ),

            "sphere": sim_utils.SphereCfg(

                radius=0.5,

                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),

            ),

            "cylinder": sim_utils.CylinderCfg(

                radius=0.5,

                height=1.0,

                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),

            ),

            "cone": sim_utils.ConeCfg(

                radius=0.5,

                height=1.0,

                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),

            ),

        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class IsaacLabTutorial3Env(DirectRLEnv):
    cfg: IsaacLabTutorial3EnvCfg

    def __init__(self, cfg: IsaacLabTutorial3EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        ###################################目标点位置，兼任可视化目标点位置
        self.target_pos = torch.tensor([0.0, 0.0, 2.0], 
                                dtype=torch.float32,
                                device=self.cfg.sim.device)
        self.target_pos = self.target_pos.unsqueeze(0).expand(self.cfg.scene.num_envs, 3) 
        ###################################

        #################################可视化图形的rot
        self.target_pos_rot = torch.tensor([1.0, 0.0, 0.0,0.0], 
                                dtype=torch.float32,
                                device=self.cfg.sim.device)
        self.target_pos_rot = self.target_pos_rot.unsqueeze(0).expand(self.cfg.scene.num_envs, 4) 
        #################################

        #################################可视化相关代码  待优化
        self.visualization_markers = define_markers() #可视化
        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()

        self.visual_count=0
        self.visual_list0=[]
        self.visual_list1=[]
        self.visual_list2=[]
        self.visual_list3=[]
        self.visual_list4=[]
        self.visual_list5=[]
        self.visual_list6=[]
        self.visual_list7=[]    
        ################################

        ##################################特定形状的辅助0张量
        self.two_zero_stuff=torch.zeros((self.cfg.scene.num_envs, 2)).cuda()
        ##################################

        ##################################当前速度初始化 x,y,z,线速度和角速度
        self.now_v=self.robot.data.root_state_w[:, 7:13].clone()
        ####################################

        ##################################障碍物运动辅助变量
        self.my_time_count=0
        ##################################

        ######################################每个深度相机队列池 待优化
        self.image_buffer1 = DepthImageBuffer()
        self.image_buffer2 = DepthImageBuffer()
        self.image_buffer3 = DepthImageBuffer()
        self.image_buffer4 = DepthImageBuffer()
        ######################################

        ###################################
        self.last_condition_state = False       #25次导航打印数据的辅助变量
        self.success_rate_count=[0,0,0]         #记录成功率的数组
        ###################################

        ################################打印每项的奖励
        self.allrew_angle=0
        self.allrew_distance=0
        self.allrew_arrival=0
        self.allcollision_penalty=0
        self.allsmoothness_penalty=0  
        self.allreward=0      
        ################################
         
        ##################################构建专家经验池初始化代码
        self.STATE_SHAPES = {
            'camera': (1, 3, 16, 16),  # [batch, channels, height, width]
            'robot-state': (1, 7)      # [batch, robot_state_dim]
        }
        self.ACTION_DIM = 2

        # 初始化专家经验池
        self.expert_pool = ExpertExperiencePool(
            state_shapes=self.STATE_SHAPES,
            action_dim=self.ACTION_DIM,
            max_size=200000,
            save_threshold=100000,
            save_path="/home/yu/expert_pool/drone_expert_data.npz",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 初始化临时缓冲区
        self.temp_buffer = ExpertTempBuffer(
            state_shapes=self.STATE_SHAPES,
            action_dim=self.ACTION_DIM
        )
        ########################################

        #############################记录与目标点最小距离的变量,目标点到无人机距离，无人机机头朝向和无人机与目标连线的夹角
        self.target_min_distance = torch.full((self.cfg.scene.num_envs, 1), float('inf')).cuda()
        self.distances=torch.full((self.cfg.scene.num_envs, 1), float(0)).cuda()
        self.angles=torch.full((self.cfg.scene.num_envs, 1), float(0)).cuda()
        ##############################

    def _setup_scene(self):
        ############################场景构建
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.robot = self.scene["robot"]
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        ############################

        ############################可动障碍物的定义
        Move_Obstacle_0 = RigidObjectCfg(
            prim_path="/World/move_obstacle/Move_Obstacle_0",
            spawn=sim_utils.CylinderCfg(
                radius=0.3,
                height=2.5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(10, -1, 1.25),),
        )
        self.Move_Obstacle_0 = RigidObject(cfg=Move_Obstacle_0)
        Move_Obstacle_1 = RigidObjectCfg(
            prim_path="/World/move_obstacle/Move_Obstacle_1",
            spawn=sim_utils.CylinderCfg(
                radius=0.5,
                height=9,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-10, -2.1, 4.5),),
        )
        self.Move_Obstacle_1 = RigidObject(cfg=Move_Obstacle_1)
        Move_Obstacle_2 = RigidObjectCfg(
            prim_path="/World/move_obstacle/Move_Obstacle_2",
            spawn=sim_utils.CylinderCfg(
                radius=0.3,
                height=7,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-10, 2.1, 3.5),),
        )
        self.Move_Obstacle_2 = RigidObject(cfg=Move_Obstacle_2)
        Move_Obstacle_3 = RigidObjectCfg(
            prim_path="/World/move_obstacle/Move_Obstacle_3",
            spawn=sim_utils.CylinderCfg(
                radius=0.6,
                height=5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(10, 2.5, 2.5),),
        )
        self.Move_Obstacle_3 = RigidObject(cfg=Move_Obstacle_3)
        Move_Obstacle_4 = RigidObjectCfg(
            prim_path="/World/move_obstacle/Move_Obstacle_4",
            spawn=sim_utils.CylinderCfg(
                radius=0.3,
                height=2.4,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-1, 10, 1.2),),
        )
        self.Move_Obstacle_4= RigidObject(cfg=Move_Obstacle_4)
        Move_Obstacle_5 = RigidObjectCfg(
            prim_path="/World/move_obstacle/Move_Obstacle_5",
            spawn=sim_utils.CylinderCfg(
                radius=0.5,
                height=9,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.1, -10, 4.5),),
        )
        self.Move_Obstacle_5 = RigidObject(cfg=Move_Obstacle_5)
        Move_Obstacle_6 = RigidObjectCfg(
            prim_path="/World/move_obstacle/Move_Obstacle_6",
            spawn=sim_utils.CylinderCfg(
                radius=0.3,
                height=7,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2.1, -10, 3.5),),
        )
        self.Move_Obstacle_6 = RigidObject(cfg=Move_Obstacle_6)
        Move_Obstacle_7 = RigidObjectCfg(
            prim_path="/World/move_obstacle/Move_Obstacle_7",
            spawn=sim_utils.CylinderCfg(
                radius=0.6,
                height=5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, 10, 2.5),),
        )
        self.Move_Obstacle_7 = RigidObject(cfg=Move_Obstacle_7)
        ##########################################

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        #################################
        self.actions = actions.clone() 
        #################################

        #################################专家经验池构建，只取第一个无人机的动作
        self.expert_action=actions.clone()[0:1,...]
        #################################

        ############################可视化 待优化
        if self.visual_count==0:
          
            # get marker locations and orientations
            self.marker_locations = self.robot.data.root_pos_w+self.marker_offset
            self.forward_marker_orientations = self.robot.data.root_quat_w
            self.marker_target=self.target_pos+self.marker_offset
            self.marker_target_rot=self.target_pos_rot

            self.visual_list0.append(self.marker_locations[0:1,:])
            self.visual_list1.append(self.marker_locations[1:2,:])
            self.visual_list2.append(self.marker_locations[2:3,:])
            self.visual_list3.append(self.marker_locations[3:4,:])
            self.visual_list4.append(self.forward_marker_orientations[0:1,:])
            self.visual_list5.append(self.forward_marker_orientations[1:2,:])
            self.visual_list6.append(self.forward_marker_orientations[2:3,:])
            self.visual_list7.append(self.forward_marker_orientations[3:4,:])
            vl0=torch.cat(self.visual_list0,dim=0)
            vl1=torch.cat(self.visual_list1,dim=0)
            vl2=torch.cat(self.visual_list2,dim=0)
            vl3=torch.cat(self.visual_list3,dim=0)
            vl4=torch.cat(self.visual_list4,dim=0)
            vl5=torch.cat(self.visual_list5,dim=0)
            vl6=torch.cat(self.visual_list6,dim=0)
            vl7=torch.cat(self.visual_list7,dim=0)
            loc=torch.cat((vl0,vl1,vl2,vl3),dim=0)
            rots=torch.cat((vl4,vl5,vl6,vl7),dim=0)
            n0 = vl0.shape[0]
            n1 = vl1.shape[0]
            n2 = vl2.shape[0]
            n3 = vl3.shape[0]
            indices = torch.cat([
                torch.tensor([4], dtype=torch.long),
                torch.zeros(n0-1, dtype=torch.long),
                torch.tensor([5], dtype=torch.long),
                torch.ones(n1-1, dtype=torch.long),
                torch.tensor([6], dtype=torch.long),
                torch.full((n2-1,), 2, dtype=torch.long),
                torch.tensor([7], dtype=torch.long),
                torch.full((n3-1,), 3, dtype=torch.long)
            ])
            self.visualization_markers.visualize(loc, rots, marker_indices=indices)

        self.visual_count+=1
        if self.visual_count==30:
            self.visual_count=0
        ##############################################

        ########################################把前进速度和绕z轴旋转的速度，变成全局速度
        l_v=self.drone_head_direction*self.actions[:,0:1]    #self.drone_head_direction 表示当前机头朝向的归一化向量 
        a_v=torch.cat((self.two_zero_stuff,self.actions[:,1:2]),dim=-1)     #角速度拼接两个0 变成（0，0，角速度）

        self.target_action_vel=torch.cat((l_v,a_v),dim=-1)      #拼接完成的全局速度
        ########################################

        self.my_time_count=0
        move_ob_root_state0 = self.Move_Obstacle_0.data.default_root_state.clone()
        move_ob_root_state0[:,0:1]=8*math.sin(self.my_time_count)+10
        self.Move_Obstacle_0.write_root_pose_to_sim(move_ob_root_state0[:,0:7])

        move_ob_root_state1 = self.Move_Obstacle_1.data.default_root_state.clone()
        move_ob_root_state1[:,0:1]=-8*math.sin(self.my_time_count)-10
        self.Move_Obstacle_1.write_root_pose_to_sim(move_ob_root_state1[:,0:7])

        move_ob_root_state2 = self.Move_Obstacle_2.data.default_root_state.clone()
        move_ob_root_state2[:,0:1]=-8*math.cos(self.my_time_count)-10
        self.Move_Obstacle_2.write_root_pose_to_sim(move_ob_root_state2[:,0:7])

        move_ob_root_state3 = self.Move_Obstacle_3.data.default_root_state.clone()
        move_ob_root_state3[:,0:1]=8*math.cos(self.my_time_count)+10
        self.Move_Obstacle_3.write_root_pose_to_sim(move_ob_root_state3[:,0:7])

        move_ob_root_state4 = self.Move_Obstacle_4.data.default_root_state.clone()
        move_ob_root_state4[:,1:2]=8*math.sin(self.my_time_count)+10
        self.Move_Obstacle_4.write_root_pose_to_sim(move_ob_root_state4[:,0:7])

        move_ob_root_state5 = self.Move_Obstacle_5.data.default_root_state.clone()
        move_ob_root_state5[:,1:2]=-8*math.sin(self.my_time_count)-10
        self.Move_Obstacle_5.write_root_pose_to_sim(move_ob_root_state5[:,0:7])

        move_ob_root_state6 = self.Move_Obstacle_6.data.default_root_state.clone()
        move_ob_root_state6[:,1:2]=-8*math.cos(self.my_time_count)-10
        self.Move_Obstacle_6.write_root_pose_to_sim(move_ob_root_state6[:,0:7])

        move_ob_root_state7 = self.Move_Obstacle_7.data.default_root_state.clone()
        move_ob_root_state7[:,1:2]=8*math.cos(self.my_time_count)+10
        self.Move_Obstacle_7.write_root_pose_to_sim(move_ob_root_state7[:,0:7])
        #print(f"Obstacle state shape:", move_ob_root_state7.shape)
        root_state2 = self.robot.data.root_state_w[:, 0:7].clone()#无人机的姿态，本代码是增加姿态扰动
        # 假设 root_state2 是一个形状为 (n, 7) 的 PyTorch 张量
        n = root_state2.shape[0]

        # 1. 将角度范围转换为弧度
        roll_low = torch.tensor(-10 * math.pi / 180.0)  # -5 度
        roll_high = torch.tensor(10 * math.pi / 180.0)   # +5 度
        pitch_low = torch.tensor(-5 * math.pi / 180.0) # 
        pitch_high = torch.tensor(10 * math.pi / 180.0)   # 

        # 2. 为每个实例生成随机的 roll 和 pitch 角度（均匀分布）
        roll_new =(torch.rand(n) * (roll_high - roll_low) + roll_low).to(root_state2.device)
        pitch_new =( torch.rand(n) * (pitch_high - pitch_low) + pitch_low).to(root_state2.device)

        # 3. 提取原有的四元数 (后四列，顺序为 w, x, y, z)
        quat_old = root_state2[:, 3:7]

        # 4. 将原有四元数转换为欧拉角 (ZYX 顺序)
        roll_old, pitch_old, yaw_old = quat_to_euler_zyx(quat_old)

        # 5. 保持 yaw 不变，使用新的 roll 和 pitch
        yaw_new = yaw_old

        # 6. 从新的欧拉角构建新四元数
        quat_new = euler_to_quat_zyx(roll_new, pitch_new, yaw_new)
        self.pitch_new=pitch_new
        self.roll_new=roll_new
        # 7. 更新 root_state2 中的四元数部分
        root_state2[:, 3:7] = quat_new
        self.robot.write_root_pose_to_sim(root_state2)#robot


    def _apply_action(self) -> None:

        now_v=self.robot.data.root_state_w[:, 7:13].clone()

        now_v[:,0:2]=self.target_action_vel[:,0:2] # xy线速度

        now_v[:,5:6]=self.target_action_vel[:,5:6]  # z角速度

        ###################################### 控制z方向高度为2m
        self.target_z=2
        now_pos=self.robot.data.root_pos_w.clone()
        now_z=now_pos[:,2:3]
        error_z=self.target_z-now_z
        linel_v_z = now_v[:, 2:3]  # 形状 (num_envs,1)
        linel_v_z.copy_(error_z+0.1)  #
        ###################################
        tensor1=now_v[:,0:3]
        tensor2=now_v[:,5:6]
        now_v=torch.cat((tensor1,self.two_zero_stuff,tensor2),dim=-1)

        self.robot.write_root_velocity_to_sim(now_v, env_ids = None)

    def _get_observations(self) -> dict:
        depth_frame = self.scene["camera"].data.output["distance_to_image_plane"].clone()

        ###############################################归一化深度图
        depth_frame = torch.nan_to_num(
            depth_frame, 
            nan=0.0,
            posinf=10,#depth_frame[depth_frame != float('inf')].max(),
            neginf=0#depth_frame.min()
        )
        self.depth_imge_for_reward=depth_frame

        max_vals = 10 #相机最远探测距离m
        depth_frame[depth_frame >= max_vals] = max_vals
        depth_norm = (depth_frame ) / (max_vals  + 1e-8)  # +1e-8防止除零
        ##################################################

        ##################################把收集到的深度图放入队列中，从队列中拿出三帧有时间顺序的图叠加 待优化
        combined_tensor1 = self.image_buffer1.update_buffer(depth_norm[0:1,...])
        combined_tensor2 = self.image_buffer2.update_buffer(depth_norm[1:2,...])
        combined_tensor3 = self.image_buffer3.update_buffer(depth_norm[2:3,...])
        combined_tensor4 = self.image_buffer4.update_buffer(depth_norm[3:4,...])
        combined_tensor = torch.cat([combined_tensor1, combined_tensor2, combined_tensor3, combined_tensor4], dim=0)
        ##################################

        ############################打印深度图
        # depth_frame_test = depth_norm[1].squeeze(-1)
        # # np_array=depth_frame[0].squeeze(-1).detach().cpu().numpy()
        # # np.savetxt("tensor1.txt",np_array,fmt='%.3f')
        # # 处理正无穷的替换值
        # valid_depths = depth_frame_test[depth_frame_test != float('inf')]
        # posinf_value = valid_depths.max() if valid_depths.numel() > 0 else 0.0

        # # 处理负无穷的替换值（同样需要检查）
        # valid_depths_neg = depth_frame_test[torch.isfinite(depth_frame_test)]
        # neginf_value = valid_depths_neg.min() if valid_depths_neg.numel() > 0 else 0.0

        # depth_frame_test = torch.nan_to_num(
        #     depth_frame_test, 
        #     nan=10,
        #     posinf=posinf_value,
        #     neginf=neginf_value
        # )

        # # 添加小量防止除零
        # depth_min = depth_frame_test.min()
        # depth_max = depth_frame_test.max()
        # depth_norm_test = (depth_frame_test - depth_min) / (depth_max - depth_min + 1e-8)

        # img = TF.to_pil_image(depth_norm_test.unsqueeze(0))
        # img.save("/home/yu/Pictures/depth_image1.png")
        ####################################

        ############################从四元数计算机头朝向
        quat = self.robot.data.root_state_w[:, 3:7]
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        w,x, y, z = torch.split(quat,dim=-1,split_size_or_sections=1)
        x1=w*w+x*x-y*y-z*z
        y1=2*(x*y+w*z)
        z1=2*(x*z-w*y)
        horizon_direction = torch.cat((x1, y1, torch.zeros_like(z1)), dim=-1)
        normlized_direction=horizon_direction / torch.norm(horizon_direction, dim=-1, keepdim=True)
        self.drone_head_direction=normlized_direction #当前机头朝向向量
        #############################

        ##############################飞机指向目标点的向量，得到距离和归一化角度
        pos=self.robot.data.root_pos_w.clone()  
        direction_vector = self.target_pos - pos  # 形状 (num_envs, 3)  计算方向向量
        direction_vector=torch.cat((direction_vector[:,0:2],torch.zeros_like(z1)),dim=-1)
        distances = torch.norm(direction_vector, p=2, dim=1,keepdim=True)   #计算距离
        direction_vector=direction_vector / torch.norm(direction_vector, dim=-1, keepdim=True)
        ################################

        ################################
        self.last_distances=distances.clone()
        #self.distances=distances.clone()
        ################################

        ##############################计算机头朝向向量和飞机指向目标点的向量的角度可以用 atan2
        angles = signed_angle(self.drone_head_direction[:,:-1], direction_vector[:,:-1]).unsqueeze(dim=1)
        self.last_angle=angles
        ###############################

        ###############################线速度大小，z角速度大小
        lin_v=self.robot.data.root_state_w[:, 7:9].clone()
        lin_v_abs=torch.norm(lin_v, p=2, dim=1,keepdim=True) #计算速度大小没有方向
        agle_z_v=self.robot.data.root_state_w[:, 12:13].clone()
        ###############################

        ############################ 作为"robot-state": obs,
        obs = torch.cat(
            (
                angles,
                distances,
                self.target_min_distance,
                lin_v_abs,
                agle_z_v,
                #self.actions,       #这里的self.actions其实就是实际意义上的last_actions
            ),
            dim=-1,
        )
        ############################

        self.last_action=self.actions.clone()       #这里把last_actions赋值为self.actions ，last_actions用于计算平滑奖励

        #############################作为返回值
        observations = {
            "policy": {
                "camera": combined_tensor.permute(0, 3, 1, 2),
                "robot-state": obs,
            }
        }
        #############################作为专家经验数据
        my_observations={

                "camera": combined_tensor.permute(0, 3, 1, 2)[0:1,...],
                "robot-state": obs[0:1,...],
        }
        self.expert_obs=my_observations
        #############################


        return observations
    def optimized_distance_calculation(self):
        """优化版本的距离计算：考虑障碍物半径，计算无人机到障碍物表面的真实距离"""
        # 无人机位置 (4, 2)
        robot_pos = self.robot.data.root_state_w[:, 0:2].clone()
        
        # 获取移动障碍物位置和半径
        move_obs_positions = []
        move_obs_radii = []
        
        for i in range(8):
            # 获取移动障碍物位置 (1, 2)
            obs_pos = getattr(self, f"Move_Obstacle_{i}").data.root_state_w[:, 0:2].clone()
            move_obs_positions.append(obs_pos)
            
            # 获取移动障碍物半径（标量）
            cfg = getattr(self, f"Move_Obstacle_{i}").cfg
            radius = cfg.spawn.radius if hasattr(cfg, 'spawn') and hasattr(cfg.spawn, 'radius') else 0.0
            move_obs_radii.append(radius)
        
        # 拼接移动障碍物位置 (8, 2)
        move_obs = torch.cat(move_obs_positions, dim=0)
        
        # 获取静态障碍物位置和半径
        static_obs_positions = []
        static_obs_radii = []
        
        for i in range(16):
            # 获取静态障碍物位置 (4, 2) - 每个障碍物有4个复制体
            obs_pos = self.scene[f"Obstacle_{i}"].data.root_state_w[:, 0:2].clone()
            static_obs_positions.append(obs_pos)
            
            # 获取静态障碍物半径（标量）
            cfg = self.scene[f"Obstacle_{i}"].cfg
            radius = cfg.spawn.radius if hasattr(cfg, 'spawn') and hasattr(cfg.spawn, 'radius') else 0.0
            # 每个半径对应4个障碍物副本
            static_obs_radii.extend([radius] * 4)  # 重复4次
        
        # 拼接静态障碍物位置 (64, 2)
        static_obs = torch.cat(static_obs_positions, dim=0)
        
        # 合并所有障碍物位置 (72, 2)
        all_obstacles = torch.cat([move_obs, static_obs], dim=0)
        
        # 合并所有障碍物半径 (72,)
        all_radii = torch.tensor(move_obs_radii + static_obs_radii, 
                                device=robot_pos.device, 
                                dtype=robot_pos.dtype)
        
        # 使用torch.cdist进行高效的距离计算（中心到中心的距离）
        center_distances = torch.cdist(robot_pos, all_obstacles, p=2)  # 形状: (4, 72)
        
        # 计算表面距离：中心距离减去障碍物半径[1,6](@ref)
        surface_distances = center_distances - all_radii.unsqueeze(0)  # 形状: (4, 72)
        
        # 确保距离不为负（处理无人机可能已进入障碍物内部的情况）
        surface_distances = torch.clamp(surface_distances, min=0)
        
        # 计算到所有障碍物的最小表面距离
        min_distances, _ = torch.min(surface_distances, dim=1, keepdim=True)  # 形状: (4, 1)
        
        return min_distances  
    def _get_rewards(self) -> torch.Tensor:#待优化
        collision=self.collision_flags
        ################################如果这一次的距离小于最小距离，表明离目标点更近一步，有奖励，否则没有奖励也没有惩罚。
        # condition_mask = self.target_min_distance > self.distances

        # self.rew_for_distance = torch.where(
        #     condition_mask,
        #     self.target_min_distance - self.distances,
        #     torch.tensor(0.0, device=self.target_min_distance.device)  # 确保张量在同一设备上
        # )

        # self.target_min_distance = torch.where(      #如果这一次距离小于最小距离，更新最小距离。
        #     condition_mask,
        #     self.distances,
        #     self.target_min_distance
        # )
        self.rew_for_distance=self.last_distances-self.distances
        ##############################

        ############################从四元数计算机头朝向
        quat = self.robot.data.root_state_w[:, 3:7]
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        w,x, y, z = torch.split(quat,dim=-1,split_size_or_sections=1)
        x1=w*w+x*x-y*y-z*z
        y1=2*(x*y+w*z)
        z1=2*(x*z-w*y)
        horizon_direction = torch.cat((x1, y1, torch.zeros_like(z1)), dim=-1)
        normlized_direction=horizon_direction / torch.norm(horizon_direction, dim=-1, keepdim=True)
        #############################

        ##############################飞机指向目标点的向量，得到距离和归一化角度
        pos=self.robot.data.root_pos_w.clone()  
        direction_vector = self.target_pos - pos  # 形状 (num_envs, 3)  计算方向向量
        direction_vector=torch.cat((direction_vector[:,0:2],torch.zeros_like(z1)),dim=-1)
        distances = torch.norm(direction_vector, p=2, dim=1,keepdim=True)   #计算距离
        direction_vector=direction_vector / torch.norm(direction_vector, dim=-1, keepdim=True)
        ################################

        ##############################计算机头朝向向量和飞机指向目标点的向量的角度可以用 atan2
        angles = signed_angle(normlized_direction[:,:-1], direction_vector[:,:-1]).unsqueeze(dim=1)
        ###############################

        total_reward = compute_rewards(
            self.rew_for_distance,
            self.last_angle,
            angles,
            collision,
            self.last_action,
            self.actions,
            self.arrived,
        )
        ##################################################################################打印实施奖励的代码
        #rew_angle = (torch.abs(last_angle) - torch.abs(angles)) * 0.01      #待优化
        # rew_distance = self.rew_for_distance * 0.1     
        # rew_arrival = self.arrived.unsqueeze(dim=-1) * 10
        # collision_penalty = collision.unsqueeze(dim=-1) * (-1)

        # ######################################平滑度惩罚
        # thresh_linear = 0.6  
        # thresh_angular = 0.6  
        # weight_smoothness = 0.1  
        # power = 2  

        # action_diff = self.actions - self.last_action
        # abs_diff = torch.abs(action_diff)

        # diff_linear = abs_diff[:, 0]  
        # diff_angular = abs_diff[:, 1] 

        # excess_linear = torch.clamp(diff_linear - thresh_linear, min=0.0)
        # excess_angular = torch.clamp(diff_angular - thresh_angular, min=0.0)

        # penalty_linear = -weight_smoothness * (excess_linear ** power)
        # penalty_angular = -weight_smoothness * (excess_angular ** power)
        # smoothness_penalty = penalty_linear + penalty_angular  # 合并惩罚
        # smoothness_penalty = smoothness_penalty.unsqueeze(1)
        # #######################################

        # my_total_reward = (
        #     rew_distance +
        #     #rew_angle+
        #     rew_arrival +
        #     collision_penalty +
        #     smoothness_penalty 
            
        # )        
        # print(f"rew_distance:{rew_distance}")
        #print(f"rew_arrival:{rew_arrival}")
        #print(f"collision_penalty:{collision_penalty}")
        #print(f"smoothness_penalty:{smoothness_penalty}")
        ###################################################################################################
        return total_reward
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:#这里的return张量是一维的，和别的不同

        ###############################超时检测
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        ###############################

        ###############################接触传感器 没有用到
        # contact_force = self.scene["contact_forces"].data.net_forces_w  # [num_envs, N, 3]
        # contact_magnitude = torch.norm(contact_force, dim=-1)  # [num_envs, N]
        # collision = torch.any(contact_magnitude > 0.01, dim=-1)  # 只要有任意值 > 0.01 就算碰撞
        ###############################

        ###############################与障碍物距离检测
        self.min_distances=self.optimized_distance_calculation()
        collision_flags = (self.min_distances <= 0.40).float().squeeze(1)
        self.collision_flags=collision_flags
        ###############################

        ##############################飞机指向目标点的向量，得到距离和归一化角度
        pos=self.robot.data.root_pos_w.clone()  
        direction_vector = self.target_pos - pos  # 形状 (num_envs, 3)  计算方向向量
        direction_vector=torch.cat((direction_vector[:,0:2],torch.zeros_like(direction_vector[:,0:1])),dim=-1)
        distances = torch.norm(direction_vector, p=2, dim=1,keepdim=True)   #计算距离
        direction_vector=direction_vector / torch.norm(direction_vector, dim=-1, keepdim=True)
        self.distances=distances
        ################################

        ###############################计算是否到达
        arrived = (distances <= 0.5).squeeze(dim=1)
        self.arrived=arrived
        ###############################

        ###############################每25次导航任务进行一次print
        self.success_rate_count[0]+=time_out.sum().item()
        self.success_rate_count[1]+=collision_flags.sum().item()
        self.success_rate_count[2]+=arrived.sum().item()

        current_sum = sum(self.success_rate_count)
        current_condition = (current_sum % 25 == 0)

        # 只有当条件从False变为True时才打印
        if current_condition and not self.last_condition_state:   
            print(self.success_rate_count)
            print(self.actions)
            # print(self.allrew_angle)
            # print(self.allrew_distance)
            # print(self.allrew_arrival)
            # print(self.allcollision_penalty)
            # print(self.allsmoothness_penalty)
            # print(self.allreward)
        self.last_condition_state = current_condition
        #####################################

        #####################################合并到达和碰撞都作为reset信号
        done = torch.logical_or(collision_flags, arrived)#collision_flags
        #####################################

        #####################################采集专家经验
        if collision_flags[0].item()==0 and time_out[0].item()==0:
            mock_state = self.expert_obs
            mock_action = self.expert_action            
            #self.temp_buffer.add_step(mock_state, mock_action) 
            if arrived[0].item()==1:
                pass
                #self.temp_buffer.process_flag(flag=1, expert_pool=self.expert_pool)
        else:
            self.temp_buffer.process_flag(flag=2, expert_pool=self.expert_pool)
        ######################################

        return  done,time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        ################################清空reset的深度图队列
        if 0 in env_ids:
            self.image_buffer1.clear_buffer()
        if 1 in env_ids:
            self.image_buffer2.clear_buffer() 
        if 2 in env_ids:
            self.image_buffer3.clear_buffer()               
        if 3 in env_ids:
            self.image_buffer4.clear_buffer()
        ################################

        ################################重置无人机的所有状态
        default_root_state = self.robot.data.default_root_state[env_ids]#robot
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)#robot
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        ################################

        ######################################随机初始化目标点
        len_env_ids=len(env_ids)
        x=torch.zeros(len_env_ids, device=self.device).uniform_(0, 0).unsqueeze(dim=1)
        y=torch.zeros(len_env_ids, device=self.device).uniform_(-25, 25).unsqueeze(dim=1)
        z=torch.zeros(len_env_ids, device=self.device).uniform_(2, 2).unsqueeze(dim=1)
        xyz=torch.cat((x,y,z),dim=-1)

        result=[]
        xyz_indes=0
        for i in range(0,self.cfg.scene.num_envs):
            if i in env_ids:
                result.append(xyz[xyz_indes:xyz_indes+1,:])
                xyz_indes+=1

            else:
                result.append(self.target_pos[i:i+1, :])

        self.target_pos=torch.cat(result,dim=0)
        ########################################

        ########################################把对应环境的障碍物reset
        for i in range(16):
            obstacle = self.scene[f"Obstacle_{i}"]

            default_obstacle_state = obstacle.data.default_root_state[env_ids]

            default_obstacle_state[:, 0] += torch.zeros(len(env_ids), device=self.device).uniform_(-0, 0)
            default_obstacle_state[:, 1] += torch.zeros(len(env_ids), device=self.device).uniform_(-0, 0)
            default_obstacle_state[:, 0:3] += self.scene.env_origins[env_ids]
            obstacle.write_root_pose_to_sim(default_obstacle_state[:, :7], env_ids)
            obstacle.write_root_velocity_to_sim(default_obstacle_state[:, 7:], env_ids)
        #########################################

        ############################从四元数计算机头朝向
        quat = self.robot.data.root_state_w[:, 3:7]
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        w,x, y, z = torch.split(quat,dim=-1,split_size_or_sections=1)
        x1=w*w+x*x-y*y-z*z
        y1=2*(x*y+w*z)
        z1=2*(x*z-w*y)
        horizon_direction = torch.cat((x1, y1, torch.zeros_like(z1)), dim=-1)
        normlized_direction=horizon_direction / torch.norm(horizon_direction, dim=-1, keepdim=True)#当前机头朝向向量
        #############################

        ##############################飞机指向目标点的向量，得到距离和归一化角度
        pos=self.robot.data.root_pos_w.clone()  
        direction_vector = self.target_pos - pos  # 形状 (num_envs, 3)  计算方向向量
        direction_vector=torch.cat((direction_vector[:,0:2],torch.zeros_like(z1)),dim=-1)
        distances = torch.norm(direction_vector, p=2, dim=1,keepdim=True)   #计算距离
        direction_vector=direction_vector / torch.norm(direction_vector, dim=-1, keepdim=True)
        ################################

        ################################
        self.distances[env_ids]=distances[env_ids]
        ################################

        ##############################计算机头朝向向量和飞机指向目标点的向量的角度可以用 atan2
        angles = signed_angle(normlized_direction[:,:-1], direction_vector[:,:-1]).unsqueeze(dim=1)
        self.angles[env_ids]=angles[env_ids]
        ###############################

        self.target_min_distance[env_ids] = self.distances[env_ids]
        self.actions[env_ids] = torch.zeros_like(self.actions[env_ids])
#########################################################可视化
        for rest_visual in env_ids:
            if rest_visual==0:
                self.visual_list0=[]
                self.visual_list0.append(self.target_pos[0:1,:])
                self.visual_list4=[]
                self.visual_list4.append(self.target_pos_rot[0:1,:])
            elif rest_visual==1:
                self.visual_list1=[]
                self.visual_list1.append(self.target_pos[1:2,:])
                self.visual_list5=[]
                self.visual_list5.append(self.target_pos_rot[1:2,:])
            elif rest_visual==2:
                self.visual_list2=[]
                self.visual_list2.append(self.target_pos[2:3,:])
                self.visual_list6=[]
                self.visual_list6.append(self.target_pos_rot[2:3,:])
            else:
                self.visual_list3=[]
                self.visual_list3.append(self.target_pos[3:4,:])
                self.visual_list7=[]
                self.visual_list7.append(self.target_pos_rot[3:4,:])
            
#######################################################


       

def signed_angle(u, v):
    # 1. 归一化输入向量（防止点积累积误差）
    u_norm = u / torch.norm(u, dim=-1, keepdim=True).clamp(min=1e-6)  # 避免除以零
    v_norm = v / torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)
    
    dot_product = (u_norm * v_norm).sum(dim=-1)
    dot_product_clamped = torch.clamp(dot_product, min=-1.0, max=1.0)  # 关键！

    cross_product = u_norm[..., 0] * v_norm[..., 1] - u_norm[..., 1] * v_norm[..., 0]
    
    angle_rad = torch.sign(cross_product) * torch.acos(dot_product_clamped)
    return angle_rad


#@torch.jit.script
def compute_rewards(
    rew_for_distance: torch.Tensor,
    last_angle: torch.Tensor,
    angles: torch.Tensor,
    collision,
    last_action,
    now_action,
    arrived
):

    rew_angle = (torch.abs(last_angle) - torch.abs(angles)) * 0.1      #待优化
    rew_distance = rew_for_distance * 1    
    rew_arrival = arrived.unsqueeze(dim=-1) * 100
    collision_penalty = collision.unsqueeze(dim=-1) * (-10)
    #print(rew_angle)
    ######################################平滑度惩罚
    thresh_linear = 0.6  
    thresh_angular = 0.6  
    weight_smoothness = 0.1  
    power = 2  

    action_diff = now_action - last_action
    abs_diff = torch.abs(action_diff)

    diff_linear = abs_diff[:, 0]  
    diff_angular = abs_diff[:, 1] 

    excess_linear = torch.clamp(diff_linear - thresh_linear, min=0.0)
    excess_angular = torch.clamp(diff_angular - thresh_angular, min=0.0)

    penalty_linear = -weight_smoothness * (excess_linear ** power)
    penalty_angular = -weight_smoothness * (excess_angular ** power)
    smoothness_penalty = penalty_linear + penalty_angular  # 合并惩罚
    smoothness_penalty = smoothness_penalty.unsqueeze(1)
    #######################################

    total_reward = (
        rew_distance +
        rew_angle+
        rew_arrival +
        collision_penalty 
        #smoothness_penalty 
    )

    return total_reward.flatten()



def quat_to_euler_zyx(q):
    """
    将四元数 (w, x, y, z) 转换为 ZYX 顺序的欧拉角 (roll, pitch, yaw)，单位弧度。
    欧拉角顺序: 先绕 Z 轴旋转 (yaw), 再绕 Y 轴旋转 (pitch), 最后绕 X 轴旋转 (roll)。
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # 计算滚转 (roll, 绕 X 轴)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # 计算俯仰 (pitch, 绕 Y 轴)
    sinp = 2 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)  # 防止数值误差导致超出 [-1,1] 范围
    pitch = torch.asin(sinp)
    
    # 计算偏航 (yaw, 绕 Z 轴)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw
def euler_to_quat_zyx(roll, pitch, yaw):
    """
    将 ZYX 顺序的欧拉角 (roll, pitch, yaw，单位弧度) 转换为四元数 (w, x, y, z)。
    """
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    quat = torch.stack([w, x, y, z], dim=-1)
    return quat


class DepthImageBuffer:
    def __init__(self, buffer_size=49):
        """
        初始化深度图缓冲区
        Args:
            buffer_size: 缓冲区大小，默认为49
        """
        self.buffer_size = buffer_size
        # 使用deque创建固定大小的队列，当队列满时自动移除最旧的数据[6](@ref)
        self.image_queue = collections.deque(maxlen=buffer_size)
    
    def update_buffer(self, depth_norm):
        """
        更新缓冲区并生成目标张量（全程在GPU上操作）
        Args:
            depth_norm: 形状为(n,16,16,1)的深度图张量（GPU张量）
        Returns:
            combined_tensor: 形状为(n,16,16,3)的PyTorch张量（在GPU上）
        """
        # 将新数据加入队列（确保是GPU张量）
        self.image_queue.append(depth_norm.clone())  # 使用clone()避免引用问题
        
        # 检查队列是否已满（达到buffer_size张图片）
        if len(self.image_queue) < self.buffer_size:
            # 队列未满，三张图片全用最新的
            latest_img = self.image_queue[-1]  # 获取最新的图片（注意这里应该是-1而不是0）
            
            # 使用torch.cat将同一张图片复制三份，形成(n,16,16,3)的张量
            combined_tensor = torch.cat([latest_img, latest_img, latest_img], dim=-1)
            

            return combined_tensor
        else:
            # 队列已满，按原逻辑提取三张不同的图片
            recent_img = self.image_queue[-1]    # 最近一次（最新的一张）
            tenth_img = self.image_queue[24]     # 第25张
            twentieth_img = self.image_queue[0]  # 最旧的一张（索引0）
            
            # 使用torch.cat沿着最后一个维度合并三张图片
            combined_tensor = torch.cat([recent_img, tenth_img, twentieth_img], dim=-1)
            
            return combined_tensor
    
    def clear_buffer(self):
        """
        清空缓冲区，移除所有存储的深度图[6,8](@ref)
        """
        self.image_queue.clear()

    
    def get_buffer_status(self):
        """获取缓冲区状态"""
        return f"当前缓冲区大小: {len(self.image_queue)}/{self.buffer_size}"
    
    def is_buffer_full(self):
        """
        检查缓冲区是否已满
        Returns:
            bool: 缓冲区已满返回True，否则返回False
        """
        return len(self.image_queue) >= self.buffer_size
    
    def get_current_size(self):
        """
        获取当前缓冲区中的元素数量
        Returns:
            int: 当前缓冲区中的图片数量
        """
        return len(self.image_queue)


class ExpertExperiencePool:
    """专家经验池：支持字典状态和GPU张量，存储专家状态-动作对"""
    def __init__(self, state_shapes: Dict, action_dim: int, max_size: int = int(1e6), 
                 save_threshold: int = 1000, save_path: str = "expert_experience.npz",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        # 存储设备
        self.device = device
        
        # 初始化存储字典（状态）和数组（动作）
        self.state_dict = {}
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        
        # 为每个状态键创建存储数组
        for key, shape in state_shapes.items():
            # 形状从 [1, ...] 改为 [max_size, ...]
            state_shape = (max_size,) + shape[1:]  # 去掉batch维度
            self.state_dict[key] = np.zeros(state_shape, dtype=np.float32)
        
        self.max_size = max_size
        self.current_size = 0
        self.ptr = 0
        
        # 持久化相关参数
        self.save_threshold = save_threshold
        self.save_path = save_path
        self.state_shapes = state_shapes  # 保存形状信息用于加载验证

    def add(self, states: Dict[str, np.ndarray], actions: np.ndarray) -> None:
        """批量添加专家状态-动作对，支持字典状态"""
        batch_size = len(actions)
        
        # 处理循环存储
        if self.ptr + batch_size <= self.max_size:
            # 存储每个状态键值
            for key in states.keys():
                # 确保形状匹配
                target_shape = self.state_dict[key][self.ptr:self.ptr+batch_size].shape
                source_data = states[key]
                
                # 如果形状不匹配，尝试自动调整
                if source_data.shape != target_shape:
                    # 尝试squeeze掉多余的维度[4](@ref)
                    if len(source_data.shape) > len(target_shape):
                        source_data = np.squeeze(source_data, axis=1)  # 去掉第1维（batch维度）
                    
                    # 如果仍然不匹配，调整形状
                    if source_data.shape != target_shape:
                        try:
                            source_data = source_data.reshape(target_shape)
                        except ValueError as e:
                            print(f"形状调整失败: {source_data.shape} -> {target_shape}")
                            raise e
                
                self.state_dict[key][self.ptr:self.ptr+batch_size] = source_data
                
            self.action[self.ptr:self.ptr+batch_size] = actions
            self.ptr += batch_size
        else:
            # 分两部分存储
            part1_size = self.max_size - self.ptr
            for key in states.keys():
                # 第一部分
                part1_data = states[key][:part1_size]
                target_shape1 = self.state_dict[key][self.ptr:].shape
                if part1_data.shape != target_shape1:
                    if len(part1_data.shape) > len(target_shape1):
                        part1_data = np.squeeze(part1_data, axis=1)
                    if part1_data.shape != target_shape1:
                        part1_data = part1_data.reshape(target_shape1)
                self.state_dict[key][self.ptr:] = part1_data
                
                # 第二部分
                part2_data = states[key][part1_size:]
                target_shape2 = self.state_dict[key][:batch_size-part1_size].shape
                if part2_data.shape != target_shape2:
                    if len(part2_data.shape) > len(target_shape2):
                        part2_data = np.squeeze(part2_data, axis=1)
                    if part2_data.shape != target_shape2:
                        part2_data = part2_data.reshape(target_shape2)
                self.state_dict[key][:batch_size-part1_size] = part2_data
                
            self.action[self.ptr:] = actions[:part1_size]
            self.action[:batch_size-part1_size] = actions[part1_size:]
            self.ptr = batch_size - part1_size
        
        self.current_size = min(self.current_size + batch_size, self.max_size)
        
        # 阈值触发保存
        if self.current_size >= self.save_threshold:
            self.save_to_file()
            print(f"经验池数量达到阈值{self.save_threshold}，已保存到{self.save_path}")

    def save_to_file(self) -> None:
        """保存到NPZ文件，包含所有状态字典和动作"""
        save_data = {}
        
        # 保存每个状态键
        for key, state_array in self.state_dict.items():
            save_data[f"state_{key}"] = state_array[:self.current_size]
        
        # 保存动作和元数据
        save_data["action"] = self.action[:self.current_size]
        save_data["current_size"] = self.current_size
        save_data["state_shapes"] = self.state_shapes  # 保存形状信息
        
        np.savez_compressed(self.save_path, **save_data)

    def load_from_file(self) -> bool:
        """从NPZ文件加载，自动转换为GPU张量"""
        if not os.path.exists(self.save_path):
            print(f"文件{self.save_path}不存在，加载失败")
            return False
        
        data = np.load(self.save_path, allow_pickle=True)
        loaded_size = data["current_size"].item()
        
        # 检查数据量
        if loaded_size > self.max_size:
            print(f"加载数据量{loaded_size}超过经验池最大容量{self.max_size}，截断加载")
            loaded_size = self.max_size
        
        # 重新初始化状态字典
        self.state_dict = {}
        state_shapes = data["state_shapes"].item() if "state_shapes" in data else self.state_shapes
        
        for key, shape in state_shapes.items():
            state_key = f"state_{key}"
            if state_key in data:
                loaded_state = data[state_key][:loaded_size]
                # 初始化存储数组
                state_shape = (self.max_size,) + shape[1:]  # 去掉batch维度
                self.state_dict[key] = np.zeros(state_shape, dtype=np.float32)
                self.state_dict[key][:loaded_size] = loaded_state
        
        # 加载动作
        self.action[:loaded_size] = data["action"][:loaded_size]
        self.current_size = loaded_size
        self.ptr = loaded_size % self.max_size
        
        print(f"从{self.save_path}加载成功，当前存储数量：{self.current_size}")
        return True

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """随机采样一批状态-动作对，返回GPU张量"""
        if self.current_size < batch_size:
            raise ValueError(f"经验池当前数量{self.current_size}小于采样批次{batch_size}")
        
        indices = np.random.randint(0, self.current_size, size=batch_size)
        
        # 构建状态字典（GPU张量）
        expert_state = {}
        for key, state_array in self.state_dict.items():
            # 转换为GPU张量 
            expert_state[key] = torch.from_numpy(state_array[indices]).float().to(self.device)
        
        # 构建动作张量（GPU张量）
        expert_action = torch.from_numpy(self.action[indices]).float().to(self.device)
        
        return expert_state, expert_action

    def get_gpu_tensors(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """获取整个经验池的GPU张量（谨慎使用，可能占用大量显存）"""
        expert_state = {}
        for key, state_array in self.state_dict.items():
            expert_state[key] = torch.from_numpy(state_array[:self.current_size]).float().to(self.device)
        
        expert_action = torch.from_numpy(self.action[:self.current_size]).float().to(self.device)
        
        return expert_state, expert_action


class ExpertTempBuffer:
    """专家临时缓冲区：支持字典状态和GPU张量"""
    def __init__(self, state_shapes: Dict, action_dim: int):
        self.state_shapes = state_shapes
        self.action_dim = action_dim
        
        # 使用字典存储临时状态
        self.temp_states = {key: [] for key in state_shapes.keys()}
        self.temp_actions = []

    def add_step(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> None:
        """添加单步导航的状态-动作对，正确处理形状"""
        # 验证状态维度
        for key, expected_shape in self.state_shapes.items():
            if key not in state:
                raise ValueError(f"状态字典缺少键: {key}")
            
            # 获取实际形状（去掉batch维度进行比较）
            actual_shape = state[key].shape[1:]  # 去掉batch维度
            expected_shape_no_batch = expected_shape[1:]  # 去掉batch维度
            
            if actual_shape != tuple(expected_shape_no_batch):
                raise ValueError(f"状态{key}形状错误，需为{expected_shape_no_batch}，实际为{actual_shape}")
        
        # 验证动作维度
        if action.shape[1:] != (self.action_dim,):  # 去掉batch维度比较
            raise ValueError(f"动作维度错误，需为{self.action_dim}，实际为{action.shape[1]}")
        
        # 存储CPU上的numpy数组（去掉batch维度）
        for key, tensor in state.items():
            # 去掉batch维度，从 [1, ...] 变为 [...]
            numpy_array = tensor.cpu().numpy()
            if numpy_array.shape[0] == 1:  # 如果batch维度为1
                numpy_array = numpy_array[0]  # 去掉batch维度
            self.temp_states[key].append(numpy_array)
        
        # 处理动作（同样去掉batch维度）
        action_numpy = action.cpu().numpy()
        if action_numpy.shape[0] == 1:  # 如果batch维度为1
            action_numpy = action_numpy[0]  # 去掉batch维度
        self.temp_actions.append(action_numpy)

    def process_flag(self, flag: int, expert_pool: ExpertExperiencePool) -> None:
        """根据标志位处理临时缓冲区数据"""
        if flag not in [1, 2]:
            raise ValueError("标志位只能为1或2：1-提交数据并清空，2-直接清空")
        
        if flag == 1:
            if len(self.temp_actions) > 0:
                # 转换为numpy数组并构建状态字典
                batch_states = {}
                for key in self.temp_states.keys():
                    # 直接转换为numpy数组，形状为 [batch_size, ...]
                    batch_states[key] = np.array(self.temp_states[key], dtype=np.float32)
                
                batch_actions = np.array(self.temp_actions, dtype=np.float32)
                
                expert_pool.add(batch_states, batch_actions)
                print(f"临时缓冲区提交{len(self.temp_actions)}条数据到专家经验池，已清空")
            else:
                print("临时缓冲区无数据，提交操作跳过")
            self._clear()
        
        elif flag == 2:
            self._clear()
            print("临时缓冲区已按标志位2直接清空")

    def _clear(self) -> None:
        """清空临时缓冲区"""
        for key in self.temp_states.keys():
            self.temp_states[key] = []
        self.temp_actions = []

    def get_current_count(self) -> int:
        """获取当前缓冲区存储的步数"""
        return len(self.temp_actions)

