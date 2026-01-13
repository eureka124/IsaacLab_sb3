# 实验记录

# MDP Process Summary

根据代码 `tutorial_env.py` 和 `tutorial_env_cfg.py` 的分析，本环境的 MDP 过程定义如下：

## 1. 状态空间 (State Space / Critic Observation)
Critic 网络使用的是特权信息 (Privileged Information)，即 `critic-state`。
- **维度**: 15
- **内容**: 主要是最近的 **5个障碍物** 的几何信息。
- **组成**: 对于每个障碍物 (共5个)，包含以下 3 个值：
  1. `relative_x`: 障碍物相对于机器人的位置 X 坐标（机体坐标系）。
  2. `relative_y`: 障碍物相对于机器人的位置 Y 坐标（机体坐标系）。
  3. `radius`: 障碍物的半径。
  - 数据来源: `tutorial_env.py` 中通过计算障碍物与机器人的距离，选取最近的 5 个，并进行坐标变换得到。如果少于 5 个障碍物，则用 0 填充。

## 2. 观察空间 (Observation Space / Policy Observation)
Policy 网络使用的是机器人自身的传感器数据。分为两部分：本体感知 (`robot-state`) 和 视觉输入 (`camera`)。

### Robot State (本体感知)
- **维度**: 6
- **内容**:
  1. `relative_target_x`: 目标点相对于机器人的位置 X 坐标（机体坐标系）。
  2. `relative_target_y`: 目标点相对于机器人的位置 Y 坐标（机体坐标系）。
  3. `last_action_vx`: 上一时刻执行的动作 vx。
  4. `last_action_vy`: 上一时刻执行的动作 vy。
  5. `velocity_x`: 当前机器人的线速度 X 分量（机体坐标系）。
  6. `velocity_y`: 当前机器人的线速度 Y 分量（机体坐标系）。

### Camera (视觉输入)
- **维度**: `[3, 16, 12]` (通道数, 高, 宽)
- **内容**: 归一化的深度图像堆叠。
- **采样策略**: 维护一个长度为 49 的图像缓冲区 (`DepthImageBuffer`)，从中采样 **3帧** 图像，分别是：
  - `latest`: 缓冲区中最新的帧。
  - `middle`: 缓冲区中间位置的帧。
  - `oldest`: 缓冲区中最旧的帧。
  - 这种采样方式为网络提供了时间维度的上下文信息。

## 3. 动作空间 (Action Space)
- **类型**: 连续动作空间 (Continuous)
- **维度**: 2
- **内容**: 机器人在机体坐标系下的速度指令。
  - **Action[0] (vx)**: 前向速度。
    - 范围: `[-0.1, 2.0]` m/s
  - **Action[1] (vy)**: 侧向速度。
    - 范围: `[-0.5, 0.5]` m/s
- **执行方式**: 动作被解释为机体坐标系下的目标速度，结合高度控制 (固定高度 2.0m) 传入 `LeePositionController` 计算底层电机推力和力矩。

