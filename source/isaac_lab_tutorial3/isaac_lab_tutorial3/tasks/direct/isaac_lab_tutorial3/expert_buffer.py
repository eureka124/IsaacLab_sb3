import numpy as np
import os
from typing import Tuple, Dict
import torch

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


# 使用示例
if __name__ == "__main__":
    # 定义状态形状字典（注意：这里定义的是带batch维度的形状）
    STATE_SHAPES = {
        'camera': (1, 3, 16, 16),  # [batch, channels, height, width]
        'robot-state': (1, 7)      # [batch, robot_state_dim]
    }
    ACTION_DIM = 2

    # 初始化专家经验池
    expert_pool = ExpertExperiencePool(
        state_shapes=STATE_SHAPES,
        action_dim=ACTION_DIM,
        max_size=10000,
        save_threshold=1000,
        save_path="drone_expert_data.npz",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 初始化临时缓冲区
    temp_buffer = ExpertTempBuffer(
        state_shapes=STATE_SHAPES,
        action_dim=ACTION_DIM
    )

    # 尝试加载已有数据
    expert_pool.load_from_file()

    # 模拟添加数据（注意：输入是带batch维度的张量）
    for _ in range(10):
        # 创建模拟状态字典（GPU张量，带batch维度）
        mock_state = {
            'camera': torch.randn(1, 3, 16, 16).cuda(),  # 形状 [1, 3, 16, 16]
            'robot-state': torch.randn(1, 7).cuda()       # 形状 [1, 7]
        }
        mock_action = torch.randn(1, 2).cuda()  # 形状 [1, 2]
        
        temp_buffer.add_step(mock_state, mock_action)

    # 提交数据到经验池
    temp_buffer.process_flag(flag=2, expert_pool=expert_pool)

    # 采样测试
    if expert_pool.current_size >= 5:
        expert_state, expert_action = expert_pool.sample(5)
        print(f"采样状态形状: camera: {expert_state['camera'].shape}, robot-state: {expert_state['robot-state'].shape}")
        print(f"采样动作形状: {expert_action.shape}")
        print(f"张量设备: {expert_action.device}")