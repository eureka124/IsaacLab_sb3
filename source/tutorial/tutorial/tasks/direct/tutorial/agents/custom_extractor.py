import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # 初始化父类，features_dim必须 > 0，这里先填 1，最后计算完会覆盖
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        # 1. Camera Network (CNN)
        # Input: STATES["camera"] -> (Channel, Height, Width)
        camera_space = observation_space["camera"]
        n_input_channels = camera_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Compute CNN output dimension
        with torch.no_grad():
            # Create a dummy observation to calculate output shape
            # Add batch dimension [1, C, H, W]
            sample = torch.as_tensor(camera_space.sample()[None]).float()
            cnn_output_dim = self.cnn(sample).shape[1]

        # features_fc equivalent
        self.camera_fc = nn.Linear(cnn_output_dim, 192)
        # Note: SKRL config says activations: none for features_fc

        # 2. Robot State Network (MLP)
        # Input: STATES["robot-state"]
        robot_state_space = observation_space["robot-state"]
        state_dim = robot_state_space.shape[0]

        # robot_state_extractor equivalent
        self.robot_state_mlp = nn.Linear(state_dim, 192)

        # Calculate total features dim
        # Policy Net: concatenate([features_fc, robot_state_extractor])
        self._features_dim = 192

        # Note about Value Function Asymmetry:
        # SKRL's Value function uses [features_extractor, robot-state, critic-state].
        # In this implementation, we map both Policy and Value inputs to the same shared feature space:
        # [Camera(192), RobotState(192)].
        # This simplifies the implementation for SB3's standard PPO.

    def forward(self, observations) -> torch.Tensor:
        # 1. Process Camera
        # Ensure observation is correctly typed and shaped for CNN (B, C, H, W)
        obs_camera = observations["camera"]
        if obs_camera.dim() == 3:  # (C, H, W)
            obs_camera = obs_camera.unsqueeze(0)

        img_features = self.cnn(obs_camera)
        img_features = self.camera_fc(img_features)

        # 2. Process Robot State
        obs_robot = observations["robot-state"]
        if obs_robot.dim() == 1:
            obs_robot = obs_robot.unsqueeze(0)
        state_features = self.robot_state_mlp(obs_robot)

        # 3. Concatenate
        return torch.add(img_features, state_features)


class GodViewExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        # Calculate features dim
        robot_state_dim = observation_space["robot-state"].shape[0]
        critic_state_dim = observation_space["critic-state"].shape[0]
        self._features_dim = robot_state_dim + critic_state_dim

    def forward(self, observations) -> torch.Tensor:
        return torch.cat(
            [observations["robot-state"], observations["critic-state"]], dim=1
        )
