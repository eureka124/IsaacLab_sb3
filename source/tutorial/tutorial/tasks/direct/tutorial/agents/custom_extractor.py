import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # 初始化父类，features_dim必须 > 0，这里先填 1，最后计算完会覆盖
        super().__init__(observation_space, features_dim=1)

        # 1. Camera Network (CNN)
        # Input: STATES["camera"] -> (Channel, Height, Width)
        camera_space = observation_space["camera"]
        n_input_channels = camera_space.shape[0]
        self.camera_shape = camera_space.shape
        self.expected_camera_channels = n_input_channels
        self.expected_camera_hw = tuple(camera_space.shape[-2:])

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
        # SKRL's Value function uses [features_extractor, robot-state, critic-toa].
        # In this implementation, we map both Policy and Value inputs to the same shared feature space:
        # [Camera(192), RobotState(192)].
        # This simplifies the implementation for SB3's standard PPO.

    def forward(self, observations) -> torch.Tensor:
        # 1. Process Camera
        # Ensure observation is correctly typed and shaped for CNN (B, C, H, W)
        obs_camera = observations["camera"]
        if obs_camera.dim() == 3:
            # Could be CHW, HWC, or NHW (batched grayscale without channel dim).
            expected_h, expected_w = self.expected_camera_hw
            if obs_camera.shape[1:] == (expected_h, expected_w) and self.expected_camera_channels == 1:
                # NHW -> NCHW for grayscale camera.
                obs_camera = obs_camera.unsqueeze(1)
            elif obs_camera.shape[0] == self.expected_camera_channels:
                obs_camera = obs_camera.unsqueeze(0)
            elif obs_camera.shape[-1] == self.expected_camera_channels:
                obs_camera = obs_camera.permute(2, 0, 1).unsqueeze(0)
            else:
                raise RuntimeError("Unexpected camera tensor shape: " f"{tuple(obs_camera.shape)}; expected channel={self.expected_camera_channels}")
        elif obs_camera.dim() == 4:
            # Prefer identifying layout by spatial dimensions first.
            expected_h, expected_w = self.expected_camera_hw
            if obs_camera.shape[2:] == (expected_h, expected_w):
                # NCHW or C/N swapped.
                if obs_camera.shape[1] == self.expected_camera_channels:
                    pass
                elif obs_camera.shape[0] == self.expected_camera_channels:
                    obs_camera = obs_camera.permute(1, 0, 2, 3)
                else:
                    raise RuntimeError("Unexpected camera tensor shape: " f"{tuple(obs_camera.shape)}; expected NCHW with channel={self.expected_camera_channels}")
            elif obs_camera.shape[1:3] == (expected_h, expected_w):
                # NHWC -> NCHW.
                obs_camera = obs_camera.permute(0, 3, 1, 2)
            else:
                raise RuntimeError("Unexpected camera tensor shape: " f"{tuple(obs_camera.shape)}; expected spatial={self.expected_camera_hw}")
        else:
            raise RuntimeError(f"Unsupported camera tensor rank: {obs_camera.dim()} for shape {tuple(obs_camera.shape)}")

        if obs_camera.shape[1] != self.expected_camera_channels:
            raise RuntimeError("Camera tensor normalization failed: " f"got shape {tuple(obs_camera.shape)}, expected channel={self.expected_camera_channels}")

        img_features = self.cnn(obs_camera)
        img_features = self.camera_fc(img_features)

        # 2. Process Robot State
        obs_robot = observations["robot-state"]
        if obs_robot.dim() == 1:
            obs_robot = obs_robot.unsqueeze(0)
        state_features = self.robot_state_mlp(obs_robot)

        # 3. Concatenate
        return torch.add(img_features, state_features)


class ActorFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        # 初始化父类，features_dim必须 > 0，这里先填 1，最后计算完会覆盖
        super().__init__(observation_space, features_dim=1)

        # 1. Camera Network (CNN)
        # Input: STATES["camera"] -> (Channel, Height, Width)
        camera_space = observation_space["camera"]
        n_input_channels = camera_space.shape[0]
        self.camera_shape = camera_space.shape
        self.expected_camera_channels = n_input_channels
        self.expected_camera_hw = tuple(camera_space.shape[-2:])

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
        # SKRL's Value function uses [features_extractor, robot-state, critic-toa].
        # In this implementation, we map both Policy and Value inputs to the same shared feature space:
        # [Camera(192), RobotState(192)].
        # This simplifies the implementation for SB3's standard PPO.

    def forward(self, observations) -> torch.Tensor:
        # 1. Process Camera
        # Ensure observation is correctly typed and shaped for CNN (B, C, H, W)
        obs_camera = observations["camera"]
        if obs_camera.dim() == 3:
            # Could be CHW, HWC, or NHW (batched grayscale without channel dim).
            expected_h, expected_w = self.expected_camera_hw
            if obs_camera.shape[1:] == (expected_h, expected_w) and self.expected_camera_channels == 1:
                # NHW -> NCHW for grayscale camera.
                obs_camera = obs_camera.unsqueeze(1)
            elif obs_camera.shape[0] == self.expected_camera_channels:
                obs_camera = obs_camera.unsqueeze(0)
            elif obs_camera.shape[-1] == self.expected_camera_channels:
                obs_camera = obs_camera.permute(2, 0, 1).unsqueeze(0)
            else:
                raise RuntimeError("Unexpected camera tensor shape: " f"{tuple(obs_camera.shape)}; expected channel={self.expected_camera_channels}")
        elif obs_camera.dim() == 4:
            # Prefer identifying layout by spatial dimensions first.
            expected_h, expected_w = self.expected_camera_hw
            if obs_camera.shape[2:] == (expected_h, expected_w):
                # NCHW or C/N swapped.
                if obs_camera.shape[1] == self.expected_camera_channels:
                    pass
                elif obs_camera.shape[0] == self.expected_camera_channels:
                    obs_camera = obs_camera.permute(1, 0, 2, 3)
                else:
                    raise RuntimeError("Unexpected camera tensor shape: " f"{tuple(obs_camera.shape)}; expected NCHW with channel={self.expected_camera_channels}")
            elif obs_camera.shape[1:3] == (expected_h, expected_w):
                # NHWC -> NCHW.
                obs_camera = obs_camera.permute(0, 3, 1, 2)
            else:
                raise RuntimeError("Unexpected camera tensor shape: " f"{tuple(obs_camera.shape)}; expected spatial={self.expected_camera_hw}")
        else:
            raise RuntimeError(f"Unsupported camera tensor rank: {obs_camera.dim()} for shape {tuple(obs_camera.shape)}")

        if obs_camera.shape[1] != self.expected_camera_channels:
            raise RuntimeError("Camera tensor normalization failed: " f"got shape {tuple(obs_camera.shape)}, expected channel={self.expected_camera_channels}")

        img_features = self.cnn(obs_camera)
        img_features = self.camera_fc(img_features)

        # 2. Process Robot State
        obs_robot = observations["robot-state"]
        if obs_robot.dim() == 1:
            obs_robot = obs_robot.unsqueeze(0)
        state_features = self.robot_state_mlp(obs_robot)

        # 3. add
        return torch.add(img_features, state_features)


class CriticFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        camera_space = observation_space["camera"]
        n_input_channels = camera_space.shape[0]
        self.camera_shape = camera_space.shape
        self.expected_camera_channels = n_input_channels
        self.expected_camera_hw = tuple(camera_space.shape[-2:])

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.toa_cnn = nn.Sequential(
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
            sample_toa = torch.as_tensor(observation_space["critic-toa"].sample()[None]).float()
            toa_cnn_output_dim = self.toa_cnn(sample_toa).shape[1]

        # features_fc equivalent
        self.camera_fc = nn.Linear(cnn_output_dim, 192)

        # 1. Robot State Network (MLP)
        robot_state_space = observation_space["robot-state"]
        state_dim = robot_state_space.shape[0]
        self.robot_state_mlp = nn.Linear(state_dim, 192)

        # 2. Critic State Network (MLP)
        critic_state_space = observation_space["critic-toa"]

        self.toa_mlp = nn.Linear(toa_cnn_output_dim, 192)

        # Total features dim is the sum of both MLP outputs
        self._features_dim = 192

    def forward(self, observations) -> torch.Tensor:
        # 1. Process Camera
        # Ensure observation is correctly typed and shaped for CNN (B, C, H, W)
        obs_camera = observations["camera"]
        if obs_camera.dim() == 3:
            # Could be CHW, HWC, or NHW (batched grayscale without channel dim).
            expected_h, expected_w = self.expected_camera_hw
            if obs_camera.shape[1:] == (expected_h, expected_w) and self.expected_camera_channels == 1:
                # NHW -> NCHW for grayscale camera.
                obs_camera = obs_camera.unsqueeze(1)
            elif obs_camera.shape[0] == self.expected_camera_channels:
                obs_camera = obs_camera.unsqueeze(0)
            elif obs_camera.shape[-1] == self.expected_camera_channels:
                obs_camera = obs_camera.permute(2, 0, 1).unsqueeze(0)
            else:
                raise RuntimeError("Unexpected camera tensor shape: " f"{tuple(obs_camera.shape)}; expected channel={self.expected_camera_channels}")
        elif obs_camera.dim() == 4:
            # Prefer identifying layout by spatial dimensions first.
            expected_h, expected_w = self.expected_camera_hw
            if obs_camera.shape[2:] == (expected_h, expected_w):
                # NCHW or C/N swapped.
                if obs_camera.shape[1] == self.expected_camera_channels:
                    pass
                elif obs_camera.shape[0] == self.expected_camera_channels:
                    obs_camera = obs_camera.permute(1, 0, 2, 3)
                else:
                    raise RuntimeError("Unexpected camera tensor shape: " f"{tuple(obs_camera.shape)}; expected NCHW with channel={self.expected_camera_channels}")
            elif obs_camera.shape[1:3] == (expected_h, expected_w):
                # NHWC -> NCHW.
                obs_camera = obs_camera.permute(0, 3, 1, 2)
            else:
                raise RuntimeError("Unexpected camera tensor shape: " f"{tuple(obs_camera.shape)}; expected spatial={self.expected_camera_hw}")
        else:
            raise RuntimeError(f"Unsupported camera tensor rank: {obs_camera.dim()} for shape {tuple(obs_camera.shape)}")

        if obs_camera.shape[1] != self.expected_camera_channels:
            raise RuntimeError("Camera tensor normalization failed: " f"got shape {tuple(obs_camera.shape)}, expected channel={self.expected_camera_channels}")

        img_features = self.cnn(obs_camera)
        img_features = self.camera_fc(img_features)

        obs_robot = observations["robot-state"]
        if obs_robot.dim() == 1:
            obs_robot = obs_robot.unsqueeze(0)
        state_features = self.robot_state_mlp(obs_robot)

        obs_critic = observations["critic-toa"]
        if obs_critic.dim() == 3:
            obs_critic = obs_critic.unsqueeze(0)
        critic_features = self.toa_cnn(obs_critic)
        critic_features = self.toa_mlp(critic_features)

        # Sum image, robot-state and critic features element-wise
        return img_features + state_features + critic_features


class GodViewExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        # Calculate features dim
        robot_state_dim = observation_space["robot-state"].shape[0]
        critic_state_dim = int(torch.tensor(observation_space["critic-toa"].shape).prod().item())
        self._features_dim = robot_state_dim + critic_state_dim

    def forward(self, observations) -> torch.Tensor:
        critic = observations["critic-toa"]
        if critic.dim() > 2:
            critic = critic.flatten(start_dim=1)
        return torch.cat([observations["robot-state"], critic], dim=1)
