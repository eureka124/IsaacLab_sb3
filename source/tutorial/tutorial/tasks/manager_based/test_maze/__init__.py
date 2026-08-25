"""Gym registration for the independent unseen test-maze environment."""

import gymnasium as gym

from ..tutorial import agents


gym.register(
    id="Template-Test-Maze-ManagerBased-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test_maze_env_cfg:TestMazeEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
