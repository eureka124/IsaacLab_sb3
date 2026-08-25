"""Gym registrations for the six training-maze environments."""

import gymnasium as gym

from ..tutorial import agents


gym.register(
    id="Template-Training-Mazes-ManagerBased-1024-v0",
    entry_point=f"{__name__}.training_mazes_env:TrainingMazesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.training_mazes_env_cfg:TrainingMazes1024EnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Training-Mazes-ManagerBased-2048-v0",
    entry_point=f"{__name__}.training_mazes_env:TrainingMazesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.training_mazes_env_cfg:TrainingMazes2048EnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
