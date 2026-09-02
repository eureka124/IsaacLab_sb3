"""Gym registrations for the six training-maze environments."""

import gymnasium as gym

from ..tutorial import agents


gym.register(
    id="Training-Mazes-v0",
    entry_point=f"{__name__}.training_mazes_env:TrainingMazesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.training_mazes_env_cfg:TrainingMazesEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
