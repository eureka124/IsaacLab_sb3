import gymnasium as gym

from . import agents


gym.register(
    id="Template-Tutorial-ManagerBased-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tutorial_env_cfg:TutorialManagerBasedEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
