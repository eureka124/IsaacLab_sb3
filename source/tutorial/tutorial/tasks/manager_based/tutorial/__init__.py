import gymnasium as gym

from . import agents

gym.register(
    id="Tutorial-Manager-v0",
    entry_point=f"{__name__}.tutorial_env:TutorialManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tutorial_env_cfg:TutorialManagerBasedEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
