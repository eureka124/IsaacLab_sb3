# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Tutorial-Direct-v0",
    entry_point=f"{__name__}.tutorial_env:TutorialEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tutorial_env_cfg:TutorialEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:god_view_skrl_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
