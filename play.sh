#!/bin/bash

# 加载conda环境
source ~/miniconda3/etc/profile.d/conda.sh

# 激活虚拟环境
conda activate env_isaaclab

python scripts/skrl/play.py --task Template-Tutorial-Direct-v0 --enable_cameras --num_envs 4 --checkpoint ~/tutorial/logs/skrl/quad/2025-12-08_12-37-20_ppo_torch/checkpoints/agent_4000.pt