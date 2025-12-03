#!/bin/bash

# 加载conda环境
source ~/miniconda3/etc/profile.d/conda.sh

# 激活虚拟环境
conda activate env_isaaclab

python scripts/skrl/play.py --task Template-Tutorial-Direct-v0 --enable_cameras --num_envs 1