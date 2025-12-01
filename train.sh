#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

# 激活虚拟环境
conda activate env_isaaclab

# 然后执行你的命令
python scripts/skrl/train.py --task Template-Tutorial-Direct-v0 --enable_cameras --headless --num_envs 80 

