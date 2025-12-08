#!/bin/bash
# 从远程服务器同步最新的输出和日志文件到本地
rsync -avz --progress jyyan:~/xuyang/tutorial/outputs ~/tutorial/
rsync -avz --progress jyyan:~/xuyang/tutorial/logs ~/tutorial/

# 加载conda环境
source ~/miniconda3/etc/profile.d/conda.sh

# 激活虚拟环境
conda activate env_isaaclab

python scripts/skrl/play.py --task Template-Tutorial-Direct-v0 --enable_cameras --num_envs 4 --checkpoint ~/tutorial/logs/skrl/quad/2025-12-08_13-21-32_ppo_torch/checkpoints/best_agent.pt 

