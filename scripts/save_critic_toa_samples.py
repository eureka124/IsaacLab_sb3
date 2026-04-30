"""Generate and save robot-aligned critic TOA samples for inspection.

Run with: python scripts/save_critic_toa_samples.py

This script requires IsaacLab to be properly initialized via AppLauncher.
"""

import argparse
import os
import sys
from pathlib import Path

# Add source/tutorial directory to path for imports
source_tutorial_dir = Path(__file__).parent.parent / "source" / "tutorial"
if str(source_tutorial_dir) not in sys.path:
    sys.path.insert(0, str(source_tutorial_dir))

from isaaclab.app import AppLauncher

# Launch Isaac Sim first
parser = argparse.ArgumentParser(description="Save critic TOA samples.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

# initialize app
app_launcher = AppLauncher(args)
app_context = app_launcher.app

# proceed only if app is enabled
if app_launcher.app is None:
    sys.exit("Failed to launch Isaac Sim app.")

# now we can safely import IsaacLab modules
import torch
import matplotlib.pyplot as plt
from tutorial.tasks.direct.tutorial.tutorial_env import TutorialEnv
from tutorial.tasks.direct.tutorial.tutorial_env_cfg import TutorialEnvCfg


def main():
    # Create config with specified number of envs
    cfg = TutorialEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # Create environment
    env = TutorialEnv(cfg, render_mode=None)

    # Reset all environments
    env_ids = torch.arange(args.num_envs, device=env.device)
    env._reset_idx(env_ids)

    # Build critic crops
    crops = env._build_robot_aligned_critic_toa(crop_size=env.critic_toa_crop_size)
    # crops shape: (num_envs, 1, H, W)
    crops_np = crops.detach().cpu().numpy()

    out_dir = os.path.join(os.getcwd(), "outputs", "critic_toa_samples")
    os.makedirs(out_dir, exist_ok=True)

    # Save samples from first 4 envs
    for i in range(min(4, crops_np.shape[0])):
        img = crops_np[i, 0]
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap="viridis_r", origin="lower")
        plt.colorbar(label="Normalized TOA")
        plt.title(f"critic-toa-env-{i} (crop_size={env.critic_toa_crop_size})")
        plt.xlabel("X (robot frame)")
        plt.ylabel("Y (robot frame)")
        save_path = os.path.join(out_dir, f"critic_toa_env_{i}.png")
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    print(f"\nCritic TOA crop size: {env.critic_toa_crop_size}×{env.critic_toa_crop_size}")
    print(f"All samples saved to: {out_dir}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
