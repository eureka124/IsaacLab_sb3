import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class CheckpointCallbackWithLimit(CheckpointCallback):
    """
    CheckpointCallback with a limit on the number of checkpoints to keep.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
        max_keep: int = 20,
    ):
        super().__init__(
            save_freq,
            save_path,
            name_prefix,
            save_replay_buffer,
            save_vecnormalize,
            verbose,
        )
        self.max_keep = max_keep

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            self._remove_old_checkpoints()
        return result

    def _remove_old_checkpoints(self):
        # List all checkpoints matching the pattern
        search_pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
        checkpoints = glob.glob(search_pattern)

        # Sort checkpoints by step count (extracted from filename)
        def get_steps(filepath):
            filename = os.path.basename(filepath)
            # Assuming filename format is "{name_prefix}_{steps}_steps.zip"
            try:
                parts = filename.replace(".zip", "").split("_")
                steps = int(parts[-2])  # ..._X_steps
                return steps
            except (ValueError, IndexError):
                return -1

        checkpoints.sort(key=get_steps)

        # Remove oldest checkpoints if count exceeds max_keep
        if len(checkpoints) > self.max_keep:
            to_remove = checkpoints[: -self.max_keep]
            for cp in to_remove:
                if self.verbose >= 1:
                    print(f"Removing old checkpoint: {cp}")
                try:
                    os.remove(cp)
                    # Also try to remove vecnormalize file if it exists/expected
                    # Standard sb3 CheckpointCallback saves vecnormalize as well if requested
                    # format: {name_prefix}_vecnormalize_{steps}_steps.pkl
                    if self.save_vecnormalize:
                        vec_filename = cp.replace(
                            f"{self.name_prefix}_", f"{self.name_prefix}_vecnormalize_"
                        ).replace(".zip", ".pkl")
                        if os.path.exists(vec_filename):
                            os.remove(vec_filename)
                except OSError as e:
                    print(f"Error removing {cp}: {e}")


class IsaacLogCallback(BaseCallback):
    """
    A custom callback that logs Isaac Lab specific metrics (success, collision, timeout)
    to TensorBoard.

    It accumulates the boolean flags from the environment's `extras` and logs the
    rates every `log_freq` episodes.
    """

    def __init__(self, log_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Accumulators
        self.success_count = 0.0
        self.collision_count = 0.0
        self.timeout_count = 0.0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        # self.locals["infos"] contains the list of info dicts for the current step
        infos = self.locals["infos"]

        for info in infos:
            # Check if one of the termination flags is present and true
            # Note: In Sb3Env, these are booleans/floats in `extras`

            s = info.get("success", 0)
            c = info.get("collided", 0)
            t = info.get("time_out", 0)

            # Convert tensors to float/bool
            if hasattr(s, "item"):
                s = s.item()
            if hasattr(c, "item"):
                c = c.item()
            if hasattr(t, "item"):
                t = t.item()

            # Check if episode terminated (any of these flags is true)
            if s or c or t:
                self.total_episodes += 1
                self.success_count += float(s)
                self.collision_count += float(c)
                self.timeout_count += float(t)

        # Log metrics if we collected enough episodes
        if self.total_episodes >= self.log_freq:
            # Normalize to get rate [0, 1]
            success_rate = self.success_count / self.total_episodes
            collision_rate = self.collision_count / self.total_episodes
            timeout_rate = self.timeout_count / self.total_episodes

            # Record to TensorBoard
            self.logger.record("Metrics/Success_Rate", success_rate)
            self.logger.record("Metrics/Collision_Rate", collision_rate)
            self.logger.record("Metrics/Timeout_Rate", timeout_rate)

            # Dump logs to ensure they are written immediately
            # self.logger.dump(step=self.num_timesteps)
            # Note: PPO usually dumps logs at its own interval, but record puts it in the buffer.

            # Reset accumulators
            self.success_count = 0.0
            self.collision_count = 0.0
            self.timeout_count = 0.0
            self.total_episodes = 0

        return True


class ToaVisualizationCallback(BaseCallback):
    """
    Custom callback to visualize the TOA (Time of Arrival) map of the first environment
     and push it to WandB.
    """

    def __init__(self, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self.visualize_toa()
        return True

    def visualize_toa(self):
        # Access the environment from the model
        env = self.training_env
        if env is None:
            return

        # Unwrap to get the TutorialEnv instance
        # SB3 environments are often wrapped in VecEnv layers
        try:
            # Try to get the original Isaac Lab env
            # The structure is usually VecNormalize -> Sb3VecEnvWrapper -> gym.Env
            curr_env = env
            while hasattr(curr_env, "venv"):
                curr_env = curr_env.venv
            if hasattr(curr_env, "unwrapped"):
                raw_env = curr_env.unwrapped
            else:
                raw_env = curr_env

            # Check if it has toa_maps
            if not hasattr(raw_env, "toa_maps"):
                if self.verbose > 0:
                    print(
                        "[ToaVisualizationCallback] Environment does not have 'toa_maps' attribute."
                    )
                return

            # Get the TOA map for the first environment (env_id=0)
            toa_map_tensor = raw_env.toa_maps[0]
            toa_map = toa_map_tensor.detach().cpu().numpy()

            # Optional: Get current robot and goal position for overlay
            robot_pos = raw_env.robot.data.root_pos_w[0, :2].detach().cpu().numpy()
            goal_pos = raw_env.target_pos[0, :2].detach().cpu().numpy()
            origin = raw_env.scene.env_origins[0, :2].detach().cpu().numpy()

            # Transform world coordinates to grid coordinates for plotting
            # Map extent in local coordinates
            x_min, x_max = raw_env.toa_x_min, raw_env.toa_x_max
            y_min, y_max = raw_env.toa_y_min, raw_env.toa_y_max

            # Local positions
            robot_local = robot_pos - origin
            goal_local = goal_pos - origin

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(
                toa_map.T,
                origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                cmap="viridis",
            )
            fig.colorbar(im, ax=ax, label="Time of Arrival")

            # Overlay robot and goal
            ax.scatter(
                robot_local[0],
                robot_local[1],
                c="red",
                marker="o",
                s=100,
                label="Robot",
            )
            ax.scatter(
                goal_local[0], goal_local[1], c="white", marker="*", s=200, label="Goal"
            )

            ax.set_title(f"TOA Map - Step {self.num_timesteps}")
            ax.set_xlabel("Local X (m)")
            ax.set_ylabel("Local Y (m)")
            ax.legend()

            # Log to WandB if active
            if wandb.run is not None:
                wandb.log(
                    {"Visualization/TOA_Map": wandb.Image(fig)}, step=self.num_timesteps
                )

            plt.close(fig)

        except Exception as e:
            if self.verbose > 0:
                print(f"[ToaVisualizationCallback] Error during visualization: {e}")
