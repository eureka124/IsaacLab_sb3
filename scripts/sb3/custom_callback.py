import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from episode_metrics import EpisodeOutcomeWindow


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

    It accumulates per-environment terminal flags and logs the rates after at
    least ``log_freq`` completed episodes. Every done episode contributes to the
    denominator, including boundary failures that are not one of the three
    requested outcome categories.
    """

    def __init__(self, log_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.outcomes = EpisodeOutcomeWindow(minimum_episodes=log_freq)

    @staticmethod
    def _as_bool(value) -> bool:
        """Convert scalar tensors, NumPy scalars, and Python values to bool."""

        if hasattr(value, "item"):
            value = value.item()
        return bool(value)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones")

        for index, info in enumerate(infos):
            done = (
                self._as_bool(dones[index])
                if dones is not None
                else info.get("episode") is not None
            )
            if not done:
                continue

            self.outcomes.add(
                success=self._as_bool(info.get("success", info.get("is_success", False))),
                collision=self._as_bool(info.get("collided", False)),
                timeout=self._as_bool(
                    info.get("time_out", info.get("TimeLimit.truncated", False))
                ),
            )

        if self.outcomes.ready:
            rates = self.outcomes.consume()
            self.logger.record("Metrics/Success_Rate", rates["success_rate"])
            self.logger.record("Metrics/Collision_Rate", rates["collision_rate"])
            self.logger.record("Metrics/Timeout_Rate", rates["timeout_rate"])
            self.logger.record("Metrics/Episodes", rates["episodes"])

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
            # SB3 VecEnv unwrap logic
            if hasattr(curr_env, "unwrapped"):
                raw_env = curr_env.unwrapped
            elif hasattr(curr_env, "envs"):
                raw_env = curr_env.envs[0]
            else:
                raw_env = curr_env

            # Recursively unwrap to find the one with toa_maps
            # This handles SB3 wrappers, Gym wrappers, and VecEnv wrappers
            max_depth = 10
            while not hasattr(raw_env, "toa_maps") and max_depth > 0:
                if hasattr(raw_env, "env"):
                    raw_env = raw_env.env
                elif hasattr(raw_env, "venv"):
                    raw_env = raw_env.venv
                    if hasattr(raw_env, "envs"):
                        raw_env = raw_env.envs[0]
                elif hasattr(raw_env, "unwrapped"):
                    raw_env = raw_env.unwrapped
                else:
                    break
                max_depth -= 1

            # Check if it has toa_maps
            if not hasattr(raw_env, "toa_maps"):
                if self.verbose > 0:
                    print(
                        f"[ToaVisualizationCallback] Found env type {type(raw_env)} but it does not have 'toa_maps'."
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
