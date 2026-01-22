import os
import glob
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
        max_keep: int = 5,
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

    def __init__(self, log_freq: int = 100, verbose=0):
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
            self.logger.record("env/success_rate", success_rate)
            self.logger.record("env/collision_rate", collision_rate)
            self.logger.record("env/timeout_rate", timeout_rate)

            # Dump logs to ensure they are written immediately
            # self.logger.dump(step=self.num_timesteps)
            # Note: PPO usually dumps logs at its own interval, but record puts it in the buffer.

            # Reset accumulators
            self.success_count = 0.0
            self.collision_count = 0.0
            self.timeout_count = 0.0
            self.total_episodes = 0

        return True
