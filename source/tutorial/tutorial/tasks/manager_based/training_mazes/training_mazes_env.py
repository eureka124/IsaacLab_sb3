"""Manager-based training-maze environment with per-episode outcome metadata."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv


class TrainingMazesEnv(ManagerBasedRLEnv):
    """Expose each cloned environment's terminal reason to RL wrappers.

    Isaac Lab's termination manager retains the term values computed immediately
    before an automatic reset. Copying them into ``extras`` here lets SB3 attach
    the correct outcome to each terminal environment's ``info`` dictionary.
    """

    def step(self, action):
        observations, rewards, terminated, truncated, extras = super().step(action)

        success = self.termination_manager.get_term("goal_reached").clone()
        collided = self.termination_manager.get_term("collision").clone()
        time_out = self.termination_manager.get_term("time_out").clone()

        # Do not mutate the environment's shared extras dictionary in place.
        extras = dict(extras)
        extras["success"] = success
        extras["is_success"] = success  # Stable-Baselines3 convention.
        extras["collided"] = collided
        extras["time_out"] = time_out
        return observations, rewards, terminated, truncated, extras
