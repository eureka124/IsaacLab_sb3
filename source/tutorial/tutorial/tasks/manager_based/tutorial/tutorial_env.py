"""Manager-based environment with the same public contract as TutorialEnv."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv

from .alignment import TUTORIAL_ACTION_HIGH, TUTORIAL_ACTION_LOW


class TutorialManagerEnv(ManagerBasedRLEnv):
    """Expose the direct tutorial's action bounds and observation keys."""

    _POLICY_OBSERVATION_KEY_ALIASES = {
        "robot_state": "robot-state",
        "critic_toa": "critic-toa",
    }

    def _configure_gym_env_spaces(self) -> None:
        super()._configure_gym_env_spaces()

        # Physical commands: [body-x velocity, body-y velocity, yaw rate].
        self.single_action_space = gym.spaces.Box(
            low=np.asarray(TUTORIAL_ACTION_LOW, dtype=np.float32),
            high=np.asarray(TUTORIAL_ACTION_HIGH, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # Config-class fields cannot contain dashes, so rename terms only at the
        # public Gym boundary. This makes policy/checkpoint input keys match
        # TutorialEnv without weakening the manager configuration API.
        if isinstance(self.single_observation_space, gym.spaces.Dict):
            group_spaces = dict(self.single_observation_space.spaces)
            policy_space = group_spaces.get("policy")
            if isinstance(policy_space, gym.spaces.Dict):
                group_spaces["policy"] = gym.spaces.Dict(
                    {
                        self._POLICY_OBSERVATION_KEY_ALIASES.get(term_name, term_name): term_space
                        for term_name, term_space in policy_space.spaces.items()
                    }
                )
                self.single_observation_space = gym.spaces.Dict(group_spaces)
                self.observation_space = gym.vector.utils.batch_space(
                    self.single_observation_space, self.num_envs
                )

    def _align_observation_keys(self, observations):
        if not isinstance(observations, dict):
            return observations
        policy_observations = observations.get("policy")
        if not isinstance(policy_observations, dict):
            return observations

        aligned_observations = dict(observations)
        aligned_observations["policy"] = {
            self._POLICY_OBSERVATION_KEY_ALIASES.get(term_name, term_name): value
            for term_name, value in policy_observations.items()
        }
        return aligned_observations

    def reset(self, *args, **kwargs):
        observations, extras = super().reset(*args, **kwargs)
        return self._align_observation_keys(observations), extras

    def step(self, actions):
        observations, rewards, terminated, truncated, extras = super().step(actions)
        return self._align_observation_keys(observations), rewards, terminated, truncated, extras
