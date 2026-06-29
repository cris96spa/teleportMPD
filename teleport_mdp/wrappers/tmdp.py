from typing import Any

import numpy as np
from gymnasium import Wrapper
from numpy import ndarray

from teleport_mdp.environments.teleport_env import TeleportEnv

STOCHASTICITY_THRESHOLD = 1e-7


class TMDP(Wrapper):
    """Teleportation MDP wrapper for Gym environments.

    A TMDP is a Markov Decision Process where the agent can teleport to a random
    state with a given probability controlled by the parameter
    ``teleport_probability``. At each step, a coin is tossed:

    - With probability ``teleport_probability``, the agent teleports to a random
      state.
    - With probability ``1 - teleport_probability``, the agent takes a step in
      the environment.

    Args:
        env: the environment to wrap.
        teleport_prob_distribution: the teleportation probability distribution
            over the state space.
        teleport_probability: the probability of teleporting to a random state.
            Default is 0.0.
    """

    def __init__(
        self,
        env: TeleportEnv,
        teleport_prob_distribution: ndarray[Any, np.dtype[Any]],
        teleport_probability: float = 0.0,
    ) -> None:
        super().__init__(env)
        if not isinstance(env, TeleportEnv):
            raise ValueError("The environment must be a subclass of TeleportEnv.")
        # Check for stochasticity
        if not abs(sum(teleport_prob_distribution) - 1) <= STOCHASTICITY_THRESHOLD:
            raise ValueError("The teleport distribution must sum to 1.")

        # Check teleport probability
        if teleport_probability < 0.0 or teleport_probability > 1.0:
            raise ValueError("The teleport probability must be in the range [0, 1].")
        self.teleport_prob_distribution = teleport_prob_distribution
        self.teleport_probability = teleport_probability
        self.env: TeleportEnv = env
        self.reset()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Take a step in the environment.

        The agent can either take a step in the environment or teleport to a
        random state:

        - With probability ``teleport_probability``, the agent teleports to a
          random state.
        - With probability ``1 - teleport_probability``, the agent takes a step
          in the environment.

        Args:
            action: the action to take.

        Returns:
            A tuple ``(s_prime, r, terminated, truncated, info)`` with the next
            state, reward, termination flag, truncation flag, and an info
            dictionary containing a teleport flag.
        """
        if self.env.np_random.random() <= self.teleport_probability:
            # Teleport branch
            s_prime: int = self.env.teleport(self.teleport_prob_distribution)
            r = 0.0
            truncated = False
            terminated = False
            info = {
                "teleport": True,
                "prob": self.teleport_prob_distribution[s_prime],
            }
        else:
            s_prime, r, terminated, truncated, info = self.env.step(action)
            r = float(r) * (1 - self.teleport_probability)
            info["teleport"] = False

        if self.render_mode == "human":
            self.render()

        return s_prime, r, terminated, truncated, info

    def render(self):
        """Render the environment."""
        self.env.render()

    def reset(self, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        """Reset the environment.

        Args:
            **kwargs: additional keyword arguments forwarded to the wrapped
                environment's ``reset`` method.

        Returns:
            A tuple ``(state, info)`` with the initial state of the environment
            and an info dictionary.
        """
        return self.env.reset(**kwargs)
