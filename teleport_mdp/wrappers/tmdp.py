from typing import Any

import numpy as np
from gymnasium import Wrapper
from numpy import ndarray

from teleport_mdp.constants import STOCHASTICITY_THRESHOLD
from teleport_mdp.environments.teleport_env import TeleportEnv


class TMDP(Wrapper):
    """Teleportation MDP wrapper for Gym environments.

    A TMDP is a Markov Decision Process where the agent can teleport to a random
    state with a given probability controlled by the parameter
    `teleport_probability`. At each step, a coin is tossed:

    - With probability `teleport_probability`, the agent teleports to a random
      state.
    - With probability `1 - teleport_probability`, the agent takes a step in
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

        self.teleport_prob_distribution = teleport_prob_distribution
        self.env: TeleportEnv = env
        self.update_tau(teleport_probability)
        self.reset()

    def update_tau(self, tau: float) -> None:
        """Set the teleport rate tau at runtime.

        Used by the teleport-rate curriculum (thesis Algorithms 2 & 3) to change
        the teleport probability during training.

        Args:
            tau: the new teleport rate, in `[0, 1)`.

        Raises:
            ValueError: if `tau` is outside `[0, 1)`. A rate of exactly 1
                would give a degenerate effective discount `gamma * (1 - tau)`
                of 0 and is rejected.
        """
        if not 0.0 <= tau < 1.0:
            raise ValueError(f"The teleport rate must be in the range [0, 1), got {tau}.")
        self.teleport_probability = tau

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Take a step in the environment.

        The agent can either take a step in the environment or teleport to a
        random state:

        - With probability `teleport_probability`, the agent teleports to a
          random state.
        - With probability `1 - teleport_probability`, the agent takes a step
          in the environment.

        Args:
            action: the action to take.

        Returns:
            A tuple `(s_prime, r, terminated, truncated, info)` with the next
            state, reward, termination flag, truncation flag, and an info
            dictionary containing a teleport flag.
        """
        # Inverse-CDF gating: with U ~ Uniform[0, 1), `U < tau` teleports with
        # probability exactly tau, so tau == 0 is behaviourally identical to the
        # base env (a `<=` here would teleport on the measure-zero draw U == 0).
        if self.env.np_random.random() < self.teleport_probability:
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
            r = float(r)
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
                environment's `reset` method.

        Returns:
            A tuple `(state, info)` with the initial state of the environment
            and an info dictionary.
        """
        return self.env.reset(**kwargs)
