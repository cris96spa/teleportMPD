from typing import Any

import numpy as np
import torch as th
from gymnasium.spaces import Space
from stable_baselines3.common.buffers import RolloutBuffer


class TeleportRolloutBuffer(RolloutBuffer):
    """:class:`~stable_baselines3.common.buffers.RolloutBuffer` with teleport truncation.

    Records a per-transition `teleport` flag from `info["teleport"]` and, in
    :meth:`compute_returns_and_advantage`, masks the teleport step so the GAE
    recursion neither credits the teleport delta nor lets advantage from later steps
    leak back across the teleport boundary. With all flags `False` it reduces
    exactly to the SB3 buffer.

    Args:
        buffer_size: Number of transitions stored per environment.
        observation_space: The observation space.
        action_space: The action space.
        device: Torch device for the returned tensors.
        gamma: Discount factor.
        gae_lambda: GAE lambda trade-off factor.
        n_envs: Number of parallel environments.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: Space,
        action_space: Space,
        device: th.device | str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        n_envs: int = 1,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )
        self.teleport_flags = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def reset(self) -> None:
        """Reset the buffer and clear all teleport flags."""
        super().reset()
        self.teleport_flags = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, *args: Any, infos: list[dict[str, Any]], **kwargs: Any) -> None:
        """Record the teleport flags for the current step, then delegate to SB3.

        Args:
            *args: Positional `RolloutBuffer.add` arguments (obs, action, reward,
                episode_start, value, log_prob).
            infos: The per-env `info` dicts from `env.step`; a truthy
                `info["teleport"]` marks the transition as a teleport.
            **kwargs: Extra keyword arguments forwarded to `RolloutBuffer.add`.
        """
        for idx, info in enumerate(infos):
            self.teleport_flags[self.pos, idx] = 1.0 if info.get("teleport", False) else 0.0
        super().add(*args, **kwargs)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """Compute truncated GAE advantages and returns.

        Mirrors SB3's GAE recursion but multiplies the per-step TD error `delta` and
        the accumulated `last_gae_lam` by `(1 - teleport)`. Zeroing `delta` drops
        the (already-zero-reward) teleport transition from credit assignment; zeroing
        the carried `last_gae_lam` stops advantage from steps after the teleport from
        propagating back to steps before it — the trajectory truncation required by the
        thesis (`gamma_eff = gamma(1 - tau)`; teleport restarts from `xi`).

        Args:
            last_values: Value estimates for the observation after the last step.
            dones: Per-env done flags for the last step.
        """
        last_values_np = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values_np
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            teleport = self.teleport_flags[step]
            keep = 1.0 - teleport

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            delta = delta * keep
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam * keep
            )
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values
