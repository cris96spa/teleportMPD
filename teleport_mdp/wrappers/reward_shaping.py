from typing import Any

import numpy as np
from gymnasium import Env, Wrapper
from numpy.typing import NDArray

from teleport_mdp.constants import GOAL_TILE

#: Reward assigned to the goal tile, overriding its distance bin.
GOAL_REWARD = 1.0
#: Closed reward interval assigned to non-goal tiles, nearest-to-farthest bin.
SHAPE_RANGE = (-1.0, 0.0)


def manhattan_shaped_rewards(
    desc: NDArray[np.bytes_],
    n_bins: int,
    *,
    shape_range: tuple[float, float] = SHAPE_RANGE,
    goal_reward: float = GOAL_REWARD,
) -> NDArray[np.float64]:
    """Per-state binned reward, decreasing with Manhattan distance to the goal.

    Reproduces the thesis's binned Manhattan-distance reward shaping: each cell's
    distance to the nearest goal is bucketed into `n_bins` bins, and each bin is
    given a reward linearly interpolated across `shape_range` (nearest bin ->
    `shape_range[1]`, farthest -> `shape_range[0]`). The goal tile itself is set
    to `goal_reward`, above the whole shaped range. This is a *direct* binned
    reward, not a strictly potential-based one, so it changes the optimal value
    scale; it is applied identically to every compared agent, keeping the vanilla
    vs. curriculum comparison fair.

    Args:
        desc: The FrozenLake map as a `[nrow, ncol]` array of tile-char bytes.
        n_bins: Number of distance bins; must be positive.
        shape_range: `(farthest_reward, nearest_reward)` for the non-goal bins.
        goal_reward: Reward assigned to every goal tile.

    Returns:
        A 1-D array of length `nrow * ncol` indexed by state
        (`row * ncol + col`).

    Raises:
        ValueError: If `n_bins` is not positive or the map has no goal tile.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive to shape rewards, got {n_bins}.")

    nrow, ncol = desc.shape
    goal_cells = np.argwhere(desc == GOAL_TILE)
    if goal_cells.size == 0:
        raise ValueError("Cannot shape rewards: the map has no goal tile.")

    rows, cols = np.indices((nrow, ncol))
    distance = np.min(
        [np.abs(rows - goal_row) + np.abs(cols - goal_col) for goal_row, goal_col in goal_cells],
        axis=0,
    )

    max_distance = float(distance.max())
    # Descending edges pair a smaller distance with a later (higher) bin, so the
    # reward grows as the goal gets closer; np.digitize buckets each distance.
    edges = np.flip(np.linspace(0.0, max_distance, n_bins + 1))
    bin_rewards = np.linspace(shape_range[0], shape_range[1], n_bins + 1)
    bin_index = np.clip(np.digitize(distance, edges, right=False) - 1, 0, n_bins)

    shaped = bin_rewards[bin_index]
    shaped[desc == GOAL_TILE] = goal_reward
    return shaped.astype(np.float64).ravel()


class ManhattanRewardShaping(Wrapper):
    """Replace FrozenLake's sparse reward with a binned Manhattan-distance signal.

    Sitting *outside* the :class:`~teleport_mdp.wrappers.tmdp.TMDP` wrapper, this
    reshapes only real environment steps: a teleport step (flagged
    `info["teleport"]`) keeps its reward of 0, so the teleport truncation is
    preserved. Every non-teleport step's reward is replaced by the shaped reward
    of the state landed in (`manhattan_shaped_rewards`), which already assigns the
    goal `+1`. With `n_bins == 0` the wrapper is not applied at all (the factory
    leaves the env unshaped), reproducing the sparse baseline exactly.

    Args:
        env: The env to wrap; its unwrapped base must expose the FrozenLake
            `desc` grid.
        n_bins: Number of Manhattan-distance bins; must be positive.
    """

    def __init__(self, env: Env, n_bins: int) -> None:
        super().__init__(env)
        self._shaped_rewards = manhattan_shaped_rewards(env.unwrapped.desc, n_bins)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Step the wrapped env, replacing non-teleport rewards with shaped ones.

        Args:
            action: The action to take.

        Returns:
            The wrapped env's `(state, reward, terminated, truncated, info)`, with
            `reward` set to the shaped reward of `state` on non-teleport steps and
            left untouched (0) on teleport steps.
        """
        state, reward, terminated, truncated, info = self.env.step(action)
        if info.get("teleport", False):
            return state, reward, terminated, truncated, info
        return state, float(self._shaped_rewards[state]), terminated, truncated, info
