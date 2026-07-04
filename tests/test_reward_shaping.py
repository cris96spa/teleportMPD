from itertools import pairwise

import numpy as np
import pytest

from teleport_mdp.constants import RIGHT
from teleport_mdp.environments.factory import build_xi, make_env, wrap_tmdp
from teleport_mdp.models import EnvConfig, TeleportConfig
from teleport_mdp.wrappers.reward_shaping import ManhattanRewardShaping, manhattan_shaped_rewards


def _corridor() -> ManhattanRewardShaping:
    """A deterministic `SFFG` corridor shaped into 3 Manhattan bins."""
    env = make_env(EnvConfig(desc=["SFFG"], is_slippery=False))
    return ManhattanRewardShaping(env, n_bins=3)


def test_shaped_grid_matches_hand_computed_4x4():
    """The 4x4, 3-bin shaping matches the hand-computed distance grid.

    Bins over Manhattan distance to the goal at (3, 3): distance 0 is the goal
    (+1), distances 1..2 -> -1/3, 3..4 -> -2/3, 5..6 -> -1.
    """
    desc = np.asarray(["SFFF", "FHFH", "FFFH", "HFFG"], dtype="c")
    expected = np.array(
        [
            [-1.0, -1.0, -1.0, -2 / 3],
            [-1.0, -1.0, -2 / 3, -2 / 3],
            [-1.0, -2 / 3, -2 / 3, -1 / 3],
            [-2 / 3, -2 / 3, -1 / 3, 1.0],
        ]
    )
    shaped = manhattan_shaped_rewards(desc, n_bins=3).reshape(4, 4)
    assert np.allclose(shaped, expected)


def test_goal_state_gets_plus_one():
    """The goal tile is set to +1, above the whole shaped range."""
    desc = np.asarray(["SFFG"], dtype="c")
    shaped = manhattan_shaped_rewards(desc, n_bins=3)
    assert shaped[3] == pytest.approx(1.0)


def test_reward_is_monotone_in_distance_on_8x8():
    """With 10 bins, the non-goal shaped reward never increases with distance."""
    desc = np.asarray(
        [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ],
        dtype="c",
    )
    nrow, ncol = desc.shape
    goal_row, goal_col = np.argwhere(desc == b"G")[0]
    distance = np.array(
        [abs(s // ncol - goal_row) + abs(s % ncol - goal_col) for s in range(nrow * ncol)]
    )
    shaped = manhattan_shaped_rewards(desc, n_bins=10)

    non_goal = distance > 0  # the goal (distance 0) is overridden to +1
    assert shaped[non_goal].min() >= -1.0
    assert shaped[non_goal].max() <= 0.0
    ordered = shaped[non_goal][np.argsort(distance[non_goal])]
    assert all(nearer >= farther - 1e-12 for nearer, farther in pairwise(ordered))


def test_pure_function_rejects_non_positive_bins():
    """Shaping needs at least one bin; 0 is handled by not wrapping at all."""
    with pytest.raises(ValueError, match="n_bins must be positive"):
        manhattan_shaped_rewards(np.asarray(["SFFG"], dtype="c"), n_bins=0)


def test_map_without_goal_is_rejected():
    """A map with no goal tile has no distance origin to shape against."""
    with pytest.raises(ValueError, match="no goal tile"):
        manhattan_shaped_rewards(np.asarray(["SFFF"], dtype="c"), n_bins=3)


def test_step_replaces_reward_with_shaped_value():
    """Each non-teleport step's reward becomes the shaped reward of the state landed in."""
    shaped_env = _corridor()
    shaped_env.reset()

    _, reward_at_1, _, _, _ = shaped_env.step(RIGHT)
    assert reward_at_1 == pytest.approx(-1.0)
    _, reward_at_2, _, _, _ = shaped_env.step(RIGHT)
    assert reward_at_2 == pytest.approx(-2 / 3)
    _, reward_at_goal, terminated, _, _ = shaped_env.step(RIGHT)
    assert reward_at_goal == pytest.approx(1.0)
    assert terminated


def test_teleport_steps_keep_zero_reward():
    """Shaping sits outside TMDP, so teleport steps keep their reward of 0."""
    base = make_env(EnvConfig(desc=["SFFG"], is_slippery=False))
    xi = build_xi(base, TeleportConfig(distribution="uniform_nonterminal"))
    shaped_env = ManhattanRewardShaping(wrap_tmdp(base, xi, tau=0.9), n_bins=3)
    shaped_env.reset()

    saw_teleport = False
    for _ in range(200):
        _, reward, terminated, _, info = shaped_env.step(RIGHT)
        if info.get("teleport"):
            saw_teleport = True
            assert reward == pytest.approx(0.0)
        if terminated:
            shaped_env.reset()
    assert saw_teleport, "Expected at least one teleport at tau=0.9."
