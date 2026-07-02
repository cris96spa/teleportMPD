"""Tests for tabular Q-learning: convergence, teleport truncation, and curriculum."""

import numpy as np
import pytest
from gymnasium import spaces

from teleport_mdp.curriculum.scheduler import StaticTeleportScheduler
from teleport_mdp.environments.factory import make_env
from teleport_mdp.models import EnvConfig, QLearningConfig
from teleport_mdp.tabular import QLearner, get_policy
from teleport_mdp.tabular.model_functions import get_d_inf_policy
from teleport_mdp.wrappers.tmdp import TMDP


def _wrap(env, tau: float) -> TMDP:
    """Wrap an env in a uniform-over-nonterminal teleport MDP at rate tau."""
    n = int(env.observation_space.n)
    xi = np.array([0.0 if env.is_terminal(s) else 1.0 for s in range(n)])
    xi /= xi.sum()
    return TMDP(env=env, teleport_prob_distribution=xi, teleport_probability=tau)


def test_converges_to_optimal_q_on_tiny_chain():
    """On a 2-state corridor, Q* is the closed-form geometric value, learned to <1e-2."""
    # A minimal deterministic FrozenLake: 1x3 corridor S F G. Action RIGHT (2) advances.
    env = make_env(EnvConfig(desc=["SFG"], is_slippery=False))
    tmdp = _wrap(env, tau=0.0)
    gamma = 0.9
    config = QLearningConfig(alpha=0.5, eps=0.3, episodes=4000, status_step=500)

    result = QLearner(config, gamma, max_steps_per_episode=50, seed=0).train(tmdp)

    # Optimal: from S (state 0) go RIGHT to F (1), then RIGHT to G (2), reward 1 at goal.
    # Q*(0, RIGHT) = gamma * Q*(1, RIGHT) = gamma * 1 = 0.9; Q*(1, RIGHT) = 1.
    assert result.q[1, 2] == pytest.approx(1.0, abs=1e-2)
    assert result.q[0, 2] == pytest.approx(gamma, abs=1e-2)
    greedy = get_policy(result.q, deterministic=True)
    assert greedy[0, 2] == pytest.approx(1.0) and greedy[1, 2] == pytest.approx(1.0)


def test_solves_4x4_frozenlake():
    """Greedy policy from learned Q reaches the goal on 4x4 non-slippery FrozenLake."""
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.0)
    config = QLearningConfig(alpha=0.6, eps=0.4, episodes=8000, status_step=1000)

    result = QLearner(config, gamma=0.99, max_steps_per_episode=200, seed=1).train(tmdp)

    # Roll out the greedy policy on the real MDP and check it reaches the goal.
    greedy = get_policy(result.q, deterministic=True)
    obs, _ = env.reset()
    reached_goal = False
    for _ in range(100):
        action = int(np.argmax(greedy[int(obs)]))
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated and reward > 0.0:
            reached_goal = True
            break
        if terminated or truncated:
            break
    assert reached_goal


class _AlwaysTeleportEnv:
    """Minimal env whose every step is a (zero-reward) teleport, for truncation tests."""

    def __init__(self, n_states: int = 4, n_actions: int = 4) -> None:
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(n_actions)
        self.s = 0
        self.teleport_probability = 1.0

    @property
    def unwrapped(self):
        """The base env (this fake env is its own base env)."""
        return self

    def reset(self, **_kwargs):
        """Reset to the fixed start state."""
        self.s = 0
        return self.s, {}

    def step(self, _action):
        """Teleport to a fixed non-terminal state with zero reward."""
        self.s = 1
        return self.s, 0.0, False, False, {"teleport": True}


def test_teleport_step_does_not_update_q():
    """Every step teleporting means no (s, a, r, s') transition: Q stays at its init."""
    env = _AlwaysTeleportEnv()
    config = QLearningConfig(alpha=1.0, eps=0.0, episodes=20, status_step=10)

    result = QLearner(config, gamma=0.99, max_steps_per_episode=50, seed=2).train(env)  # type: ignore[arg-type]

    # No real transitions ever occur, so no TD update fires and Q remains all zeros.
    assert np.allclose(result.q, 0.0)
    assert all(r == pytest.approx(0.0) for r in result.returns)


def test_static_curriculum_anneals_tau_to_zero():
    """The static scheduler drives tau to 0 over the status steps."""
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.5)
    episodes, status_step = 1000, 100
    n_updates = episodes // status_step
    scheduler = StaticTeleportScheduler(gamma=0.99, tau_0=0.5, n_updates=n_updates)
    config = QLearningConfig(alpha=0.5, eps=0.3, episodes=episodes, status_step=status_step)

    result = QLearner(config, gamma=0.99, max_steps_per_episode=200, seed=3).train(
        tmdp, scheduler=scheduler
    )

    assert result.tau_history[0] == pytest.approx(0.5)
    assert tmdp.teleport_probability == pytest.approx(0.0, abs=1e-9)
    # Monotone non-increasing schedule.
    assert all(b <= a + 1e-12 for a, b in zip(result.tau_history, result.tau_history[1:]))


def test_logger_receives_metrics_per_status_step():
    """A logger gets train/return and curriculum/tau at every status step."""
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.0)
    config = QLearningConfig(alpha=0.5, eps=0.2, episodes=300, status_step=100)

    logged: list[tuple[dict[str, float], int]] = []

    class _Recorder:
        def log_metrics(self, metrics: dict[str, float], step: int) -> None:
            logged.append((metrics, step))

    QLearner(config, gamma=0.99, seed=4).train(tmdp, logger=_Recorder())

    assert [step for _, step in logged] == [0, 100, 200]
    assert all("train/return" in m and "curriculum/tau" in m for m, _ in logged)


def test_visit_distribution_is_normalized():
    """The visit distributions returned sum to 1."""
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.0)
    config = QLearningConfig(alpha=0.5, eps=0.3, episodes=500, status_step=100)

    result = QLearner(config, gamma=0.99, seed=5).train(tmdp)

    assert result.visit_distribution.sum() == pytest.approx(1.0)
    assert result.disc_visit_distribution.sum() == pytest.approx(1.0)
    assert get_d_inf_policy(
        get_policy(result.q, deterministic=True),
        get_policy(result.q_snapshots[-1], deterministic=True),
    ) == pytest.approx(0.0)
