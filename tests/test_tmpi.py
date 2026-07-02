"""Tests for Teleport Model Policy Iteration: bound validity, monotonicity, tau annealing."""

from itertools import pairwise

import numpy as np
import pytest

from teleport_mdp.environments.factory import make_env
from teleport_mdp.models import EnvConfig, TMPIConfig
from teleport_mdp.tabular import TMPI
from teleport_mdp.tabular.tmpi import _dense_model, _teleport_kernel
from teleport_mdp.wrappers.tmdp import TMDP


def _wrap(env, tau: float) -> TMDP:
    """Wrap an env in a uniform-over-nonterminal teleport MDP at rate tau."""
    n = int(env.observation_space.n)
    xi = np.array([0.0 if env.is_terminal(s) else 1.0 for s in range(n)])
    xi /= xi.sum()
    return TMDP(env=env, teleport_prob_distribution=xi, teleport_probability=tau)


def test_bound_is_a_true_lower_bound_on_realized_gain():
    """Every iteration's realized J gain is at least the Teleport Bound it reported."""
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.5)
    config = TMPIConfig(threshold=1e-9, max_iterations=400, temperature=0.2)

    result = TMPI(config, gamma=0.9).optimize(tmdp)

    perf = result.performance_history
    for i in range(len(perf) - 1):
        realized_gain = perf[i + 1] - perf[i]
        assert realized_gain >= result.bound_history[i] - 1e-8


def test_performance_is_monotonically_non_decreasing():
    """TMPI only accepts non-negative-bound steps, so J never decreases."""
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.5)
    config = TMPIConfig(threshold=1e-9, max_iterations=400, temperature=0.2)

    result = TMPI(config, gamma=0.9).optimize(tmdp)

    assert all(b >= a - 1e-9 for a, b in pairwise(result.performance_history))


def test_drives_tau_to_zero_on_corridor():
    """On a deterministic corridor, TMPI anneals the teleport rate all the way to 0."""
    env = make_env(EnvConfig(desc=["SFFG"], is_slippery=False))
    tmdp = _wrap(env, tau=0.4)
    config = TMPIConfig(threshold=1e-9, max_iterations=2000, temperature=0.1)

    result = TMPI(config, gamma=0.9).optimize(tmdp)

    assert result.tau_history[-1] == pytest.approx(0.0)
    assert tmdp.teleport_probability == pytest.approx(0.0)
    # tau is monotonically non-increasing throughout.
    assert all(b <= a + 1e-12 for a, b in zip(result.tau_history, result.tau_history[1:]))


def test_performance_never_exceeds_one_on_frozenlake():
    """The +1 goal reward is collected at most once: teleport cannot exit the goal.

    A bug that mixes teleport into terminal states lets the agent leave the
    absorbing goal and re-collect reward, inflating J above 1.
    """
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.6)
    config = TMPIConfig(threshold=1e-9, max_iterations=300, temperature=0.1)

    result = TMPI(config, gamma=0.9).optimize(tmdp)

    assert result.performance <= 1.0 + 1e-9
    assert float(np.max(result.v)) <= 1.0 + 1e-9


def test_teleport_kernel_keeps_terminal_states_absorbing():
    """The teleport kernel mixes xi only on non-terminal rows; terminals stay as P."""
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.5)
    p, _, _, non_terminal = _dense_model(tmdp)
    xi = np.asarray(tmdp.teleport_prob_distribution)

    p_tau = _teleport_kernel(p, xi, tau=0.5, non_terminal=non_terminal)

    terminals = np.flatnonzero(~non_terminal.astype(bool))
    assert np.allclose(p_tau[terminals], p[terminals])
    non_terminals = np.flatnonzero(non_terminal.astype(bool))
    assert np.allclose(p_tau[non_terminals], 0.5 * p[non_terminals] + 0.5 * xi[None, None, :])


def test_tau_zero_reduces_to_policy_iteration():
    """With tau already 0 there is no model change, yet J still improves via policy steps."""
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.0)
    config = TMPIConfig(threshold=1e-9, max_iterations=300, temperature=0.1)

    result = TMPI(config, gamma=0.9).optimize(tmdp)

    assert all(t == pytest.approx(0.0) for t in result.tau_history)
    assert result.performance > result.performance_history[0]


def test_logger_receives_metrics_each_iteration():
    """A logger gets performance/bound/alpha/tau at every iteration step."""
    env = make_env(EnvConfig(map_name="4x4", is_slippery=False))
    tmdp = _wrap(env, tau=0.3)
    config = TMPIConfig(threshold=1e-9, max_iterations=5, temperature=0.2)

    logged: list[tuple[dict[str, float], int]] = []

    class _Recorder:
        def log_metrics(self, metrics: dict[str, float], step: int) -> None:
            logged.append((metrics, step))

    TMPI(config, gamma=0.9).optimize(tmdp, logger=_Recorder())

    assert [step for _, step in logged] == list(range(len(logged)))
    assert all(
        {"tmpi/performance", "tmpi/bound", "tmpi/alpha", "curriculum/tau"} <= set(m)
        for m, _ in logged
    )
