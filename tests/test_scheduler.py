import pytest

from teleport_mdp.curriculum import (
    DynamicTeleportScheduler,
    StaticTeleportScheduler,
    TeleportScheduler,
)


@pytest.mark.parametrize(
    ("gamma", "tau", "n"),
    [(0.99, 0.9, 50), (0.95, 0.5, 10), (0.9, 0.8, 123)],
)
def test_eps_model_and_tau_prime_are_inverse(gamma, tau, n):
    """A single decrement at the n-update budget removes exactly tau / n."""
    eps_tau = TeleportScheduler.compute_eps_model(gamma, tau, n)
    tau_prime = TeleportScheduler.compute_tau_prime(gamma, tau, eps_tau)
    assert tau_prime == pytest.approx(tau - tau / n)


@pytest.mark.parametrize(
    ("gamma", "tau_0", "n"),
    [(0.99, 0.9, 50), (0.95, 0.5, 10), (0.9, 0.8, 7)],
)
def test_compute_n_updates_round_trip(gamma, tau_0, n):
    """compute_n_updates inverts compute_eps_model for the budget that reaches 0 in n."""
    eps_tau = TeleportScheduler.compute_eps_model(gamma, tau_0, n)
    assert TeleportScheduler.compute_n_updates(gamma, tau_0, eps_tau) == n


@pytest.mark.parametrize(
    ("gamma", "tau_0", "n"),
    [(0.99, 0.9, 50), (0.95, 0.5, 10), (0.9, 0.8, 25)],
)
def test_static_reaches_zero_at_n_updates(gamma, tau_0, n):
    """Static schedule is monotone non-increasing and hits exactly 0 at update n."""
    scheduler = StaticTeleportScheduler(gamma, tau_0, n)
    tau = tau_0
    prev = tau_0
    for step in range(1, n + 1):
        tau = scheduler.next_tau(tau)
        assert tau <= prev + 1e-12
        assert tau >= -1e-12
        if step < n:
            assert tau > 0.0
        prev = tau
    assert tau == pytest.approx(0.0, abs=1e-9)


def test_static_clamps_below_zero():
    """Extra calls past convergence stay pinned at 0."""
    scheduler = StaticTeleportScheduler(0.99, 0.9, 5)
    tau = 0.0
    assert scheduler.next_tau(tau) == pytest.approx(0.0)


def test_static_rejects_nonpositive_tau0():
    """A static curriculum needs a positive initial rate to anneal."""
    with pytest.raises(ValueError):
        StaticTeleportScheduler(0.99, 0.0, 10)


def test_dynamic_large_shift_pauses_anneal():
    """A policy shift that consumes the whole budget leaves tau unchanged."""
    scheduler = DynamicTeleportScheduler(gamma=0.99, eps=0.1, eps_tau_max=0.05)
    # eps_pi = gamma/(1-gamma) * D_inf = 99 * D_inf; D_inf = 0.01 -> eps_pi = 0.99 > eps.
    tau = scheduler.next_tau(0.9, policy_shift=0.01)
    assert tau == pytest.approx(0.9)


def test_dynamic_small_shift_decreases_tau():
    """A tiny policy shift leaves budget to lower tau (hand-computed)."""
    gamma, eps, eps_tau_max = 0.99, 1.0, 0.05
    scheduler = DynamicTeleportScheduler(gamma=gamma, eps=eps, eps_tau_max=eps_tau_max)
    d_inf = 0.001
    eps_pi = gamma / (1.0 - gamma) * d_inf  # 0.099
    eps_tau = min(eps - eps_pi, eps_tau_max)  # min(0.901, 0.05) = 0.05
    expected = TeleportScheduler.compute_tau_prime(gamma, 0.9, eps_tau)
    tau = scheduler.next_tau(0.9, policy_shift=d_inf)
    assert tau == pytest.approx(expected)
    assert tau < 0.9


def test_dynamic_never_below_zero_and_stays_at_zero():
    """Tau is clamped at 0 and stays there regardless of policy shift."""
    scheduler = DynamicTeleportScheduler(gamma=0.99, eps=1.0, eps_tau_max=10.0)
    assert scheduler.next_tau(0.0, policy_shift=0.0) == pytest.approx(0.0)
    tau = scheduler.next_tau(0.001, policy_shift=0.0)
    assert tau >= 0.0


def test_dynamic_requires_policy_shift():
    """The dynamic scheduler cannot run without the policy shift."""
    scheduler = DynamicTeleportScheduler(gamma=0.99, eps=0.1, eps_tau_max=0.05)
    with pytest.raises(ValueError):
        scheduler.next_tau(0.9)
