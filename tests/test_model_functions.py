"""Tests for the tabular model functions on small, hand-checkable MDPs."""

import numpy as np
import pytest

from teleport_mdp.tabular import (
    compute_d,
    compute_delta,
    compute_expected_model_advantage,
    compute_expected_policy_advantage,
    compute_model_advantage,
    compute_p_tau,
    compute_policy_advantage,
    compute_q_from_u,
    compute_r_s_a,
    compute_relative_model_advantage,
    compute_relative_policy_advantage,
    compute_transition_kernel,
    compute_u_from_v,
    compute_v_from_q,
    get_d_exp_model,
    get_d_exp_policy,
    get_d_inf_model,
    get_d_inf_policy,
    get_policy,
    get_sup_difference,
    softmax_policy,
)


@pytest.fixture
def toy_mdp():
    """A 2-state, 2-action MDP with deterministic transitions and a fixed policy."""
    # P[s, a, s']: action 0 stays, action 1 swaps state.
    p = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    reward = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    pi = np.array([[1.0, 0.0], [0.0, 1.0]])  # state 0 -> action 0, state 1 -> action 1
    xi = np.array([0.5, 0.5])
    return p, reward, pi, xi


# region Model construction
def test_compute_p_tau_blends_and_stays_stochastic(toy_mdp):
    """P_tau mixes P and xi by tau and keeps every (s, a) row a distribution."""
    p, _, _, xi = toy_mdp
    p_tau = compute_p_tau(p, xi, tau=0.25)
    expected_00 = 0.75 * np.array([1.0, 0.0]) + 0.25 * xi
    assert np.allclose(p_tau[0, 0], expected_00)
    assert np.allclose(p_tau.sum(axis=2), 1.0)


def test_compute_p_tau_endpoints(toy_mdp):
    """tau=0 recovers P, tau=1 collapses every row to xi."""
    p, _, _, xi = toy_mdp
    assert np.allclose(compute_p_tau(p, xi, 0.0), p)
    assert np.allclose(compute_p_tau(p, xi, 1.0), np.broadcast_to(xi, p.shape))


def test_compute_r_s_a_hand_value(toy_mdp):
    """R(s, a) = sum_s' P(s'|s,a) R(s,a,s') on the deterministic toy MDP."""
    p, reward, _, _ = toy_mdp
    r_s_a = compute_r_s_a(p, reward)
    # Deterministic P picks one next state per (s, a): the diagonal reward entries.
    expected = np.array([[1.0, 4.0], [5.0, 8.0]])
    assert np.allclose(r_s_a, expected)


def test_compute_transition_kernel(toy_mdp):
    """The policy-marginalised kernel selects each state's chosen action row."""
    p, _, pi, _ = toy_mdp
    kernel = compute_transition_kernel(p, pi)
    assert np.allclose(kernel, np.array([[1.0, 0.0], [0.0, 1.0]]))


# endregion


# region Policies
def test_get_policy_deterministic_is_one_hot_argmax():
    """A deterministic policy puts all mass on the argmax action."""
    q = np.array([[0.1, 0.9], [2.0, -1.0]])
    pi = get_policy(q, deterministic=True)
    assert np.allclose(pi, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_softmax_policy_rows_sum_to_one_and_rank_preserving():
    """Softmax policies are valid distributions ordered by their scores."""
    q = np.array([[0.0, 1.0], [3.0, 1.0]])
    pi = softmax_policy(q)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert pi[0, 1] > pi[0, 0]
    assert pi[1, 0] > pi[1, 1]


def test_softmax_policy_rejects_nonpositive_temperature():
    """A non-positive temperature is rejected."""
    with pytest.raises(ValueError):
        softmax_policy(np.zeros((2, 2)), temperature=0.0)


# endregion


# region State-visit distributions
def test_compute_d_matches_closed_form_inverse():
    """compute_d matches the explicit (1-gamma) mu (I - gamma P_pi)^-1 closed form."""
    # Single-action MDP so P_pi == P_tau[:, 0, :] is a plain 2x2 stochastic matrix.
    p_pi = np.array([[0.7, 0.3], [0.4, 0.6]])
    p_tau = p_pi.reshape(2, 1, 2)
    pi = np.ones((2, 1))
    mu = np.array([1.0, 0.0])
    gamma = 0.9

    d = compute_d(mu, p_tau, pi, gamma)
    reference = (1.0 - gamma) * mu @ np.linalg.inv(np.eye(2) - gamma * p_pi)

    assert np.allclose(d, reference, atol=1e-8)
    assert d.sum() == pytest.approx(1.0)
    assert np.all(d >= 0.0)


def test_compute_d_uniform_chain_is_stationary():
    """A doubly-stochastic kernel from uniform mu yields the uniform visit dist."""
    p_pi = np.array([[0.5, 0.5], [0.5, 0.5]])
    p_tau = p_pi.reshape(2, 1, 2)
    pi = np.ones((2, 1))
    mu = np.array([0.5, 0.5])

    d = compute_d(mu, p_tau, pi, gamma=0.95)
    assert np.allclose(d, np.array([0.5, 0.5]))


def test_compute_delta_is_policy_weighted_d(toy_mdp):
    """delta(s, a) = pi(a|s) d(s) and sums to 1."""
    _, _, pi, _ = toy_mdp
    d = np.array([0.3, 0.7])
    delta = compute_delta(d, pi)
    assert np.allclose(delta, pi * d[:, None])
    assert delta.sum() == pytest.approx(1.0)


def test_compute_delta_rejects_negative_mass():
    """A negative state-visit entry raises."""
    pi = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError):
        compute_delta(np.array([-0.1]), pi)


# endregion


# region Value functions
def test_value_chain_round_trip(toy_mdp):
    """V from Q, U from V, then Q from U reconstructs Q on the toy MDP."""
    p, reward, pi, xi = toy_mdp
    p_tau = compute_p_tau(p, xi, tau=0.2)
    q = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = compute_v_from_q(q, pi)
    # V picks the policy action's Q value.
    assert np.allclose(v, np.array([1.0, 4.0]))

    u = compute_u_from_v(reward, gamma=0.9, v=v)
    assert np.allclose(u, reward + 0.9 * v.reshape(1, 1, -1))

    q_back = compute_q_from_u(p_tau, u)
    # Reconstructed Q equals R(s,a) under P_tau plus gamma * E_{P_tau}[V].
    r_tau = compute_r_s_a(p_tau, reward)
    expected = r_tau + 0.9 * np.einsum("san,n->sa", p_tau, v)
    assert np.allclose(q_back, expected)


# endregion


# region Advantage identities
def test_policy_advantage_zero_under_policy(toy_mdp):
    """sum_a pi(a|s) A(s, a) = 0 when V is the policy value of Q."""
    _, _, pi, _ = toy_mdp
    q = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = compute_v_from_q(q, pi)
    a = compute_policy_advantage(q, v)
    assert np.allclose(np.einsum("sa,sa->s", pi, a), 0.0, atol=1e-12)


def test_model_advantage_zero_under_p_tau(toy_mdp):
    """sum_s' P_tau(s'|s,a) A(s, a, s') = 0 when Q is consistent with U."""
    p, reward, pi, xi = toy_mdp
    p_tau = compute_p_tau(p, xi, tau=0.3)
    q = np.array([[0.5, 1.5], [2.5, 3.5]])
    v = compute_v_from_q(q, pi)
    u = compute_u_from_v(reward, gamma=0.9, v=v)
    q_consistent = compute_q_from_u(p_tau, u)
    model_adv = compute_model_advantage(u, q_consistent)
    assert np.allclose(np.einsum("san,san->sa", p_tau, model_adv), 0.0, atol=1e-12)


def test_relative_policy_advantage_against_self_is_zero(toy_mdp):
    """The relative policy advantage of pi against its own advantage is zero."""
    _, _, pi, _ = toy_mdp
    q = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = compute_v_from_q(q, pi)
    a = compute_policy_advantage(q, v)
    rel = compute_relative_policy_advantage(pi, a)
    assert np.allclose(rel, 0.0, atol=1e-12)


def test_relative_model_advantage_hand_value(toy_mdp):
    """A_{P,xi}(s,a) = sum_s' (P - xi)(s') U(s,a,s') matches a manual sum."""
    p, reward, pi, xi = toy_mdp
    v = compute_v_from_q(np.array([[1.0, 2.0], [3.0, 4.0]]), pi)
    u = compute_u_from_v(reward, gamma=0.9, v=v)
    rel = compute_relative_model_advantage(p, xi, u)
    expected = np.einsum("san,san->sa", p - xi[None, None, :], u)
    assert np.allclose(rel, expected)


def test_expected_advantages_are_distribution_weighted(toy_mdp):
    """Expected advantages are d/delta-weighted sums of the relative advantages."""
    _, _, pi, _ = toy_mdp
    d = np.array([0.4, 0.6])
    delta = compute_delta(d, pi)
    rel_pol = np.array([1.0, -2.0])
    rel_model = np.array([[0.5, 1.0], [2.0, -1.0]])
    assert compute_expected_policy_advantage(rel_pol, d) == pytest.approx(0.4 - 1.2)
    assert compute_expected_model_advantage(rel_model, delta) == pytest.approx(
        float(np.sum(delta * rel_model))
    )


# endregion


# region Dissimilarity terms
def test_sup_difference_is_range():
    """get_sup_difference returns max minus min."""
    assert get_sup_difference(np.array([1.0, 5.0, -2.0])) == pytest.approx(7.0)


def test_d_inf_and_exp_policy_hand_values():
    """Policy dissimilarities match hand-computed L1 norms."""
    pi = np.array([[1.0, 0.0], [0.0, 1.0]])
    pi_prime = np.array([[0.0, 1.0], [0.0, 1.0]])
    # State 0 differs by L1 = 2, state 1 by 0.
    assert get_d_inf_policy(pi, pi_prime) == pytest.approx(2.0)
    d = np.array([0.5, 0.5])
    assert get_d_exp_policy(pi, pi_prime, d) == pytest.approx(1.0)


def test_d_inf_and_exp_model_hand_values(toy_mdp):
    """Model dissimilarities match hand-computed L1 norms vs xi."""
    p, _, pi, xi = toy_mdp
    # Each deterministic row is e.g. [1, 0] vs xi=[0.5, 0.5] -> L1 = 1.0.
    assert get_d_inf_model(p, xi) == pytest.approx(1.0)
    d = np.array([0.4, 0.6])
    delta = compute_delta(d, pi)
    # Every (s, a) has L1 = 1.0, so the expected value equals sum(delta) = 1.0.
    assert get_d_exp_model(p, xi, delta) == pytest.approx(1.0)


# endregion
