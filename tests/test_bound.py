import pytest

from teleport_mdp.tabular.bound import (
    BoundMetrics,
    candidate_pairs,
    compute_alpha_0,
    compute_alpha_tau,
    compute_tau_prime_0,
    compute_tau_prime_1,
    compute_teleport_bound,
    optimal_pair,
)

GAMMA = 0.9
TAU = 0.5


@pytest.fixture
def metrics() -> BoundMetrics:
    """A hand-checkable set of bound quantities (all advantages positive)."""
    return BoundMetrics(
        pol_adv=0.2,
        model_adv=0.1,
        delta_u=1.0,
        d_inf_pol=0.5,
        d_exp_pol=0.25,
        d_inf_model=2.0,
        d_exp_model=1.0,
    )


def test_no_op_bound_is_zero(metrics):
    """B(alpha=0, tau'=tau) has no advantage and no penalty, so it is exactly 0."""
    assert compute_teleport_bound(metrics, 0.0, TAU, TAU, GAMMA) == pytest.approx(0.0)


def test_bound_hand_value_full_policy_step(metrics):
    """B(alpha=1, tau'=tau) matches the hand-computed advantage minus penalty."""
    # advantage = pol_adv / (1 - gamma); penalty = gamma * dU * dExpPol * dInfPol / (2 (1-gamma)^2).
    advantage = 0.2 / (1 - GAMMA)
    penalty = GAMMA * 1.0 * (0.25 * 0.5) / (2 * (1 - GAMMA) ** 2)
    assert compute_teleport_bound(metrics, 1.0, TAU, TAU, GAMMA) == pytest.approx(
        advantage - penalty
    )


def test_alpha_tau_matches_table_1(metrics):
    """alpha_tau = (1-gamma) A / (gamma dU D_exp^pi D_inf^pi)."""
    assert compute_alpha_tau(metrics, GAMMA) == pytest.approx(0.02 / 0.1125)


def test_alpha_0_is_alpha_tau_minus_model_penalty(metrics):
    """alpha_0 = alpha_tau - (tau/2)(D_exp^model/D_exp^pi + D_inf^model/D_inf^pi)."""
    expected = compute_alpha_tau(metrics, GAMMA) - TAU / 2 * (1.0 / 0.25 + 2.0 / 0.5)
    assert compute_alpha_0(metrics, TAU, GAMMA) == pytest.approx(expected)


def test_tau_prime_0_matches_table_1(metrics):
    """tau'_0 = tau - (1-gamma) A_model / (gamma^2 dU D_exp^model D_inf^model)."""
    expected = TAU - 0.1 * 0.1 / (GAMMA**2 * 1.0 * 1.0 * 2.0)
    assert compute_tau_prime_0(metrics, TAU, GAMMA) == pytest.approx(expected)


def test_tau_prime_1_is_tau_prime_0_plus_policy_term(metrics):
    """tau'_1 = tau'_0 + 1/(2 gamma)(D_exp^pi/D_exp^model + D_inf^pi/D_inf^model)."""
    expected = compute_tau_prime_0(metrics, TAU, GAMMA) + 1 / (2 * GAMMA) * (0.25 / 1.0 + 0.5 / 2.0)
    assert compute_tau_prime_1(metrics, TAU, GAMMA) == pytest.approx(expected)


def test_candidate_set_structure_and_clipping(metrics):
    """V holds the no-op plus the four boundary pairs, clipped to [0,1] x [0,tau]."""
    pairs = candidate_pairs(metrics, TAU, GAMMA)
    assert (0.0, TAU) in pairs  # no-op baseline is always present
    assert len(pairs) == 5
    for alpha, tau_prime in pairs:
        assert 0.0 <= alpha <= 1.0
        assert 0.0 <= tau_prime <= TAU


def test_no_policy_candidates_when_policy_advantage_nonpositive(metrics):
    """A non-positive policy advantage drops the two policy candidates."""
    pairs = candidate_pairs(metrics._replace(pol_adv=0.0), TAU, GAMMA)
    # Only the no-op and the two model candidates remain.
    assert len(pairs) == 3
    assert all(tau_prime <= TAU for _, tau_prime in pairs)


def test_no_model_candidates_when_model_advantage_nonpositive(metrics):
    """A non-positive model advantage drops the two model tau-reduction candidates.

    Only the no-op and the two policy candidates remain (the `alpha_0` policy
    candidate still sits at `tau'=0` per Table 1, but no `(0, tau'_0)` or
    `(1, tau'_1)` model candidate survives).
    """
    pairs = candidate_pairs(metrics._replace(model_adv=0.0), TAU, GAMMA)
    assert len(pairs) == 3
    assert all(alpha < 1.0 for alpha, _ in pairs)  # (1, tau'_1) is gone
    assert all(not 0.0 < tau_prime < TAU for _, tau_prime in pairs)  # (0, tau'_0) is gone


def test_optimal_pair_selects_argmax_and_snaps_small_tau(metrics):
    """optimal_pair returns the bound-maximizing candidate; tiny tau' snaps to 0."""
    pairs = candidate_pairs(metrics, TAU, GAMMA)
    bounds = [compute_teleport_bound(metrics, a, TAU, tp, GAMMA) for a, tp in pairs]
    _, _, bound_star = optimal_pair(metrics, TAU, GAMMA)
    assert bound_star == pytest.approx(max(bounds))
    assert bound_star >= 0.0  # never worse than the no-op

    # A candidate tau' just under the threshold is reported as exactly 0.
    near_zero = metrics._replace(model_adv=1e3)  # forces tau'_0 well below threshold
    _, tau_snapped, _ = optimal_pair(near_zero, TAU, GAMMA, tau_threshold=1e-6)
    assert tau_snapped == pytest.approx(0.0, abs=1e-12)
