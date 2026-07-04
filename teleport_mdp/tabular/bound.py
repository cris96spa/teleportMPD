from typing import NamedTuple

FloatCandidate = tuple[float, float]


class BoundMetrics(NamedTuple):
    """Exact bound quantities of the current model-policy pair `(P_tau, pi)`.

    Attributes:
        pol_adv: Expected relative policy advantage of the target policy over
            `pi`, `A^{pi_bar, tau}_{pi, mu}` (scalar).
        model_adv: Expected relative model advantage of `xi` over `P`,
            `A^{xi, pi}_{P, tau, mu}` (scalar).
        delta_u: Range `sup U - inf U` of the state-action-nextstate value.
        d_inf_pol: Sup L1 policy distance `D^{pi_bar, pi}_inf`.
        d_exp_pol: Expected L1 policy distance `D^{pi_bar, pi}_exp`.
        d_inf_model: Sup L1 model distance `D^{P, xi}_inf`.
        d_exp_model: Expected L1 model distance `D^{P, xi}_exp`.
    """

    pol_adv: float
    model_adv: float
    delta_u: float
    d_inf_pol: float
    d_exp_pol: float
    d_inf_model: float
    d_exp_model: float


def compute_teleport_bound(
    metrics: BoundMetrics, alpha: float, tau: float, tau_prime: float, gamma: float
) -> float:
    """Teleport Bound `B(alpha, tau')` (Thm 4.2), a lower bound on `J' - J`.

    The candidate policy is `pi' = alpha * pi_bar + (1 - alpha) pi`, so the
    policy-dependent terms scale linearly in `alpha` (and quadratically in the
    dissimilarity), which is why `metrics` holds the target-vs-current
    quantities and `alpha` enters explicitly here.

    Args:
        metrics: Exact bound quantities of the current pair.
        alpha: Policy-mixing coefficient in `[0, 1]`.
        tau: Current teleport rate.
        tau_prime: Candidate teleport rate in `[0, tau]`.
        gamma: Discount factor in `(0, 1)`.

    Returns:
        The Teleport Bound value.
    """
    delta_tau = tau - tau_prime
    advantage = (alpha * metrics.pol_adv + delta_tau * metrics.model_adv) / (1.0 - gamma)
    dissimilarity = (
        alpha**2 * metrics.d_exp_pol * metrics.d_inf_pol
        + alpha * abs(delta_tau) * metrics.d_exp_pol * metrics.d_inf_model
        + alpha * abs(delta_tau) * metrics.d_exp_model * metrics.d_inf_pol
        + gamma * delta_tau**2 * metrics.d_exp_model * metrics.d_inf_model
    )
    penalty = gamma * metrics.delta_u * dissimilarity / (2.0 * (1.0 - gamma) ** 2)
    return advantage - penalty


def compute_alpha_tau(metrics: BoundMetrics, gamma: float) -> float:
    """Bound-optimal `alpha` when `tau' = tau` (Table 1, top-left)."""
    return (
        (1.0 - gamma)
        * metrics.pol_adv
        / (gamma * metrics.delta_u * metrics.d_exp_pol * metrics.d_inf_pol)
    )


def compute_alpha_0(metrics: BoundMetrics, tau: float, gamma: float) -> float:
    """Bound-optimal `alpha` when `tau' = 0` (Table 1, bottom-left)."""
    dissimilarity = (
        tau
        / 2.0
        * (metrics.d_exp_model / metrics.d_exp_pol + metrics.d_inf_model / metrics.d_inf_pol)
    )
    return compute_alpha_tau(metrics, gamma) - dissimilarity


def compute_tau_prime_0(metrics: BoundMetrics, tau: float, gamma: float) -> float:
    """Bound-optimal `tau'` when `alpha = 0` (Table 1, top-right)."""
    return tau - (1.0 - gamma) * metrics.model_adv / (
        gamma**2 * metrics.delta_u * metrics.d_exp_model * metrics.d_inf_model
    )


def compute_tau_prime_1(metrics: BoundMetrics, tau: float, gamma: float) -> float:
    """Bound-optimal `tau'` when `alpha = 1` (Table 1, bottom-right)."""
    dissimilarity = (
        1.0
        / (2.0 * gamma)
        * (metrics.d_exp_pol / metrics.d_exp_model + metrics.d_inf_pol / metrics.d_inf_model)
    )
    return compute_tau_prime_0(metrics, tau, gamma) + dissimilarity


def _clip(value: float, low: float, high: float) -> float:
    """Clip a scalar to `[low, high]`."""
    return max(low, min(high, value))


def candidate_pairs(metrics: BoundMetrics, tau: float, gamma: float) -> list[FloatCandidate]:
    """Build the boundary candidate set `V` (Thm 5.1), clipped to feasibility.

    Always includes the no-op `(0, tau)` (bound 0), so the selected pair never
    has a negative bound. Policy candidates are added only when the policy
    advantage is positive and its dissimilarities are non-degenerate; model
    candidates likewise gate on the model advantage. `alpha` is clipped to
    `[0, 1]` and `tau'` to `[0, tau]` (monotone non-increasing teleport rate).

    Args:
        metrics: Exact bound quantities of the current pair.
        tau: Current teleport rate.
        gamma: Discount factor in `(0, 1)`.

    Returns:
        The list of candidate `(alpha, tau')` pairs.
    """
    pairs: list[FloatCandidate] = [(0.0, tau)]

    if metrics.pol_adv > 0.0 and metrics.d_inf_pol > 0.0 and metrics.d_exp_pol > 0.0:
        pairs.append((_clip(compute_alpha_tau(metrics, gamma), 0.0, 1.0), tau))
        pairs.append((_clip(compute_alpha_0(metrics, tau, gamma), 0.0, 1.0), 0.0))

    if metrics.model_adv > 0.0 and metrics.d_inf_model > 0.0 and metrics.d_exp_model > 0.0:
        pairs.append((0.0, _clip(compute_tau_prime_0(metrics, tau, gamma), 0.0, tau)))
        pairs.append((1.0, _clip(compute_tau_prime_1(metrics, tau, gamma), 0.0, tau)))

    return pairs


def optimal_pair(
    metrics: BoundMetrics, tau: float, gamma: float, *, tau_threshold: float = 1e-6
) -> tuple[float, float, float]:
    """Select the candidate maximizing the Teleport Bound.

    Args:
        metrics: Exact bound quantities of the current pair.
        tau: Current teleport rate.
        gamma: Discount factor in `(0, 1)`.
        tau_threshold: `tau'` below this is snapped to exactly 0.

    Returns:
        The winning `(alpha_star, tau_star, bound_star)`.
    """
    pairs = candidate_pairs(metrics, tau, gamma)
    bounds = [
        compute_teleport_bound(metrics, alpha, tau, tau_prime, gamma) for alpha, tau_prime in pairs
    ]
    best = max(range(len(pairs)), key=lambda i: bounds[i])
    alpha_star, tau_star = pairs[best]
    if tau_star < tau_threshold:
        tau_star = 0.0
    return alpha_star, tau_star, bounds[best]
