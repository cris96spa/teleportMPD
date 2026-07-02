import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

FloatArray = NDArray[np.float64]


# region Model construction
def compute_p_tau(p: FloatArray, xi: FloatArray, tau: float) -> FloatArray:
    """Build the teleport transition probability matrix `P_tau = (1 - tau) P + tau * xi`.

    The teleport distribution `xi` over next states is broadcast across every
    `(s, a)` pair (thesis `P_tau(s'|s, a) = (1 - tau) P(s'|s, a) + tau xi(s')`).

    Args:
        p: Base transition tensor `[nS, nA, nS]`.
        xi: Teleport distribution over states `[nS]` (must sum to 1).
        tau: Teleport rate in `[0, 1]`.

    Returns:
        The teleport transition tensor `[nS, nA, nS]`.
    """
    return (1.0 - tau) * p + tau * xi[None, None, :]


def compute_r_s_a(p: FloatArray, reward: FloatArray) -> FloatArray:
    """Expected reward of taking action `a` in state `s`.

    Computes `R(s, a) = sum_s' P(s'|s, a) R(s, a, s')`.

    Args:
        p: Transition tensor `[nS, nA, nS]`.
        reward: Reward tensor `[nS, nA, nS]`.

    Returns:
        The expected reward matrix `[nS, nA]`.
    """
    return np.einsum("san,san->sa", p, reward)


def compute_transition_kernel(p: FloatArray, pi: FloatArray) -> FloatArray:
    """Marginalise a transition tensor over actions under a policy.

    Computes `P_pi(s'|s) = sum_a pi(a|s) P(s'|s, a)`.

    Args:
        p: Transition tensor `[nS, nA, nS]` (typically already a `P_tau`).
        pi: Policy `[nS, nA]`.

    Returns:
        The state-to-state transition kernel `[nS, nS]`.
    """
    return np.einsum("san,sa->sn", p, pi)


# endregion


# region Policies
def get_policy(q: FloatArray, deterministic: bool = True) -> FloatArray:
    """Derive a policy matrix from a state-action value function.

    Args:
        q: State-action value function `[nS, nA]`.
        deterministic: If `True`, return a greedy one-hot policy (mass on the
            first `argmax` action per state); otherwise return a softmax policy.

    Returns:
        The policy `[nS, nA]` with rows summing to 1.
    """
    if deterministic:
        pi = np.zeros_like(q)
        pi[np.arange(q.shape[0]), np.argmax(q, axis=1)] = 1.0
        return pi
    return softmax_policy(q)


def softmax_policy(x: FloatArray, temperature: float = 1.0) -> FloatArray:
    """Build a softmax policy from per-action scores.

    Computes `pi(a|s) = softmax(x(s, .) / temperature)` row-wise.

    Args:
        x: Per-action scores `[nS, nA]` (e.g. a `Q` matrix).
        temperature: Softmax temperature; lower values sharpen the policy.

    Returns:
        The softmax policy `[nS, nA]` with rows summing to 1.

    Raises:
        ValueError: If `temperature` is not strictly positive.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be strictly positive, got {temperature}.")
    return softmax(x / temperature, axis=1)


# endregion


# region Value functions
def compute_v_from_q(q: FloatArray, pi: FloatArray) -> FloatArray:
    """Value function from a state-action value function under a policy.

    Computes `V(s) = sum_a pi(a|s) Q(s, a)`.

    Args:
        q: State-action value function `[nS, nA]`.
        pi: Policy `[nS, nA]`.

    Returns:
        The value function `[nS]`.
    """
    return np.sum(q * pi, axis=1)


def compute_u_from_v(reward: FloatArray, gamma: float, v: FloatArray) -> FloatArray:
    """State-action-nextstate value function from a value function (Eq. 4).

    Computes `U(s, a, s') = R(s, a, s') + gamma V(s')` (thesis Eq. 4). The reward
    is left unscaled: the teleport's effective discount is realised through
    trajectory truncation, not reward scaling.

    Args:
        reward: Reward tensor `[nS, nA, nS]`.
        gamma: Discount factor in `(0, 1)`.
        v: Value function `[nS]`.

    Returns:
        The state-action-nextstate value tensor `[nS, nA, nS]`.
    """
    return reward + gamma * v.reshape(1, 1, -1)


def compute_q_from_u(p_tau: FloatArray, u: FloatArray) -> FloatArray:
    """State-action value function from a state-action-nextstate value function.

    Computes `Q(s, a) = sum_s' P_tau(s'|s, a) U(s, a, s')`.

    Args:
        p_tau: Teleport transition tensor `[nS, nA, nS]`.
        u: State-action-nextstate value tensor `[nS, nA, nS]`.

    Returns:
        The state-action value function `[nS, nA]`.
    """
    return np.einsum("san,san->sa", p_tau, u)


# endregion


# region Advantage functions
def compute_policy_advantage(q: FloatArray, v: FloatArray) -> FloatArray:
    """Policy advantage `A(s, a) = Q(s, a) - V(s)`.

    Args:
        q: State-action value function `[nS, nA]`.
        v: Value function `[nS]`.

    Returns:
        The policy advantage `[nS, nA]`.
    """
    return q - v[:, None]


def compute_model_advantage(u: FloatArray, q: FloatArray) -> FloatArray:
    """Model advantage `A(s, a, s') = U(s, a, s') - Q(s, a)`.

    Args:
        u: State-action-nextstate value tensor `[nS, nA, nS]`.
        q: State-action value function `[nS, nA]`.

    Returns:
        The model advantage `[nS, nA, nS]`.
    """
    return u - q[:, :, None]


def compute_relative_policy_advantage(pi_prime: FloatArray, a: FloatArray) -> FloatArray:
    """Relative policy advantage of `pi_prime` against the advantage `A`.

    Computes `A^{pi'}_pi(s) = sum_a pi'(a|s) A(s, a)`.

    Args:
        pi_prime: Candidate policy `[nS, nA]`.
        a: Policy advantage of the current policy `[nS, nA]`.

    Returns:
        The relative policy advantage per state `[nS]`.
    """
    return np.einsum("sa,sa->s", pi_prime, a)


def compute_relative_model_advantage(p: FloatArray, xi: FloatArray, u: FloatArray) -> FloatArray:
    """Relative model advantage of the teleport distribution against `P`.

    Computes `A_{P,xi}(s, a) = sum_s' (P(s'|s, a) - xi(s')) U(s, a, s')`.

    Args:
        p: Base transition tensor `[nS, nA, nS]`.
        xi: Teleport distribution over states `[nS]`.
        u: State-action-nextstate value tensor `[nS, nA, nS]`.

    Returns:
        The relative model advantage `[nS, nA]`.
    """
    return np.einsum("san,san->sa", p - xi[None, None, :], u)


def compute_expected_policy_advantage(rel_policy_adv: FloatArray, d: FloatArray) -> float:
    """Expected relative policy advantage under the state-visit distribution.

    Computes `sum_s d(s) A^{pi'}_pi(s)`.

    Args:
        rel_policy_adv: Relative policy advantage per state `[nS]`.
        d: Gamma-discounted state-visit distribution `[nS]`.

    Returns:
        The expected relative policy advantage as a scalar.
    """
    return float(d @ rel_policy_adv)


def compute_expected_model_advantage(rel_model_adv: FloatArray, delta: FloatArray) -> float:
    """Expected relative model advantage under the state-action-visit distribution.

    Computes `sum_{s,a} delta(s, a) A_{P,xi}(s, a)`.

    Args:
        rel_model_adv: Relative model advantage `[nS, nA]`.
        delta: Gamma-discounted state-action-visit distribution `[nS, nA]`.

    Returns:
        The expected relative model advantage as a scalar.
    """
    return float(np.sum(delta * rel_model_adv))


# endregion


# region State-visit distributions
def compute_d(mu: FloatArray, p_tau: FloatArray, pi: FloatArray, gamma: float) -> FloatArray:
    """Gamma-discounted state-visit distribution `d` (thesis Eq. 2).

    Solves `d = (1 - gamma) mu (I - gamma P_pi)^{-1}` where
    `P_pi(s'|s) = sum_a pi(a|s) P_tau(s'|s, a)`. The teleport is assumed already
    folded into `p_tau` (use :func:`compute_p_tau`), so this matches the thesis
    closed form `(1 - gamma) mu (I - gamma((1-tau)P_pi + tau Xi))^{-1}`.

    Args:
        mu: Initial state distribution `[nS]` (must sum to 1).
        p_tau: Teleport transition tensor `[nS, nA, nS]`.
        pi: Policy `[nS, nA]`.
        gamma: Discount factor in `(0, 1)`.

    Returns:
        The gamma-discounted state-visit distribution `[nS]` (sums to 1).
    """
    n_states = mu.shape[0]
    p_pi = compute_transition_kernel(p_tau, pi)
    # d is a left eigen-style solve: d (I - gamma P_pi) = (1 - gamma) mu, so we
    # solve the transposed system (I - gamma P_pi)^T x = mu and scale by (1-gamma).
    a = np.eye(n_states) - gamma * p_pi
    return (1.0 - gamma) * np.linalg.solve(a.T, mu)


def compute_delta(d: FloatArray, pi: FloatArray) -> FloatArray:
    """Gamma-discounted state-action-visit distribution `delta = pi * d`.

    Computes `delta(s, a) = pi(a|s) d(s)`.

    Args:
        d: Gamma-discounted state-visit distribution `[nS]`.
        pi: Policy `[nS, nA]`.

    Returns:
        The gamma-discounted state-action-visit distribution `[nS, nA]`.

    Raises:
        ValueError: If the resulting distribution has negative mass.
    """
    delta = pi * d[:, None]
    if np.any(delta < 0.0):
        raise ValueError("State-action visit distribution contains negative values.")
    return delta


# endregion


# region Dissimilarity terms (Teleport Bound, Thm 4.2)
def get_sup_difference(value_function: FloatArray) -> float:
    """Range (sup minus inf) of a value function, the `delta_U` bound term.

    Args:
        value_function: Any value array (e.g. `U` or `Q`).

    Returns:
        `max(value_function) - min(value_function)` as a scalar.
    """
    return float(np.max(value_function) - np.min(value_function))


def get_d_inf_policy(pi: FloatArray, pi_prime: FloatArray) -> float:
    """Sup over states of the L1 distance between two policies (`D_inf` policy).

    Computes `sup_s ||pi(.|s) - pi'(.|s)||_1`.

    Args:
        pi: A policy `[nS, nA]`.
        pi_prime: Another policy `[nS, nA]`.

    Returns:
        The supremum L1 policy distance as a scalar.
    """
    return float(np.max(np.sum(np.abs(pi - pi_prime), axis=1)))


def get_d_exp_policy(pi: FloatArray, pi_prime: FloatArray, d: FloatArray) -> float:
    """Expected L1 distance between two policies under `d` (`D_exp` policy).

    Computes `sum_s d(s) ||pi(.|s) - pi'(.|s)||_1`.

    Args:
        pi: A policy `[nS, nA]`.
        pi_prime: Another policy `[nS, nA]`.
        d: Gamma-discounted state-visit distribution `[nS]`.

    Returns:
        The expected L1 policy distance as a scalar.
    """
    return float(d @ np.sum(np.abs(pi - pi_prime), axis=1))


def get_d_inf_model(p: FloatArray, xi: FloatArray) -> float:
    """Sup over `(s, a)` of the L1 distance between `P` and `xi` (`D_inf` model).

    Computes `sup_{s,a} ||P(.|s, a) - xi||_1`.

    Args:
        p: Base transition tensor `[nS, nA, nS]`.
        xi: Teleport distribution over states `[nS]`.

    Returns:
        The supremum L1 model distance as a scalar.
    """
    return float(np.max(np.sum(np.abs(p - xi[None, None, :]), axis=2)))


def get_d_exp_model(p: FloatArray, xi: FloatArray, delta: FloatArray) -> float:
    """Expected L1 distance between `P` and `xi` under `delta` (`D_exp` model).

    Computes `sum_{s,a} delta(s, a) ||P(.|s, a) - xi||_1`.

    Args:
        p: Base transition tensor `[nS, nA, nS]`.
        xi: Teleport distribution over states `[nS]`.
        delta: Gamma-discounted state-action-visit distribution `[nS, nA]`.

    Returns:
        The expected L1 model distance as a scalar.
    """
    l1_norm = np.sum(np.abs(p - xi[None, None, :]), axis=2)
    return float(np.sum(delta * l1_norm))


# endregion
