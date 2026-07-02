from typing import NamedTuple, Protocol

import numpy as np

from teleport_mdp.models import TMPIConfig
from teleport_mdp.tabular.bound import BoundMetrics, optimal_pair
from teleport_mdp.tabular.model_functions import (
    FloatArray,
    compute_d,
    compute_delta,
    compute_expected_model_advantage,
    compute_expected_policy_advantage,
    compute_policy_advantage,
    compute_q_from_u,
    compute_relative_model_advantage,
    compute_relative_policy_advantage,
    compute_u_from_v,
    compute_value_function,
    get_d_exp_model,
    get_d_exp_policy,
    get_d_inf_policy,
    get_sup_difference,
    softmax_policy,
)
from teleport_mdp.wrappers.tmdp import TMDP


class MetricLogger(Protocol):
    """Minimal metric sink matching MLflow's `log_metrics` signature."""

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log a batch of scalar metrics at a given step."""
        ...


class TMPIResult(NamedTuple):
    """Outcome of a TMPI run."""

    pi: FloatArray
    v: FloatArray
    performance: float
    performance_history: list[float]
    bound_history: list[float]
    alpha_history: list[float]
    tau_history: list[float]
    iterations: int


class TMPI:
    """Exact model-based Teleport Model Policy Iteration.

    Args:
        config: TMPI hyperparameters (convergence threshold, iteration cap,
            softmax temperature of the target policy).
        gamma: Discount factor in `(0, 1)`, held at the experiment level.
    """

    def __init__(self, config: TMPIConfig, gamma: float) -> None:
        self._config = config
        self._gamma = gamma

    def optimize(self, tmdp: TMDP, logger: MetricLogger | None = None) -> TMPIResult:
        """Run TMPI on a tabular teleport MDP until convergence.

        Args:
            tmdp: The teleport-wrapped tabular env; its known dynamics are read
                directly (no sampling). Its teleport rate is updated in place to
                track `tau` across iterations.
            logger: Optional metric sink; receives `tmpi/performance`,
                `tmpi/bound`, `tmpi/alpha`, and `curriculum/tau` per iteration.

        Returns:
            The :class:`TMPIResult` with the final policy, value, and histories.
        """
        p, reward, mu, non_terminal = _dense_model(tmdp)
        xi = np.asarray(tmdp.teleport_prob_distribution, dtype=np.float64)
        gamma = self._gamma

        n_states, n_actions = p.shape[0], p.shape[1]
        pi = np.full((n_states, n_actions), 1.0 / n_actions, dtype=np.float64)
        tau = float(tmdp.teleport_probability)

        performance_history: list[float] = []
        bound_history: list[float] = []
        alpha_history: list[float] = []
        tau_history: list[float] = []

        iterations = 0
        for iteration in range(self._config.max_iterations):
            p_tau = _teleport_kernel(p, xi, tau, non_terminal)
            v = compute_value_function(p_tau, reward, pi, gamma)
            performance = float(mu @ v)

            metrics, target_pi = self._bound_metrics(p, xi, reward, mu, pi, tau, v, non_terminal)
            alpha_star, tau_star, bound_star = optimal_pair(
                metrics, tau, gamma, tau_threshold=self._config.threshold
            )

            performance_history.append(performance)
            bound_history.append(bound_star)
            alpha_history.append(alpha_star)
            tau_history.append(tau)
            if logger is not None:
                logger.log_metrics(
                    {
                        "tmpi/performance": performance,
                        "tmpi/bound": bound_star,
                        "tmpi/alpha": alpha_star,
                        "curriculum/tau": tau,
                    },
                    step=iteration,
                )

            iterations = iteration + 1
            if bound_star <= self._config.threshold:
                break

            pi = alpha_star * target_pi + (1.0 - alpha_star) * pi
            tau = tau_star
            tmdp.update_tau(tau)

        v = compute_value_function(_teleport_kernel(p, xi, tau, non_terminal), reward, pi, gamma)
        return TMPIResult(
            pi=pi,
            v=v,
            performance=float(mu @ v),
            performance_history=performance_history,
            bound_history=bound_history,
            alpha_history=alpha_history,
            tau_history=tau_history,
            iterations=iterations,
        )

    def _bound_metrics(
        self,
        p: FloatArray,
        xi: FloatArray,
        reward: FloatArray,
        mu: FloatArray,
        pi: FloatArray,
        tau: float,
        v: FloatArray,
        non_terminal: FloatArray,
    ) -> tuple[BoundMetrics, FloatArray]:
        """Compute the exact Teleport-Bound metrics and the softmax target policy.

        The target policy `pi_bar` is `softmax(Q / temperature)`; the metrics are
        the target-vs-current policy quantities and the `P`-vs-`xi` model
        quantities, all evaluated under the current pair's visit distribution.
        The model-dissimilarity terms are restricted to non-terminal states,
        since teleport (hence the model change from `tau` to `tau'`) never fires
        from the absorbing terminal states.
        """
        gamma = self._gamma
        p_tau = _teleport_kernel(p, xi, tau, non_terminal)
        u = compute_u_from_v(reward, gamma, v)
        q = compute_q_from_u(p_tau, u)
        target_pi = softmax_policy(q, temperature=self._config.temperature)

        d = compute_d(mu, p_tau, pi, gamma)
        delta = compute_delta(d, pi)
        # Teleport does not perturb terminal-state dynamics, so drop their mass
        # from the model advantage and the model dissimilarity expectation.
        model_delta = delta * non_terminal[:, None]

        policy_adv = compute_policy_advantage(q, v)
        pol_adv = compute_expected_policy_advantage(
            compute_relative_policy_advantage(target_pi, policy_adv), d
        )
        model_adv = compute_expected_model_advantage(
            compute_relative_model_advantage(p, xi, u), model_delta
        )

        # A constant U (e.g. an all-zero-reward region) yields delta_U = 0, which
        # would divide by zero in the candidate formulas; fall back to the finite
        # horizon range as the legacy code does.
        delta_u = get_sup_difference(u)
        if delta_u <= 0.0:
            delta_u = (1.0 - gamma**10) / (1.0 - gamma)

        return (
            BoundMetrics(
                pol_adv=pol_adv,
                model_adv=model_adv,
                delta_u=delta_u,
                d_inf_pol=get_d_inf_policy(pi, target_pi),
                d_exp_pol=get_d_exp_policy(pi, target_pi, d),
                d_inf_model=_d_inf_model_non_terminal(p, xi, non_terminal),
                d_exp_model=get_d_exp_model(p, xi, model_delta),
            ),
            target_pi,
        )


def _teleport_kernel(
    p: FloatArray, xi: FloatArray, tau: float, non_terminal: FloatArray
) -> FloatArray:
    """Teleport transition tensor with teleport applied only on non-terminal states.

    `P_tau(.|s, a) = (1 - tau) P + tau xi` for non-terminal `s`, and the base
    absorbing `P` for terminal `s` (the episode ends there, so no teleport). This
    keeps terminal states absorbing and prevents the exact model from teleporting
    the agent out of the goal to re-collect reward.
    """
    mask = non_terminal[:, None, None]
    return np.where(mask.astype(bool), (1.0 - tau) * p + tau * xi[None, None, :], p)


def _d_inf_model_non_terminal(p: FloatArray, xi: FloatArray, non_terminal: FloatArray) -> float:
    """Sup over non-terminal `(s, a)` of the L1 distance `||P(.|s, a) - xi||_1`."""
    l1_norm = np.sum(np.abs(p - xi[None, None, :]), axis=2)
    return float(l1_norm[non_terminal.astype(bool)].max())


def _dense_model(tmdp: TMDP) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Extract dense `(P[nS,nA,nS], R[nS,nA,nS], mu[nS], non_terminal[nS])`.

    `non_terminal` is a `{0, 1}` float mask flagging the non-terminal states
    (those from which teleport can fire).
    """
    env = tmdp.env
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    p = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    reward = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for s in range(n_states):
        for a in range(n_actions):
            for transition in env.P[s][a]:
                p[s, a, transition.next_state] += transition.probability
                reward[s, a, transition.next_state] = transition.reward
    mu = np.asarray(env.initial_state_distrib, dtype=np.float64)
    non_terminal = np.array(
        [0.0 if env.is_terminal(s) else 1.0 for s in range(n_states)], dtype=np.float64
    )
    return p, reward, mu, non_terminal
