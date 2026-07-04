from typing import NamedTuple, Protocol

import numpy as np
from tqdm.auto import tqdm

from teleport_mdp.curriculum.scheduler import TeleportScheduler
from teleport_mdp.models import QLearningConfig
from teleport_mdp.tabular.model_functions import FloatArray, get_d_inf_policy, get_policy
from teleport_mdp.wrappers.tmdp import TMDP


class MetricLogger(Protocol):
    """Minimal metric sink: anything exposing MLflow's `log_metrics` signature.

    :class:`utils.experiment_logger.MlflowLogger` satisfies this (it forwards
    `log_metrics` to `mlflow`), so no SB3 machinery is needed for the tabular
    path.
    """

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log a batch of scalar metrics at a given step."""
        ...


class QLearningResult(NamedTuple):
    """Outcome of a Q-learning run, with snapshots for learning-curve analysis."""

    q: FloatArray
    q_snapshots: list[FloatArray]
    returns: list[float]
    visit_distribution: FloatArray
    disc_visit_distribution: FloatArray
    tau_history: list[float]


class QLearner:
    """ε-greedy tabular Q-learning with teleport truncation and a shared curriculum.

    Args:
        config: Tabular Q-learning hyperparameters.
        gamma: Discount factor in `(0, 1)`, held at the experiment level.
        max_steps_per_episode: Hard cap on steps per episode; teleports keep an
            episode running (no reset), so the cap bounds long teleport-heavy
            episodes. Defaults to `1000`.
        seed: Seed for the ε-greedy action RNG (independent of the env RNG).
    """

    def __init__(
        self,
        config: QLearningConfig,
        gamma: float,
        *,
        max_steps_per_episode: int = 1000,
        seed: int | None = None,
    ) -> None:
        self._config = config
        self._gamma = gamma
        self._max_steps = max_steps_per_episode
        self._rng = np.random.default_rng(seed)

    def train(
        self,
        env: TMDP,
        scheduler: TeleportScheduler | None = None,
        logger: MetricLogger | None = None,
        *,
        progress: bool = False,
        desc: str = "Q-learning",
    ) -> QLearningResult:
        """Run Q-learning on a teleport-wrapped env, optionally annealing `tau`.

        Args:
            env: The teleport-MDP-wrapped environment to learn on.
            scheduler: Teleport-rate scheduler invoked once per status step; when
                `None` the teleport rate is left fixed (no curriculum).
            logger: Optional metric sink; receives `train/return` and
                `curriculum/tau` at each status step.
            progress: If `True`, show a tqdm progress bar over episodes with a
                live return/tau readout.
            desc: Label for the progress bar.

        Returns:
            The :class:`QLearningResult` with the final Q, snapshots, and analytics.
        """
        n_states = int(env.observation_space.n)  # type: ignore[attr-defined]
        n_actions = int(env.action_space.n)  # type: ignore[attr-defined]
        q = np.zeros((n_states, n_actions), dtype=np.float64)

        visits = np.zeros(n_states, dtype=np.float64)
        disc_visits = np.zeros(n_states, dtype=np.float64)

        q_snapshots: list[FloatArray] = []
        returns: list[float] = []
        tau_history: list[float] = []
        prev_policy = get_policy(q, deterministic=True)

        alpha_0 = self._config.alpha
        eps_0 = self._config.eps if self._config.eps > 0.0 else min(1.0, 2.0 * alpha_0)
        episodes = self._config.episodes

        bar = tqdm(range(episodes), desc=desc, unit="ep", disable=not progress, dynamic_ncols=True)
        for episode in bar:
            frac = episode / episodes
            alpha = alpha_0 * (1.0 - frac)
            eps = eps_0 * (1.0 - frac) ** 2
            ep_return = self._run_episode(env, q, visits, disc_visits, alpha, eps)
            returns.append(ep_return)

            if episode % self._config.status_step == 0:
                q_snapshots.append(q.copy())
                tau = self._current_tau(env)
                tau_history.append(tau)
                window = returns[-self._config.status_step :]
                mean_return = float(np.mean(window))
                if logger is not None:
                    logger.log_metrics(
                        {"train/return": mean_return, "curriculum/tau": tau},
                        step=episode,
                    )
                if progress:
                    bar.set_postfix(ret=f"{mean_return:.3f}", tau=f"{tau:.3f}", refresh=False)
                if scheduler is not None:
                    prev_policy = self._step_curriculum(env, q, prev_policy, scheduler)

        bar.close()
        q_snapshots.append(q.copy())
        return QLearningResult(
            q=q,
            q_snapshots=q_snapshots,
            returns=returns,
            visit_distribution=self._normalize(visits),
            disc_visit_distribution=self._normalize(disc_visits),
            tau_history=tau_history,
        )

    def _run_episode(
        self,
        env: TMDP,
        q: FloatArray,
        visits: FloatArray,
        disc_visits: FloatArray,
        alpha: float,
        eps: float,
    ) -> float:
        """Run one episode, applying truncated TD updates; returns the env return."""
        env.reset()
        ep_return = 0.0
        disc_step = 0
        for _ in range(self._max_steps):
            s = int(env.unwrapped.s)  # type: ignore[attr-defined]
            a = self._eps_greedy(q, s, eps)
            s_prime, reward, terminated, truncated, info = env.step(a)

            if info["teleport"]:
                # The action was not executed: no (s, a, r, s') to learn from, and
                # the trajectory is truncated, so reset the discount counter.
                disc_step = 0
                continue

            ep_return += reward
            target = reward if terminated else reward + self._gamma * float(np.max(q[s_prime]))
            q[s, a] += alpha * (target - q[s, a])

            visits[s] += 1.0
            disc_visits[s] += self._gamma**disc_step
            disc_step += 1

            if terminated or truncated:
                break
        return ep_return

    def _eps_greedy(self, q: FloatArray, state: int, eps: float) -> int:
        """ε-greedy action with random tie-breaking among the greedy maximizers."""
        if self._rng.random() < eps:
            return int(self._rng.integers(q.shape[1]))
        row = q[state]
        best = np.flatnonzero(row == row.max())
        return int(self._rng.choice(best))

    def _step_curriculum(
        self,
        env: TMDP,
        q: FloatArray,
        prev_policy: FloatArray,
        scheduler: TeleportScheduler,
    ) -> FloatArray:
        """Advance the teleport rate via the scheduler; returns the new greedy policy."""
        current_policy = get_policy(q, deterministic=True)
        policy_shift = get_d_inf_policy(prev_policy, current_policy)
        next_tau = scheduler.next_tau(self._current_tau(env), policy_shift=policy_shift)
        env.update_tau(next_tau)
        return current_policy

    @staticmethod
    def _current_tau(env: TMDP) -> float:
        """Read the env's current teleport rate."""
        return float(env.teleport_probability)

    @staticmethod
    def _normalize(counts: FloatArray) -> FloatArray:
        """Normalize a visit-count vector to a distribution (zeros if no visits)."""
        total = float(counts.sum())
        if total <= 0.0:
            return counts
        return counts / total
