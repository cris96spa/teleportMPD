from abc import ABC, abstractmethod
from math import ceil


class TeleportScheduler(ABC):
    """Maps the current teleport rate to the next one for a curriculum update.

    A scheduler is called once per PPO update with the current `tau` (and, for
    the dynamic variant, the policy shift of that update) and returns the next
    `tau`. Implementations must keep `tau` in `[0, tau_0]` and monotone
    non-increasing.
    """

    @abstractmethod
    def next_tau(self, tau: float, *, policy_shift: float | None = None) -> float:
        """Return the next teleport rate for the upcoming update.

        Args:
            tau: The current teleport rate.
            policy_shift: The policy shift `D_inf = max_s ||pi'(.|s) - pi(.|s)||_1`
                of the just-completed update. Required by the dynamic scheduler;
                ignored by the static one.

        Returns:
            The next teleport rate, in `[0, tau]`.
        """
        raise NotImplementedError

    @staticmethod
    def compute_eps_model(gamma: float, tau: float, n: int) -> float:
        """Per-update model-shift budget that drives `tau` to 0 in `n` updates.

        Implements `eps_model = 2*gamma*tau / (n*(1 - gamma))` (thesis Theorem 5.2).

        Args:
            gamma: Discount factor in `(0, 1)`.
            tau: Teleport rate to be annealed away (typically `tau_0`).
            n: Number of updates over which `tau` should reach 0; must be positive.

        Returns:
            The per-update model-shift budget `eps_model`.

        Raises:
            ValueError: If `n` is not positive.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}.")
        return 2.0 * gamma * tau / (n * (1.0 - gamma))

    @staticmethod
    def compute_tau_prime(gamma: float, tau: float, eps_model: float) -> float:
        """Apply a model-shift budget as a teleport-rate decrement, clamped at 0.

        Implements `tau' = max(0, tau - eps_model*(1 - gamma)/(2*gamma))`, the inverse
        of :meth:`compute_eps_model` for a single step.

        Args:
            gamma: Discount factor in `(0, 1)`.
            tau: Current teleport rate.
            eps_model: Model-shift budget to spend on lowering `tau` this update.

        Returns:
            The next teleport rate `tau'`, never below 0.
        """
        tau_prime = tau - eps_model * (1.0 - gamma) / (2.0 * gamma)
        return max(0.0, tau_prime)

    @staticmethod
    def compute_n_updates(gamma: float, tau: float, eps_model: float) -> int:
        """Number of updates needed to anneal `tau` to 0 at a given per-update budget.

        Implements `n = ceil(2*gamma*tau / ((1 - gamma)*eps_model))` (thesis Theorem 5.2),
        the inverse of :meth:`compute_eps_model`.

        Args:
            gamma: Discount factor in `(0, 1)`.
            tau: Teleport rate to be annealed away.
            eps_model: Per-update model-shift budget; must be positive.

        Returns:
            The number of updates to reach `tau == 0`.

        Raises:
            ValueError: If `eps_model` is not positive.
        """
        if eps_model <= 0.0:
            raise ValueError(f"eps_model must be positive, got {eps_model}.")
        return ceil(2.0 * gamma * tau / ((1.0 - gamma) * eps_model))


class StaticTeleportScheduler(TeleportScheduler):
    """Linear teleport anneal to 0 over a fixed number of updates (Algorithm 2).

    Precomputes the per-update model-shift budget `eps_tau` from the initial rate
    and the update budget so that, starting at `tau_0`, `tau` decreases by a
    constant `tau_0 / n_updates` each call and reaches exactly 0 at update
    `n_updates`. The policy shift is ignored.

    Args:
        gamma: Discount factor in `(0, 1)`.
        tau_0: Initial teleport rate to anneal away; must be positive.
        n_updates: Number of updates over which `tau` reaches 0; must be positive.

    Raises:
        ValueError: If `tau_0` is not positive.
    """

    def __init__(self, gamma: float, tau_0: float, n_updates: int) -> None:
        if tau_0 <= 0.0:
            raise ValueError(f"tau_0 must be positive for a curriculum, got {tau_0}.")
        self._gamma = gamma
        self._tau_0 = tau_0
        self._n_updates = n_updates
        self._eps_tau = self.compute_eps_model(gamma, tau_0, n_updates)

    @property
    def eps_tau(self) -> float:
        """The precomputed per-update model-shift budget."""
        return self._eps_tau

    def next_tau(self, tau: float, *, policy_shift: float | None = None) -> float:
        """Decrement `tau` by the fixed budget, clamped at 0 (`policy_shift` ignored)."""
        return self.compute_tau_prime(self._gamma, tau, self._eps_tau)


class DynamicTeleportScheduler(TeleportScheduler):
    """Policy-shift-aware teleport anneal (Algorithm 3).

    Each update has a fixed total state-visit shift budget `eps`. The policy
    update itself consumes `eps_pi = gamma/(1 - gamma) * D_inf` of it; whatever
    remains, `eps_tau = eps - eps_pi` (capped per update at `eps_tau_max`), is
    spent lowering `tau`. So a large policy shift pauses the teleport anneal and a
    small one lets it proceed.

    Canonical-form note (resolves a legacy/thesis discrepancy): the thesis puts the
    `gamma/(1 - gamma)` factor *inside* `eps_pi` (i.e. it multiplies only the
    policy shift `D_inf`). Legacy `TeleportPPO.update_teleport_rate` instead
    applied that factor to the whole net budget `eps - D_inf` via
    `gamma_eps_model = (eps_shift - d_inf)*gamma/(1 - gamma)`. This implementation
    follows the thesis form, which is the canonical one.

    Args:
        gamma: Discount factor in `(0, 1)`.
        eps: Total per-update state-visit shift budget; must be positive.
        eps_tau_max: Maximum model shift spent on `tau` in a single update;
            must be positive.

    Raises:
        ValueError: If `eps` or `eps_tau_max` is not positive.
    """

    def __init__(self, gamma: float, eps: float, eps_tau_max: float) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if eps_tau_max <= 0.0:
            raise ValueError(f"eps_tau_max must be positive, got {eps_tau_max}.")
        self._gamma = gamma
        self._eps = eps
        self._eps_tau_max = eps_tau_max

    def next_tau(self, tau: float, *, policy_shift: float | None = None) -> float:
        """Lower `tau` by the budget left after the policy shift (Algorithm 3).

        Args:
            tau: The current teleport rate.
            policy_shift: The update's policy shift `D_inf`; required.

        Returns:
            The next teleport rate. `tau` is left unchanged when it is already 0
            or the policy consumed the whole shift budget (`eps_tau <= 0`).

        Raises:
            ValueError: If `policy_shift` is None.
        """
        if policy_shift is None:
            raise ValueError("DynamicTeleportScheduler requires a policy_shift.")
        if tau <= 0.0:
            return 0.0
        eps_pi = self._gamma / (1.0 - self._gamma) * policy_shift
        eps_tau = self._eps - eps_pi
        if eps_tau <= 0.0:
            return tau
        return self.compute_tau_prime(self._gamma, tau, min(eps_tau, self._eps_tau_max))
