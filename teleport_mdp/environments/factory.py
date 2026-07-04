from pathlib import Path
from typing import TYPE_CHECKING, cast

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, TimeLimit
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from teleport_mdp.constants import STOCHASTICITY_THRESHOLD
from teleport_mdp.enums import TeleportDistribution
from teleport_mdp.environments.register import ENV_ID
from teleport_mdp.environments.teleport_frozen_lake import TeleportFrozenLakeEnv
from teleport_mdp.models import EnvConfig, ExperimentConfig, TeleportConfig
from teleport_mdp.utils.frozen_lake import generate_random_map
from teleport_mdp.wrappers.tmdp import TMDP

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Default per-episode step cap applied by `make_vec_env`
DEFAULT_MAX_EPISODE_STEPS = 200


def make_env(cfg: EnvConfig) -> TeleportFrozenLakeEnv:
    """Build a single, unwrapped FrozenLake env from an :class:`EnvConfig`.

    Map resolution priority: explicit `desc` > built-in `map_name` > a freshly
    generated random map (`size`, `p`, `seed`). The env is created through the
    gym registry and returned unwrapped so it can be fed to :class:`TMDP`.

    Args:
        cfg: The environment configuration.

    Returns:
        The unwrapped :class:`TeleportFrozenLakeEnv`.
    """
    if cfg.desc is not None:
        desc: list[str] | None = cfg.desc
        map_name: str | None = None
    elif cfg.map_name is not None:
        desc = None
        map_name = cfg.map_name
    else:
        desc = generate_random_map(size=cfg.size, p=cfg.p, seed=cfg.seed)
        map_name = None

    env = gym.make(
        ENV_ID,
        render_mode=cfg.render_mode,
        desc=desc,
        map_name=map_name,
        is_slippery=cfg.is_slippery,
    )
    return cast(TeleportFrozenLakeEnv, env.unwrapped)


def _load_custom_xi(path: Path, n_states: int) -> "NDArray[np.float64]":
    """Load a custom (unnormalized) teleport distribution from disk.

    Args:
        path: Path to a `.npy` array or a delimited text file of length `n_states`.
        n_states: The expected number of states.

    Returns:
        The loaded non-negative weights (normalization happens in :func:`build_xi`).

    Raises:
        ValueError: If the loaded array has the wrong shape or holds negative weights.
    """
    raw = np.load(path) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
    xi = np.asarray(raw, dtype=np.float64).ravel()
    if xi.shape != (n_states,):
        raise ValueError(f"Custom xi has shape {xi.shape}, expected ({n_states},).")
    if np.any(xi < 0.0):
        raise ValueError("Custom xi must be non-negative.")
    return xi


def build_xi(env: TeleportFrozenLakeEnv, cfg: TeleportConfig) -> "NDArray[np.float64]":
    """Build the normalized teleport distribution xi over states.

    `UNIFORM` spreads mass over every state; `UNIFORM_NONTERMINAL` puts zero
    mass on terminal states (holes/goal) and renormalizes; `CUSTOM` loads weights
    from `cfg.custom_xi_path`. The result always sums to 1.

    Args:
        env: The (unwrapped) environment, used for its state count and terminality.
        cfg: The teleport configuration selecting the distribution.

    Returns:
        A 1-D float array of length `n_states` summing to 1.

    Raises:
        ValueError: If the distribution kind is unknown, a custom path is missing,
            the total mass is zero, or the result does not sum to 1.
    """
    n_states = int(env.observation_space.n)  # type: ignore[attr-defined]

    if cfg.distribution == TeleportDistribution.UNIFORM:
        xi = np.ones(n_states, dtype=np.float64)
    elif cfg.distribution == TeleportDistribution.UNIFORM_NONTERMINAL:
        xi = np.array(
            [0.0 if env.is_terminal(s) else 1.0 for s in range(n_states)],
            dtype=np.float64,
        )
    elif cfg.distribution == TeleportDistribution.CUSTOM:
        if cfg.custom_xi_path is None:
            raise ValueError("custom_xi_path must be set for a custom teleport distribution.")
        xi = _load_custom_xi(cfg.custom_xi_path, n_states)
    else:  # pragma: no cover - exhaustiveness guard
        raise ValueError(f"Unsupported teleport distribution: {cfg.distribution}.")

    total = float(xi.sum())
    if total <= 0.0:
        raise ValueError("The teleport distribution has zero total mass; cannot normalize.")
    xi /= total

    if abs(float(xi.sum()) - 1.0) > STOCHASTICITY_THRESHOLD:
        raise ValueError("The teleport distribution must sum to 1.")
    return xi


def wrap_tmdp(
    env: TeleportFrozenLakeEnv,
    xi: "NDArray[np.float64]",
    tau: float,
) -> TMDP:
    """Wrap an env in the teleport MDP at a given initial rate.

    Args:
        env: The unwrapped teleport environment.
        xi: The teleport distribution over states (must sum to 1).
        tau: The initial teleport rate, in `[0, 1)`.

    Returns:
        The :class:`TMDP`-wrapped environment.
    """
    return TMDP(
        env=env,
        teleport_prob_distribution=xi,
        teleport_probability=tau,
    )


def make_vec_env(
    cfg: ExperimentConfig,
    n_envs: int = 1,
    seed: int | None = None,
    max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
) -> VecEnv:
    """Build an SB3 `VecEnv` of one-hot FrozenLake envs from an experiment config.

    Each env is built as `TeleportFrozenLakeEnv -> [TMDP] -> TimeLimit ->
    FlattenObservation` (the Discrete observation is flattened to a one-hot `Box`
    so an SB3 `MlpPolicy` can consume it). The :class:`TMDP` layer is only added
    when `teleport.tau_0 > 0`, so a vanilla (`tau_0 == 0`) baseline is a plain
    FrozenLake.

    Args:
        cfg: The full experiment configuration.
        n_envs: Number of parallel environments.
        seed: Base RNG seed; falls back to `cfg.seed` when `None`.
        max_episode_steps: Per-episode step cap; `None` disables the time limit.

    Returns:
        The constructed Stable-Baselines3 :class:`VecEnv`.
    """
    resolved_seed = cfg.seed if seed is None else seed

    def _create_environment() -> gym.Env:
        """Build a single environment instance for SB3's vectorized wrapper."""
        base = make_env(cfg.env)
        env: gym.Env
        if cfg.teleport.tau_0 > 0.0:
            xi = build_xi(base, cfg.teleport)
            env = wrap_tmdp(base, xi, cfg.teleport.tau_0)
        else:
            env = base
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return FlattenObservation(env)

    return sb3_make_vec_env(_create_environment, n_envs=n_envs, seed=resolved_seed)
