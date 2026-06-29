"""Agent factory — returns the correct SB3 agent class for an experiment.

``AgentRegistry`` maps a :class:`~teleport_mdp.enums.Algorithm` to the builder that
assembles the agent. The on-policy factory only knows PPO; the tabular algorithms
(Q-learning, TMPI) have their own runners. Within the PPO family, the *class* is chosen
by the scheduler factory: no scheduler (``curriculum: none``) yields vanilla
:class:`~stable_baselines3.PPO`, otherwise :class:`~teleport_mdp.agents.TeleportPPO`.
"""

from collections.abc import Callable
from typing import Any, ClassVar, cast

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from teleport_mdp.agents import TeleportPPO
from teleport_mdp.enums import Algorithm
from teleport_mdp.models import ExperimentConfig, PPOConfig
from teleport_mdp.registries.base import ComponentRegistry
from teleport_mdp.registries.scheduler import build_scheduler
from teleport_mdp.utils.device import get_torch_device


class AgentRegistry(ComponentRegistry[BaseAlgorithm]):
    """Registry of training agents keyed by :class:`Algorithm`."""

    _registry: ClassVar[dict[str, Callable[..., BaseAlgorithm]]] = {}


def _ppo_kwargs(cfg: ExperimentConfig, env: VecEnv, seed: int) -> dict[str, Any]:
    """Assemble the shared SB3 PPO constructor keyword arguments.

    Args:
        cfg: The experiment configuration.
        env: The vectorized training environment.
        seed: RNG seed for the agent.

    Returns:
        The keyword arguments common to ``PPO`` and ``TeleportPPO``.
    """
    ppo = cast(PPOConfig, cfg.algorithm)
    return {
        "policy": "MlpPolicy",
        "env": env,
        "n_steps": ppo.n_steps,
        "batch_size": ppo.batch_size,
        "n_epochs": ppo.n_epochs,
        "gae_lambda": ppo.gae_lambda,
        "normalize_advantage": ppo.normalize_advantage,
        "learning_rate": ppo.learning_rate,
        "clip_range": ppo.clip_range,
        "clip_range_vf": ppo.clip_range_vf,
        "ent_coef": ppo.ent_coef,
        "vf_coef": ppo.vf_coef,
        "max_grad_norm": ppo.max_grad_norm,
        "target_kl": ppo.target_kl,
        "use_sde": ppo.use_sde,
        "sde_sample_freq": ppo.sde_sample_freq,
        "stats_window_size": ppo.stats_window_size,
        "gamma": cfg.gamma,
        "policy_kwargs": ppo.policy_kwargs,
        "seed": seed,
        "device": get_torch_device(),
        "verbose": 0,
    }


@AgentRegistry.register(Algorithm.PPO)
def _build_ppo(*, cfg: ExperimentConfig, env: VecEnv, seed: int) -> PPO:
    """Build vanilla ``PPO`` or curriculum ``TeleportPPO`` from the config.

    The scheduler factory decides the class: ``None`` (``curriculum: none``) gives
    vanilla PPO; any scheduler gives ``TeleportPPO`` wired to it.

    Args:
        cfg: The experiment configuration.
        env: The vectorized training environment.
        seed: RNG seed for the agent.

    Returns:
        The constructed PPO-family agent.
    """
    kwargs = _ppo_kwargs(cfg, env, seed)
    scheduler = build_scheduler(cfg)
    if scheduler is None:
        return PPO(**kwargs)
    return TeleportPPO(scheduler=scheduler, **kwargs)


def build_agent(cfg: ExperimentConfig, env: VecEnv, seed: int) -> BaseAlgorithm:
    """Build the training agent selected by the experiment's algorithm.

    Args:
        cfg: The experiment configuration.
        env: The vectorized training environment.
        seed: RNG seed for the agent.

    Returns:
        The constructed agent.

    Raises:
        NotImplementedError: For algorithms this on-policy factory does not handle
            (the tabular algorithms have their own runners, tasks 11/12).
    """
    if not AgentRegistry.is_registered(cfg.algorithm.kind):
        raise NotImplementedError(
            f"The on-policy agent factory does not handle '{cfg.algorithm.kind.value}'; "
            "the tabular algorithms have their own runners (tasks 11/12)."
        )
    agent = AgentRegistry.create(cfg.algorithm.kind, cfg=cfg, env=env, seed=seed)
    assert agent is not None
    return agent
