from collections.abc import Callable
from typing import ClassVar

from teleport_mdp.curriculum.scheduler import (
    DynamicTeleportScheduler,
    StaticTeleportScheduler,
    TeleportScheduler,
)
from teleport_mdp.enums import Curriculum
from teleport_mdp.models import ExperimentConfig, PPOConfig
from teleport_mdp.registries.base import ComponentRegistry


class SchedulerRegistry(ComponentRegistry[TeleportScheduler]):
    """Registry of teleport-rate schedulers keyed by :class:`Curriculum`."""

    _registry: ClassVar[dict[str, Callable[..., TeleportScheduler]]] = {}
    _PASSTHROUGH_KEY: ClassVar[str | None] = Curriculum.NONE.value


@SchedulerRegistry.register(Curriculum.STATIC)
def build_static(*, cfg: ExperimentConfig) -> StaticTeleportScheduler:
    """Build the static scheduler, reaching `tau == 0` over the run's updates.

    Args:
        cfg: The experiment configuration.

    Returns:
        A static teleport scheduler whose budget is the number of PPO updates,
        `total_timesteps // (n_steps * n_envs)`.
    """
    assert isinstance(cfg.algorithm, PPOConfig)
    n_updates = max(1, cfg.total_timesteps // (cfg.algorithm.n_steps * cfg.n_envs))
    return StaticTeleportScheduler(cfg.gamma, cfg.teleport.tau_0, n_updates)


@SchedulerRegistry.register(Curriculum.DYNAMIC)
def build_dynamic(*, cfg: ExperimentConfig) -> DynamicTeleportScheduler:
    """Build the dynamic, policy-shift-aware scheduler.

    Args:
        cfg: The experiment configuration (`curriculum.eps` / `eps_tau_max` are
            guaranteed present by the config validator).

    Returns:
        A dynamic teleport scheduler.
    """
    assert cfg.curriculum.eps is not None
    assert cfg.curriculum.eps_tau_max is not None
    return DynamicTeleportScheduler(
        cfg.gamma, eps=cfg.curriculum.eps, eps_tau_max=cfg.curriculum.eps_tau_max
    )


def build_scheduler(cfg: ExperimentConfig) -> TeleportScheduler | None:
    """Build the teleport-rate scheduler for an experiment's curriculum.

    Args:
        cfg: The experiment configuration.

    Returns:
        The configured scheduler, or `None` when `curriculum.kind` is `none`.
    """
    return SchedulerRegistry.create(cfg.curriculum.kind, cfg=cfg)
