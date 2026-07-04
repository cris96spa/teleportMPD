from typing import Any

import pytest
from stable_baselines3 import PPO

from teleport_mdp.agents import TeleportPPO
from teleport_mdp.curriculum import DynamicTeleportScheduler, StaticTeleportScheduler
from teleport_mdp.enums import Algorithm, Curriculum
from teleport_mdp.environments.factory import make_vec_env
from teleport_mdp.models import EnvConfig, ExperimentConfig
from teleport_mdp.registries import AgentRegistry, SchedulerRegistry, build_agent, build_scheduler


def _config(**overrides: Any) -> ExperimentConfig:
    base: dict[str, Any] = {
        "name": "exp",
        "algorithm": {"kind": "ppo", "n_steps": 16, "batch_size": 16, "n_epochs": 1},
        "env": EnvConfig(map_name="4x4", is_slippery=False),
        "gamma": 0.99,
        "total_timesteps": 64,
        "seed": 0,
    }
    base.update(overrides)
    return ExperimentConfig(**base)


def test_build_scheduler_passthrough_for_no_curriculum():
    """The 'none' curriculum is the passthrough key, so no scheduler is built."""
    assert build_scheduler(_config()) is None
    assert SchedulerRegistry.is_registered(Curriculum.NONE)


def test_build_scheduler_selects_static_and_dynamic():
    """The scheduler factory returns the class matching the curriculum kind."""
    static_cfg = _config(teleport={"tau_0": 0.5}, curriculum={"kind": "static"})
    dynamic_cfg = _config(
        teleport={"tau_0": 0.5}, curriculum={"kind": "dynamic", "eps": 1.0, "eps_tau_max": 0.05}
    )
    assert isinstance(build_scheduler(static_cfg), StaticTeleportScheduler)
    assert isinstance(build_scheduler(dynamic_cfg), DynamicTeleportScheduler)


def test_build_agent_returns_vanilla_ppo_without_curriculum():
    """No scheduler -> the agent factory returns a plain PPO (not TeleportPPO)."""
    cfg = _config()
    env = make_vec_env(cfg, n_envs=1, seed=0)
    try:
        agent = build_agent(cfg, env, seed=0)
        assert isinstance(agent, PPO)
        assert not isinstance(agent, TeleportPPO)
    finally:
        env.close()


@pytest.mark.parametrize(
    "curriculum",
    [{"kind": "static"}, {"kind": "dynamic", "eps": 1.0, "eps_tau_max": 0.05}],
)
def test_build_agent_returns_teleport_ppo_for_curriculum(curriculum):
    """A curriculum -> the agent factory returns a TeleportPPO wired to a scheduler."""
    cfg = _config(teleport={"tau_0": 0.5}, curriculum=curriculum)
    env = make_vec_env(cfg, n_envs=1, seed=0)
    try:
        agent = build_agent(cfg, env, seed=0)
        assert isinstance(agent, TeleportPPO)
    finally:
        env.close()


def test_build_agent_rejects_unregistered_algorithm():
    """Algorithms without an on-policy builder raise NotImplementedError."""
    cfg = _config(algorithm={"kind": "q_learning"})
    env = make_vec_env(cfg, n_envs=1, seed=0)
    try:
        assert not AgentRegistry.is_registered(Algorithm.Q_LEARNING)
        with pytest.raises(NotImplementedError):
            build_agent(cfg, env, seed=0)
    finally:
        env.close()
