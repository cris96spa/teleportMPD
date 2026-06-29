from pathlib import Path

import numpy as np
import pytest

from teleport_mdp.enums import TeleportDistribution
from teleport_mdp.environments.factory import (
    build_xi,
    make_env,
    make_vec_env,
    wrap_tmdp,
)
from teleport_mdp.models import EnvConfig, ExperimentConfig, TeleportConfig

# region xi construction


def test_uniform_xi_sums_to_one():
    """A uniform teleport distribution puts equal mass on every state."""
    env = make_env(EnvConfig(map_name="4x4"))
    xi = build_xi(env, TeleportConfig(distribution=TeleportDistribution.UNIFORM))
    n = env.observation_space.n  # type: ignore[attr-defined]
    assert xi.shape == (n,)
    assert xi.sum() == pytest.approx(1.0)
    assert np.allclose(xi, 1.0 / n)


def test_uniform_nonterminal_xi_has_zero_mass_on_terminal_states():
    """The non-terminal uniform distribution never teleports into a hole or goal."""
    env = make_env(EnvConfig(map_name="4x4"))
    xi = build_xi(env, TeleportConfig(distribution=TeleportDistribution.UNIFORM_NONTERMINAL))
    n = env.observation_space.n  # type: ignore[attr-defined]
    assert xi.sum() == pytest.approx(1.0)
    for state in range(n):
        if env.is_terminal(state):
            assert xi[state] == pytest.approx(0.0)
        else:
            assert xi[state] > 0.0


def test_custom_xi_is_loaded_and_normalized(tmp_path: Path):
    """A custom xi is loaded from disk and renormalized to sum to 1."""
    env = make_env(EnvConfig(map_name="4x4"))
    n = env.observation_space.n  # type: ignore[attr-defined]
    weights = np.arange(1, n + 1, dtype=np.float64)  # unnormalized, strictly positive
    path = tmp_path / "xi.npy"
    np.save(path, weights)

    cfg = TeleportConfig(distribution=TeleportDistribution.CUSTOM, custom_xi_path=path)
    xi = build_xi(env, cfg)
    assert xi.sum() == pytest.approx(1.0)
    assert np.allclose(xi, weights / weights.sum())


# endregion

# region update_tau


def test_update_tau_rejects_out_of_range():
    """`update_tau` enforces the half-open teleport range [0, 1)."""
    env = make_env(EnvConfig(map_name="4x4"))
    xi = build_xi(env, TeleportConfig(distribution=TeleportDistribution.UNIFORM))
    tmdp = wrap_tmdp(env, xi, tau=0.5)

    with pytest.raises(ValueError):
        tmdp.update_tau(1.0)
    with pytest.raises(ValueError):
        tmdp.update_tau(-0.1)


def test_update_tau_sets_rate():
    """`update_tau` updates the live teleport probability."""
    env = make_env(EnvConfig(map_name="4x4"))
    xi = build_xi(env, TeleportConfig(distribution=TeleportDistribution.UNIFORM))
    tmdp = wrap_tmdp(env, xi, tau=0.5)
    tmdp.update_tau(0.1)
    assert tmdp.teleport_probability == pytest.approx(0.1)


# endregion

# region tau == 0 equivalence


def test_tau_zero_matches_base_env():
    """With tau == 0 the TMDP never teleports and reproduces the base trajectory."""
    env_cfg = EnvConfig(map_name="4x4", is_slippery=False)
    base = make_env(env_cfg)
    reference = make_env(env_cfg)
    xi = build_xi(base, TeleportConfig(distribution=TeleportDistribution.UNIFORM))
    tmdp = wrap_tmdp(base, xi, tau=0.0)

    tmdp.reset(seed=0)
    reference.reset(seed=0)
    actions = [2, 1, 2, 1, 1, 2, 1, 2]  # deterministic map -> fixed walk to the goal
    for action in actions:
        s_t, r_t, term_t, trunc_t, info_t = tmdp.step(action)
        s_r, r_r, term_r, trunc_r, _ = reference.step(action)
        assert not info_t["teleport"]
        assert (s_t, r_t, term_t, trunc_t) == (s_r, r_r, term_r, trunc_r)
        if term_t:
            break


# endregion

# region vec env / SB3 integration


def test_make_vec_env_trains_ppo():
    """`make_vec_env` yields a VecEnv an SB3 PPO accepts and can step through."""
    from stable_baselines3 import PPO

    cfg = ExperimentConfig(
        name="factory-smoke",
        algorithm={"kind": "ppo", "n_steps": 16, "batch_size": 16, "n_epochs": 1},
        env=EnvConfig(map_name="4x4", is_slippery=False),
        seed=0,
    )
    vec_env = make_vec_env(cfg, n_envs=1, max_episode_steps=50)
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=cfg.algorithm.n_steps,
            batch_size=cfg.algorithm.batch_size,
            n_epochs=cfg.algorithm.n_epochs,
            gamma=cfg.gamma,
            device="cpu",
        )
        model.learn(total_timesteps=32)
    finally:
        vec_env.close()


def test_make_vec_env_wraps_tmdp_when_teleporting():
    """A positive tau_0 routes the vec env through the TMDP teleport layer."""
    cfg = ExperimentConfig(
        name="factory-teleport",
        algorithm={"kind": "ppo"},
        env=EnvConfig(map_name="4x4", is_slippery=False),
        teleport={"tau_0": 0.5, "distribution": "uniform_nonterminal"},
        seed=0,
    )
    vec_env = make_vec_env(cfg, n_envs=1, max_episode_steps=50)
    try:
        # FrozenLake obs space is Discrete(16) -> one-hot Box(16,).
        assert vec_env.observation_space.shape == (16,)
        vec_env.reset()
        teleported = False
        for _ in range(200):
            _, _, _, infos = vec_env.step(np.array([vec_env.action_space.sample()]))
            if infos[0].get("teleport"):
                teleported = True
                break
        assert teleported, "Expected at least one teleport with tau_0=0.5."
    finally:
        vec_env.close()


# endregion
