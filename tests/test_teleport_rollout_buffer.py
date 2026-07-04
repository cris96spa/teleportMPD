import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from teleport_mdp.agents import TeleportRolloutBuffer

OBS_SPACE = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
ACTION_SPACE = spaces.Discrete(2)


def _zeros_step(reward: float, episode_start: float):
    """One transition with value 0 so that delta == reward."""
    return {
        "obs": np.zeros((1, 2), dtype=np.float32),
        "action": np.zeros((1, 1), dtype=np.float32),
        "reward": np.array([reward], dtype=np.float32),
        "episode_start": np.array([episode_start], dtype=np.float32),
        "value": th.zeros(1),
        "log_prob": th.zeros(1),
    }


def _fill_teleport_buffer(rewards, episode_starts, teleports, gamma, gae_lambda):
    buf = TeleportRolloutBuffer(
        buffer_size=len(rewards),
        observation_space=OBS_SPACE,
        action_space=ACTION_SPACE,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=1,
    )
    buf.reset()
    for reward, start, teleport in zip(rewards, episode_starts, teleports, strict=True):
        step = _zeros_step(reward, start)
        buf.add(
            step["obs"],
            step["action"],
            step["reward"],
            step["episode_start"],
            step["value"],
            step["log_prob"],
            infos=[{"teleport": bool(teleport)}],
        )
    return buf


def test_no_teleports_matches_vanilla_buffer():
    """With all flags False the buffer matches SB3's RolloutBuffer exactly."""
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    starts = [1.0, 0.0, 0.0, 0.0, 0.0]
    gamma, gae_lambda = 0.99, 0.95
    dones = np.array([False])
    last_values = th.zeros(1)

    teleport_buf = _fill_teleport_buffer(rewards, starts, [0] * 5, gamma, gae_lambda)
    teleport_buf.compute_returns_and_advantage(last_values=last_values, dones=dones)

    vanilla = RolloutBuffer(
        buffer_size=5,
        observation_space=OBS_SPACE,
        action_space=ACTION_SPACE,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=1,
    )
    vanilla.reset()
    for reward, start in zip(rewards, starts, strict=True):
        step = _zeros_step(reward, start)
        vanilla.add(
            step["obs"],
            step["action"],
            step["reward"],
            step["episode_start"],
            step["value"],
            step["log_prob"],
        )
    vanilla.compute_returns_and_advantage(last_values=last_values, dones=dones)

    np.testing.assert_allclose(teleport_buf.advantages, vanilla.advantages, atol=1e-6)
    np.testing.assert_allclose(teleport_buf.returns, vanilla.returns, atol=1e-6)


def test_single_teleport_zeros_delta_and_stops_propagation():
    """A teleport at step 1 zeroes its advantage and stops later credit leaking back.

    Hand-computed with gamma=0.5, lambda=1.0, all values 0 so delta == reward:
        no-teleport advantages = [3.25, 4.5, 5.0, 4.0]
        teleport at step 1     = [1.0,  0.0, 5.0, 4.0]
    (step 1 zeroed; step 0 keeps only its own delta=1.0, no leak from steps > 1).
    """
    rewards = [1.0, 2.0, 3.0, 4.0]
    starts = [1.0, 0.0, 0.0, 0.0]
    teleports = [0, 1, 0, 0]
    buf = _fill_teleport_buffer(rewards, starts, teleports, gamma=0.5, gae_lambda=1.0)
    buf.compute_returns_and_advantage(last_values=th.zeros(1), dones=np.array([False]))

    np.testing.assert_allclose(buf.advantages.flatten(), [1.0, 0.0, 5.0, 4.0], atol=1e-6)
    # values are 0, so returns == advantages.
    np.testing.assert_allclose(buf.returns.flatten(), [1.0, 0.0, 5.0, 4.0], atol=1e-6)


def test_always_teleport_zeros_all_advantages():
    """tau=1 (every step teleports) -> all advantages 0 and returns == values."""
    rewards = [1.0, 2.0, 3.0, 4.0]
    starts = [1.0, 0.0, 0.0, 0.0]
    buf = _fill_teleport_buffer(rewards, starts, [1, 1, 1, 1], gamma=0.99, gae_lambda=0.95)
    buf.compute_returns_and_advantage(last_values=th.zeros(1), dones=np.array([False]))

    np.testing.assert_allclose(buf.advantages, 0.0, atol=1e-6)
    np.testing.assert_allclose(buf.returns, buf.values, atol=1e-6)
