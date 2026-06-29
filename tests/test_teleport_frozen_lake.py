from typing import cast

import gymnasium as gym
import numpy as np
import pytest

from teleport_mdp.environments.teleport_frozen_lake import (
    TeleportFrozenLakeEnv,
)
from teleport_mdp.utils.frozen_lake import generate_random_map
from teleport_mdp.wrappers.tmdp import TMDP


@pytest.fixture
def env():
    """Fixture to initialize the environment with default parameters."""
    return cast(TeleportFrozenLakeEnv, gym.make("teleport_env/FrozenLake-v0").unwrapped)


@pytest.fixture
def tmdp_env(env: TeleportFrozenLakeEnv):
    """Fixture to initialize the TMDP wrapper with a sample teleport probability distribution."""
    # Example teleport probability distribution that sums to 1
    teleport_prob_distribution = (
        np.ones(env.observation_space.n) / env.observation_space.n
    )  # type: ignore
    return TMDP(
        env=env,
        teleport_prob_distribution=teleport_prob_distribution,
        teleport_probability=0.2,
    )


# region Teleport Frozen Lake Tests
def test_environment_creation(env):
    """Test environment instantiation with default parameters."""
    assert isinstance(env, TeleportFrozenLakeEnv)
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.action_space.n == 4  # Four possible actions (LEFT, DOWN, RIGHT, UP)


def test_custom_map_creation():
    """Test environment creation with a custom random map."""
    custom_map = generate_random_map(size=6, p=0.8, seed=42)
    env: TeleportFrozenLakeEnv = gym.make(
        "teleport_env/FrozenLake-v0", desc=custom_map, is_slippery=True
    ).unwrapped  # type: ignore
    assert env.desc.shape == (6, 6), "Custom map size should be 6x6."


def test_step_function(env):
    """Test taking a single step in the environment."""
    initial_obs, _ = env.reset()
    action = env.action_space.sample()  # Take a random action
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(next_obs, int)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "prob" in info, "Info dictionary should contain 'prob' key"


def test_episode_termination(env):
    """Test that the episode terminates when reaching a terminal state."""
    env.reset()
    terminated = False
    steps = 0
    max_steps = 100  # To prevent an infinite loop

    while not terminated and steps < max_steps:
        action = env.action_space.sample()
        _, _, terminated, _, _ = env.step(action)
        steps += 1

    assert steps < max_steps, (
        "Environment should reach a terminal state within a reasonable number of steps."
    )


def test_rendering(env):
    """Test rendering modes: 'ansi' and 'rgb_array'."""
    env.reset()
    env.render_mode = "ansi"
    # Test ANSI rendering
    ansi_output = env.render()
    assert isinstance(ansi_output, str), "ANSI mode should return a string."

    # Test RGB Array rendering (if available)
    if "rgb_array" in env.metadata.get("render_modes", []):
        env.render_mode = "rgb_array"
        rgb_array_output = env.render()
        assert isinstance(rgb_array_output, np.ndarray), (
            "RGB mode should return an array."
        )


def test_reset_function(env):
    """Test the reset function to ensure environment resets correctly."""
    obs, info = env.reset()
    assert isinstance(obs, int), "Reset should return an integer observation."
    assert "prob" in info, "Info dictionary should contain 'prob' key on reset."


def test_teleport(env: TeleportFrozenLakeEnv):
    """Test the teleport method to ensure it returns a valid non-terminal state."""
    # Set up a teleport probability distribution where each state has an equal chance of being chosen
    num_states = env.observation_space.n  # type: ignore
    teleport_prob_distribution = np.ones(num_states) / num_states

    # Perform the teleport
    new_state = env.teleport(teleport_prob_distribution)

    # Assert that the new state is a valid observation
    assert isinstance(new_state, int), "teleport should return an integer state"
    assert 0 <= new_state < num_states, (
        "teleport should return a valid state within the observation space range"
    )

    # Assert that the teleported state is non-terminal (not a hole or goal)
    assert not env.is_terminal(new_state), "teleport should not return a terminal state"


# endregion

# region TMDP Wrapper Tests


def test_tmdp_initialization(env):
    """Test that TMDP initializes correctly with valid parameters and raises errors on invalid input."""
    # Valid initialization
    teleport_prob_distribution = (
        np.ones(env.observation_space.n) / env.observation_space.n
    )
    tmdp = TMDP(
        env=env,
        teleport_prob_distribution=teleport_prob_distribution,
        teleport_probability=0.2,
    )
    assert isinstance(tmdp, TMDP)

    # Check invalid teleport probability distribution
    invalid_teleport_distribution = np.ones(env.observation_space.n) * 0.5
    with pytest.raises(ValueError, match="The teleport distribution must sum to 1."):
        TMDP(env=env, teleport_prob_distribution=invalid_teleport_distribution)

    # Check teleport probability out of range
    with pytest.raises(
        ValueError, match="The teleport probability must be in the range \\[0, 1\\]."
    ):
        TMDP(
            env=env,
            teleport_prob_distribution=teleport_prob_distribution,
            teleport_probability=1.5,
        )


def test_tmdp_step_no_teleport(tmdp_env):
    """Test the TMDP step function without teleportation (teleport_probability=0)."""
    tmdp_env.teleport_probability = (
        0.0  # Set teleport probability to zero to disable teleportation
    )
    tmdp_env.reset()
    action = tmdp_env.action_space.sample()

    next_state, reward, terminated, truncated, info = tmdp_env.step(action)

    assert not info["teleport"], (
        "Teleport should not occur with teleport_probability set to 0."
    )
    assert isinstance(next_state, int)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_tmdp_step_with_teleport(tmdp_env):
    """Test the TMDP step function with teleportation (teleport_probability=1)."""
    tmdp_env.teleport_probability = (
        1.0  # Set teleport probability to one to force teleportation
    )
    tmdp_env.reset()
    action = tmdp_env.action_space.sample()

    next_state, reward, terminated, truncated, info = tmdp_env.step(action)

    assert info["teleport"], "Teleport should occur with teleport_probability set to 1."
    assert next_state >= 0 and next_state < tmdp_env.env.observation_space.n
    assert isinstance(reward, float)
    assert reward == 0, "Reward should be zero during teleportation."
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_tmdp_step_partial_teleport_probability(tmdp_env):
    """Test TMDP step function with partial teleportation probability (teleport_probability=0.5)."""
    tmdp_env.teleport_probability = 0.5  # Set teleport probability to 0.5
    tmdp_env.reset()
    action = tmdp_env.action_space.sample()

    # Run multiple steps to observe teleport behavior
    teleport_count = 0
    non_teleport_count = 0
    for _ in range(400):
        _, _, _, _, info = tmdp_env.step(action)
        if info["teleport"]:
            teleport_count += 1
        else:
            non_teleport_count += 1

    assert teleport_count > 0, "Expected some teleports with teleport_probability=0.5."
    assert non_teleport_count > 0, (
        "Expected some regular steps with teleport_probability=0.5."
    )


def test_tmdp_reset(tmdp_env):
    """Test the reset function of TMDP wrapper."""
    initial_state, info = tmdp_env.reset()
    assert isinstance(initial_state, int), "Reset should return an integer state."
    assert "prob" in info, "Info dictionary should contain 'prob' key after reset."
    assert not getattr(info, "teleport", False), (
        "Info should indicate no teleport on initial reset."
    )


# endregion
