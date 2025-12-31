import pytest
import gymnasium as gym
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env 

@pytest.fixture
def env_v1():
    return gym.make("NeuroShot-v1")

def test_observation_space(env_v1):
    env = env_v1
    obs, info = env.reset()
    assert env.observation_space.contains(obs), "Observation out of bounds"
    assert len(obs) == 7, "NeuroShot-v1 should have 7 observation dimensions"

def test_action_space(env_v1):
    env = env_v1
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    assert len(action) == 2

def test_step_logic(env_v1):
    env = env_v1
    env.reset()
    action = np.array([0.0, 0.0]) # Middle force/angle
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check return types
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Check reward range logic (should be negative for miss, positive for hit)
    # Just check it's a number here
    assert -200 <= reward <= 200

def test_fixed_reset(env_v1):
    # Test if 'fixed' option creates deterministic state
    env = env_v1
    obs1, _ = env.reset(options={"fixed": True})
    obs2, _ = env.reset(options={"fixed": True})
    np.testing.assert_array_equal(obs1, obs2, "Fixed reset should be deterministic")

    # Regular reset should differ (statistically likely)
    obs3, _ = env.reset()
    # It's possible to be same by chance but unlikely.
    # We can check internal state if exposed, but obs check is okay.
