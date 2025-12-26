"""
Verification script to test all NeuroShot environment versions.
Tests instantiation, observation spaces, and fixed mode functionality.
"""

import gymnasium as gym
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env

def test_environment(env_id, expected_obs_dim):
    """Test a single environment version."""
    print(f"\n{'='*60}")
    print(f"Testing {env_id}")
    print(f"{'='*60}")
    
    try:
        # Test 1: Instantiation
        print(f"✓ Creating environment...")
        test_env = gym.make(env_id)
        print(f"  Environment created successfully")
        
        # Test 2: Observation space
        print(f"✓ Checking observation space...")
        assert test_env.observation_space.shape == (expected_obs_dim,), \
            f"Expected {expected_obs_dim} dimensions, got {test_env.observation_space.shape}"
        print(f"  Observation space: {test_env.observation_space.shape} ✓")
        
        # Test 3: Random reset
        print(f"✓ Testing random reset...")
        obs1, info1 = test_env.reset()
        obs2, info2 = test_env.reset()
        assert obs1.shape == (expected_obs_dim,), f"Observation shape mismatch"
        print(f"  Random reset works ✓")
        print(f"  Sample observation range: [{obs1.min():.3f}, {obs1.max():.3f}]")
        
        # Test 4: Fixed mode
        print(f"✓ Testing fixed mode...")
        obs_fixed1, _ = test_env.reset(options={"fixed": True})
        obs_fixed2, _ = test_env.reset(options={"fixed": True})
        assert np.allclose(obs_fixed1, obs_fixed2), "Fixed mode should be deterministic"
        print(f"  Fixed mode is deterministic ✓")
        
        # Test 5: Step function
        print(f"✓ Testing step function...")
        action = test_env.action_space.sample()
        obs, reward, terminated, truncated, info = test_env.step(action)
        assert obs.shape == (expected_obs_dim,), "Step observation shape mismatch"
        assert 'error' in info, "Info should contain 'error' key"
        print(f"  Step function works ✓")
        print(f"  Sample reward: {reward:.2f}, Error: {info['error']:.2f}")
        
        # Test 6: Observation normalization
        print(f"✓ Checking observation normalization...")
        obs_samples = []
        for _ in range(100):
            obs, _ = test_env.reset()
            obs_samples.append(obs)
        obs_array = np.array(obs_samples)
        obs_min = obs_array.min(axis=0)
        obs_max = obs_array.max(axis=0)
        print(f"  Observation ranges (100 samples):")
        for i in range(expected_obs_dim):
            print(f"    Dim {i}: [{obs_min[i]:.3f}, {obs_max[i]:.3f}]")
        
        # Check if roughly normalized (should be between -2 and 2 for most values)
        assert obs_array.min() > -2.5 and obs_array.max() < 2.5, \
            "Observations should be roughly normalized"
        print(f"  Observations are normalized ✓")
        
        test_env.close()
        print(f"\n✅ All tests passed for {env_id}!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing {env_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all environment tests."""
    print("="*60)
    print("NeuroShot Environment Verification")
    print("="*60)
    
    environments = [
        ("NeuroShot-v0", 4),      # Basket X, Speed, Ball X, Ball Y
        ("NeuroShot-v0.1", 6),    # + Amplitude, Frequency
        ("NeuroShot-v0.2", 5),    # + Wind (no oscillation)
        ("NeuroShot-v1", 7),      # Wind + Amplitude + Frequency
    ]
    
    results = {}
    for env_id, obs_dim in environments:
        results[env_id] = test_environment(env_id, obs_dim)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    for env_id, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{env_id:20s} {status}")
    
    all_passed = all(results.values())
    print("="*60)
    if all_passed:
        print("🎉 All environments verified successfully!")
        return 0
    else:
        print("⚠️  Some environments failed verification")
        return 1

if __name__ == "__main__":
    exit(main())
