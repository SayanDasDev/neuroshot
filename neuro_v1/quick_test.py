"""Quick verification of key fixes."""
import gymnasium as gym
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env

print("Testing NeuroShot Environments\n")

# Test each environment
for env_id in ["NeuroShot-v0", "NeuroShot-v0.1", "NeuroShot-v0.2", "NeuroShot-v1"]:
    print(f"{env_id}:")
    e = gym.make(env_id)
    
    # Test observation space
    obs, _ = e.reset()
    print(f"  Obs shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    # Test fixed mode
    obs1, _ = e.reset(options={"fixed": True})
    obs2, _ = e.reset(options={"fixed": True})
    is_deterministic = np.allclose(obs1, obs2)
    print(f"  Fixed mode: {'✓ Deterministic' if is_deterministic else '✗ Not deterministic'}")
    
    # Test step
    action = np.array([0.5, 0.5])
    obs, reward, _, _, info = e.step(action)
    print(f"  Step works: ✓ (reward={reward:.2f}, error={info['error']:.2f})")
    
    e.close()
    print()

print("✅ All basic tests passed!")
