# neuro_v1/benchmark.py
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG  # <--- Added DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise # <--- Needed for DDPG
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env

def benchmark_algorithms(steps=50000):
    envs_id = "NeuroShot-v1"
    
    # DDPG requires action noise for exploration
    env_temp = gym.make(envs_id)
    n_actions = env_temp.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    algos = {
        "PPO": (PPO, {}),
        "A2C": (A2C, {}),
        "DDPG": (DDPG, {"action_noise": action_noise}), # <--- Added DDPG Configuration
    }
    
    results = {}
    
    for name, (algo_class, kwargs) in algos.items():
        print(f"Training {name}...")
        env = Monitor(gym.make(envs_id))
        
        try:
            # DDPG is off-policy, so we use "MlpPolicy" just like PPO
            model = algo_class("MlpPolicy", env, verbose=0, **kwargs)
            model.learn(total_timesteps=steps)
            
            rewards = env.get_episode_rewards()
            results[name] = rewards
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            
    # Plotting comparison
    plt.figure(figsize=(10, 6))
    
    for name, rewards in results.items():
        if len(rewards) > 0:
            series = pd.Series(rewards)
            smooth = series.rolling(window=50, min_periods=1).mean()
            plt.plot(smooth, label=name)
        
    plt.title(f'Algorithm Benchmark: PPO vs A2C vs DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("reports/benchmark", exist_ok=True)
    plt.savefig("reports/benchmark/algo_comparison.png")
    print("Benchmark saved to reports/benchmark/algo_comparison.png")

if __name__ == "__main__":
    benchmark_algorithms(steps=20000) # Increase steps for better results