import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env

def benchmark_algorithms(steps=50000):
    envs_id = "NeuroShot-v1"
    algos = {
        "PPO": PPO,
        "A2C": A2C,
        # "SAC": SAC # SAC requires Box action space, NeuroShot is Box, should work
    }
    
    results = {}
    
    for name, algo_class in algos.items():
        print(f"Training {name}...")
        env = Monitor(gym.make(envs_id))
        
        try:
            model = algo_class("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=steps)
            
            # Extract rewards from monitor
            rewards = env.get_episode_rewards()
            results[name] = rewards
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            
    # Plotting comparison
    plt.figure(figsize=(10, 6))
    
    for name, rewards in results.items():
        # Smooth curve
        series = pd.Series(rewards)
        smooth = series.rolling(window=50, min_periods=1).mean()
        plt.plot(smooth, label=name)
        
    plt.title(f'Algorithm Benchmark on {envs_id}')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("reports/benchmark", exist_ok=True)
    plt.savefig("reports/benchmark/algo_comparison.png")
    print("Benchmark saved to reports/benchmark/algo_comparison.png")

if __name__ == "__main__":
    # Short run for demonstration, increase for real benchmark
    benchmark_algorithms(steps=20000)
