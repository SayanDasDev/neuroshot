# neuro_v1/benchmark_part1.py
import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def run_comparative_study(total_timesteps=100000):
    # 1. Problem Selection: LunarLanderContinuous-v2
    # It satisfies the "Continuous Control" requirement.
    env_id = "LunarLanderContinuous-v3"
    print(f"Starting Part 1 Benchmark on {env_id}...")

    # 2. Algorithms Studied (Three distinct types)
    # PPO: Proximal Policy Optimization (On-Policy)
    # A2C: Advantage Actor Critic (On-Policy)
    # SAC: Soft Actor Critic (Off-Policy, specialized for continuous)
    algos = {
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC 
    }

    results = {}

    for name, algo_class in algos.items():
        print(f"Training {name}...")
        
        # Create env
        env = Monitor(gym.make(env_id))
        
        try:
            # Initialize model with standard hyperparameters
            model = algo_class("MlpPolicy", env, verbose=0, seed=42)
            
            # Train
            model.learn(total_timesteps=total_timesteps)
            
            # Extract rewards
            rewards = env.get_episode_rewards()
            results[name] = rewards
            print(f"  > {name} finished with {len(rewards)} episodes.")
            
        except Exception as e:
            print(f"  > {name} Failed: {e}")
            results[name] = []
        finally:
            env.close()

    # 5. Results and Analysis (Plotting)
    plot_results(results)

def plot_results(results):
    plt.figure(figsize=(10, 6))
    
    for name, rewards in results.items():
        if len(rewards) > 0:
            # Smooth the curve so it looks professional (Rolling Mean)
            series = pd.Series(rewards)
            # Window of 50 episodes for smoothing
            smooth = series.rolling(window=50, min_periods=1).mean()
            plt.plot(smooth, label=name)

    plt.title('Part 1: Comparative Study on LunarLanderContinuous-v2')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save for Report
    os.makedirs("final_reports/part1", exist_ok=True)
    save_path = "final_reports/part1/lunar_lander_comparison.png"
    plt.savefig(save_path)
    print(f"\n✅ Graph saved to {save_path}")
    print("Include this image in 'Part 1' of your LaTeX report.")

if __name__ == "__main__":
    # 100,000 steps is enough to see the difference between algorithms
    run_comparative_study(total_timesteps=100000)