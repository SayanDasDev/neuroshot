# neuro_v1/heatmap_gen.py
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Import Environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env

def generate_heatmap(model_path, episodes=100):
    env = gym.make("NeuroShot-v1")
    model = PPO.load(model_path)
    
    x_positions = []
    wind_forces = []
    results = [] # 1 for Hit, 0 for Miss

    print(f"Running {episodes} episodes to generate heatmap...")

    for _ in range(episodes):
        # We need to access the internal state to record X and Wind
        obs, info = env.reset()
        
        # Access internal variables (Works because NeuroShotEnv is registered)
        # Note: You might need to access env.unwrapped if using wrappers
        unwrapped_env = env.unwrapped
        x_pos = unwrapped_env.basket_x
        wind = unwrapped_env.wind_force
        
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(action)
        
        x_positions.append(x_pos)
        wind_forces.append(wind)
        
        # Check if it was a hit (Error < 25 or Reward > 50)
        is_hit = 1 if info.get('error', 100) < 25 else 0
        results.append(is_hit)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Scatter plot: Green for hits, Red for misses
    colors = ['green' if r == 1 else 'red' for r in results]
    plt.scatter(x_positions, wind_forces, c=colors, alpha=0.6)
    
    plt.title('Accuracy Heatmap: Basket Position vs Wind')
    plt.xlabel('Basket X Position')
    plt.ylabel('Wind Force')
    plt.axhline(0, color='gray', linestyle='--') # Zero wind line
    
    # Save
    os.makedirs("final_reports", exist_ok=True)
    plt.savefig("final_reports/accuracy_heatmap.png")
    print("Heatmap saved to final_reports/accuracy_heatmap.png")

if __name__ == "__main__":
    # Point this to your BEST model
    model_path = os.path.join("neuro_v1_models", "models", "neuroshot_v1_ppo_model_1100000_steps.zip") 
    if os.path.exists(model_path):
        generate_heatmap(model_path, episodes=200)
    else:
        print("Model not found! Rename your best checkpoint to 'neuroshot_v1_ppo_model.zip'")