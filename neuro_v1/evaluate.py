import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from neuro_v1.utils.config import ConfigLoader

def evaluate(episodes=100, model_path=None):
    config = ConfigLoader.load()
    env_name = config['env']['id']
    
    if model_path is None:
        model_path = os.path.join(config['logging']['model_dir'], config['logging']['model_name'])
    
    print(f"Evaluating model: {model_path} on {env_name}")
    
    model = PPO.load(model_path)
    env = gym.make(env_name)
    
    rewards = []
    successes = []
    errors = []
    
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if done:
                rewards.append(episode_reward)
                successes.append(1 if episode_reward > 50 else 0)
                errors.append(info.get('error', 0))
                
    # Analysis
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    success_rate = np.mean(successes) * 100
    avg_error = np.mean(errors)
    
    print(f"\nEvaluation Results ({episodes} episodes):")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Error: {avg_error:.2f}")
    
    # Plotting
    os.makedirs("reports/evaluation", exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Histogram of rewards
    plt.subplot(1, 2, 1)
    plt.hist(rewards, bins=20, color='skyblue', edgecolor='black')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.axvline(mean_reward, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_reward:.1f}')
    plt.legend()
    
    # Success Pie Chart
    plt.subplot(1, 2, 2)
    plt.pie([success_rate, 100-success_rate], labels=['Success', 'Failure'], 
            colors=['#66b3ff', '#ff9999'], autopct='%1.1f%%', startangle=90)
    plt.title(f'Success Rate (N={episodes})')
    
    save_path = f"reports/evaluation/eval_summary_{env_name}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    evaluate()
