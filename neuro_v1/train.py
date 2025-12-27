import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path so Python can find 'env'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env 

class PlottingCallback(BaseCallback):
    """
    Custom callback for plotting training progress using matplotlib.
    Tracks episode rewards, success rate, and errors.
    """
    def __init__(self, check_freq=10000, save_path='./training_plots', verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_errors = []
        self.episode_successes = []
        self.timesteps = []
        
    def _on_step(self) -> bool:
        # Get the monitor wrapper to access episode info
        if len(self.model.ep_info_buffer) > 0:
            # Get the most recent episode info
            for info in self.model.ep_info_buffer:
                if 'r' in info:  # 'r' is the episode reward
                    self.episode_rewards.append(info['r'])
                    # Success if reward > 50 (means we got the +100 bonus minus some distance penalty)
                    self.episode_successes.append(1 if info['r'] > 50 else 0)
        
        # Every check_freq steps, update the plot
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            self._plot_progress()
            
        return True
    
    def _plot_progress(self):
        """Generate and save training progress plots"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Calculate rolling averages (window of 100 episodes)
        window = min(100, len(self.episode_rewards))
        if window > 0:
            rewards_smooth = np.convolve(self.episode_rewards, 
                                        np.ones(window)/window, mode='valid')
            successes_smooth = np.convolve(self.episode_successes, 
                                          np.ones(window)/window, mode='valid')
        
        # Plot 1: Episode Rewards
        axes[0].plot(self.episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
        if window > 0:
            axes[0].plot(range(window-1, len(self.episode_rewards)), 
                        rewards_smooth, color='darkblue', linewidth=2, 
                        label=f'Rolling Avg ({window} eps)')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Line')
        axes[0].axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title(f'Training Progress - {self.n_calls:,} Steps')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Success Rate
        if window > 0:
            axes[1].plot(range(window-1, len(self.episode_successes)), 
                        successes_smooth * 100, color='green', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Success Rate (%)')
        axes[1].set_title(f'Success Rate (Rolling {window} episodes)')
        axes[1].set_ylim([0, 105])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.save_path, f'training_progress_{self.n_calls}.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if self.verbose > 0:
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            success_rate = np.mean(self.episode_successes[-100:]) * 100 if len(self.episode_successes) >= 100 else np.mean(self.episode_successes) * 100
            print(f"\n{'='*60}")
            print(f"Progress Update at {self.n_calls:,} steps:")
            print(f"  Avg Reward (last 100 eps): {avg_reward:.2f}")
            print(f"  Success Rate (last 100 eps): {success_rate:.1f}%")
            print(f"  Total Episodes: {len(self.episode_rewards)}")
            print(f"  Plot saved: {plot_path}")
            print(f"{'='*60}\n")

class NeuroShotTrainer:
    def __init__(self, env_name="NeuroShot-v1", model_name="neuroshot_v1_ppo_model"):
        self.env_name = env_name
        self.model_name = model_name

        # 1. Create the Environment Helper
        # We wrap the environment in 'Monitor' so we get nice stats (Reward mean, etc.)
        def make_env():
            return Monitor(gym.make(self.env_name))

        # 2. Vectorize the Environment
        # This is the standard way SB3 expects environments to be passed
        self.env = DummyVecEnv([make_env])

        # 3. Define the Model
        # Using standard parameters that work well for this kind of physics problem
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            verbose=1,
        )

    def train(self, steps, use_callback=True):
        print(f"Training started on {self.env_name} for {steps:,} steps...")
        print(f"This will take approximately {steps/1000000:.1f}M steps")
        print(f"Expected training time: ~{steps/50000:.0f}-{steps/30000:.0f} minutes\n")
        
        if use_callback:
            callback = PlottingCallback(check_freq=10000, verbose=1)
            self.model.learn(total_timesteps=steps, callback=callback)
        else:
            self.model.learn(total_timesteps=steps)
            
        print("\n" + "="*60)
        print("Training finished!")
        print("="*60)

    def save(self):
        self.model.save(self.model_name)
        print(f"Model saved to {self.model_name}.zip")

if __name__ == "__main__":
    trainer = NeuroShotTrainer()
    # Train for 5 million steps (increased from 1M for better performance)
    trainer.train(steps=5000000, use_callback=True)
    trainer.save()