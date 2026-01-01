import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # <--- FIX 1: Use non-interactive backend to save RAM
import matplotlib.pyplot as plt
import pandas as pd
import gc  # <--- FIX 2: Import Garbage Collector

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env  # Register envs
from neuro_v1.utils.config import ConfigLoader

class PlottingCallback(BaseCallback):
    """
    Custom callback for plotting training progress and saving metrics to CSV.
    """
    def __init__(self, check_freq: int, save_path: str, model_dir: str, model_name: str, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.model_dir = model_dir
        self.model_name = model_name
        os.makedirs(save_path, exist_ok=True)
        
        # Configure CSV logging
        config = ConfigLoader.get()
        self.csv_file = os.path.join(save_path, config.get('logging', {}).get('csv_log_file', 'training_metrics.csv'))
        
        # Initialize DataFrame logic
        self.episode_rewards = []
        self.episode_successes = []
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=['step', 'reward', 'success_rate']).to_csv(self.csv_file, index=False)

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_successes.append(1 if info['r'] > 50 else 0)
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            self._plot_and_log()
            
        return True
    
    def _plot_and_log(self):
        # Calculate stats
        window = min(100, len(self.episode_rewards))
        avg_reward = np.mean(self.episode_rewards[-window:])
        success_rate = np.mean(self.episode_successes[-window:])
        
        # Log to CSV
        new_row = pd.DataFrame([{'step': self.n_calls, 'reward': avg_reward, 'success_rate': success_rate}])
        new_row.to_csv(self.csv_file, mode='a', header=False, index=False)
        
        # Plotting
        try:
            self._generate_plots(window)
        except Exception as e:
            print(f"Warning: Plotting failed (might be out of memory). Error: {e}")

        # Save checkpoint
        checkpoint_path = os.path.join(self.model_dir, f"{self.model_name}_{self.n_calls}_steps")
        self.model.save(checkpoint_path)
        if self.verbose > 0:
            print(f"Callback saved model to {checkpoint_path}")

        if self.verbose > 0:
             print(f"\n[Step {self.n_calls}] Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate*100:.1f}%")

        # <--- FIX 3: Force Garbage Collection to free up RAM
        gc.collect()

    def _generate_plots(self, window):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        if len(self.episode_rewards) > 0:
            # <--- FIX 4: Downsample data! Only plot every 10th point to save memory
            # If you have >10k points, plot fewer.
            step_size = max(1, len(self.episode_rewards) // 5000) 
            
            # Reward Plot
            axes[0].plot(self.episode_rewards[::step_size], alpha=0.3, color='blue', label='Raw (Subsampled)')
            
            # Use pandas rolling on the full data, but then subsample the result for plotting
            rewards_smooth = pd.Series(self.episode_rewards).rolling(window=window, min_periods=1).mean()
            axes[0].plot(rewards_smooth[::step_size], color='darkblue', label=f'Avg ({window})')
            
            axes[0].axhline(y=50, color='green', linestyle='--', label='Success')
            axes[0].set_title(f'Rewards (Showing 1 in {step_size} pts)')
            axes[0].legend()
            
            # Success Plot
            success_smooth = pd.Series(self.episode_successes).rolling(window=window, min_periods=1).mean() * 100
            axes[1].plot(success_smooth[::step_size], color='green')
            axes[1].set_title(f'Success Rate % (Avg {window})')
            axes[1].set_ylim(0, 105)

        plt.tight_layout()
        plot_file = os.path.join(self.save_path, f'progress_{self.n_calls}.png')
        plt.savefig(plot_file)
        # <--- FIX 5: Explicitly close the figure
        plt.close(fig)
        plt.close('all')

class NeuroShotTrainer:
    def __init__(self, config_path="config/default.yaml"):
        self.config = ConfigLoader.load(config_path)
        self.env_name = self.config['env']['id']
        
        # Directory setup
        self.models_dir = self.config['logging']['model_dir']
        self.log_dir = self.config['logging']['log_dir']
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.env = DummyVecEnv([lambda: Monitor(gym.make(self.env_name))])
        
        # Hyperparams
        hp = self.config['training']
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=float(hp['learning_rate']),
            n_steps=hp['n_steps'],
            batch_size=hp['batch_size'],
            gamma=hp['gamma'],
            seed=hp.get('seed', 42),
            verbose=self.config['logging']['verbose'],
            tensorboard_log=self.log_dir
        )

    def train(self):
        steps = self.config['training']['total_timesteps']
        check_freq = self.config['logging']['check_freq']
        save_path = self.config['logging']['save_path']
        
        print(f"Starting training on {self.env_name} for {steps} steps...")
        
        callback = PlottingCallback(
            check_freq=check_freq, 
            save_path=save_path,
            model_dir=self.models_dir,
            model_name=self.config['logging']['model_name']
        )
        self.model.learn(total_timesteps=int(steps), callback=callback)
        
        # Save final model
        model_path = os.path.join(self.models_dir, self.config['logging']['model_name'])
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    trainer = NeuroShotTrainer()
    trainer.train()