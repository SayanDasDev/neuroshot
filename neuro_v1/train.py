import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import sys
import os

# Add the parent directory to the path so Python can find 'env'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env 

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

    def train(self, steps=200_000):
        print(f"Training started on {self.env_name} for {steps} steps...")
        self.model.learn(total_timesteps=steps)
        print("Training finished!")

    def save(self):
        self.model.save(self.model_name)
        print(f"Model saved to {self.model_name}.zip")

if __name__ == "__main__":
    trainer = NeuroShotTrainer()
    # Train for real (200,000 steps)
    trainer.train(steps=200000)
    trainer.save()