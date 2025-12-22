import time
import gymnasium as gym
from stable_baselines3 import PPO
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env

class NeuroShotPlayer:
    def __init__(self, env_name="NeuroShot-v1", model_path="neuroshot_v1_ppo_model"):
        self.env = gym.make(env_name, render_mode="human")
        self.model = PPO.load(model_path)

    def play(self, episodes=5, fixed=False):
        print(f"Watching Agent play for {episodes} episodes...")

        for ep in range(episodes):
            obs, info = self.env.reset(options={"fixed": fixed})

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)

            print(
                f"Episode {ep+1}: "
                f"Reward = {reward:.2f} | "
                f"Error = {info.get('error', 0):.2f}"
            )

            time.sleep(1.0)

    def close(self):
        self.env.close()

if __name__ == "__main__":
    player = NeuroShotPlayer()
    try:
        player.play(episodes=10, fixed=False)  # set fixed=True to debug
    finally:
        player.close()
