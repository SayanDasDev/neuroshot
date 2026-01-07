import time
import gymnasium as gym
from stable_baselines3 import PPO
import sys, os
import numpy as np  # Needed for angle conversion

# Add path to finding the env folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env

class NeuroShotPlayer:
    def __init__(self, env_name="NeuroShot-v1", model_path=None):
        if model_path is None:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Possible locations for models to auto-detect the best one
            possible_dirs = [
                os.path.join(script_dir, "neuro_v1_models", "models"), # Nested case
                # os.path.join(script_dir, "models"),             # Standard case
                script_dir                                      # Same dir case
            ]
            
            found_model = False
            for d in possible_dirs:
                if os.path.exists(d):
                    # Find all zip files
                    files = [f for f in os.listdir(d) if f.endswith(".zip") and "neuroshot" in f]
                    if files:
                        # Sort by step count. Filename format: name_STEPS.zip
                        def get_steps(fname):
                            parts = fname.replace(".zip", "").split("_")
                            for p in parts:
                                if p.isdigit():
                                    return int(p)
                            return 0
                            
                        files.sort(key=get_steps, reverse=True)
                        model_path = os.path.join(d, files[0])
                        print(f"Found latest model: {model_path} (Steps: {get_steps(files[0])})")
                        found_model = True
                        break
            
            if not found_model:
                print("No model found! Using default path.")
                model_path = os.path.join(script_dir, "neuroshot_v1_ppo_model")

        self.env = gym.make(env_name, render_mode="human")
        self.model = PPO.load(model_path)

    def play(self, episodes=10, fixed=False):
        print(f"Watching Agent play for {episodes} episodes...")
        print("-" * 60)

        for ep in range(episodes):
            # Reset environment
            obs, info = self.env.reset(options={"fixed": fixed})

            # Get action from AI
            action, _ = self.model.predict(obs, deterministic=True)
            
            # --- DEBUG: Decode Action to show Real Physics Values ---
            # Force: -1..1 -> 30..110
            real_force = ((action[0] + 1) / 2) * 80 + 30
            # Angle: -1..1 -> 15..85 degrees
            real_angle = ((action[1] + 1) / 2) * 70 + 15
            # --------------------------------------------------------

            # Execute Step
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Print Stats
            print(
                f"Episode {ep+1:02d}: "
                f"Reward={reward:6.2f} | "
                f"Error={info.get('error', 0):6.2f} | "
                f"Force={real_force:6.2f} | "  # Shows the distinct force
                f"Angle={real_angle:6.2f}°"    # Shows the distinct angle
            )

            # Wait a bit so you can see the shot
            time.sleep(1.0)

    def close(self):
        self.env.close()

if __name__ == "__main__":
    player = NeuroShotPlayer()
    try:
        # Run with fixed=False to show the AI handling random targets!
        player.play(episodes=10, fixed=False)
    finally:
        player.close()

