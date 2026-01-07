import time
import gymnasium as gym
from stable_baselines3 import PPO
import sys, os
import numpy as np
import cv2  # REQUIRED: pip install opencv-python

# Add path to finding the env folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env

class NeuroShotPlayer:
    def __init__(self, env_name="NeuroShot-v1", model_path=None, record_video=False):
        # 1. Detect Model Path
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_dirs = [
                os.path.join(script_dir, "neuro_v1_models", "models"),
                script_dir
            ]
            
            found_model = False
            for d in possible_dirs:
                if os.path.exists(d):
                    files = [f for f in os.listdir(d) if f.endswith(".zip") and "neuroshot" in f]
                    if files:
                        # Sort by step count (e.g. model_1000.zip)
                        def get_steps(fname):
                            parts = fname.replace(".zip", "").split("_")
                            for p in parts:
                                if p.isdigit(): return int(p)
                            return 0
                            
                        files.sort(key=get_steps, reverse=True)
                        model_path = os.path.join(d, files[0])
                        print(f"Found latest model: {model_path}")
                        found_model = True
                        break
            
            if not found_model:
                print("No model found! Using default path.")
                model_path = os.path.join(script_dir, "neuroshot_v1_ppo_model")

        # 2. Initialize Environment
        # Crucial: Use 'rgb_array' to get pixel data instead of opening a window
        self.render_mode = "rgb_array" if record_video else "human"
        self.env = gym.make(env_name, render_mode=self.render_mode)
        self.model = PPO.load(model_path)
        self.record_video = record_video

    def play(self, episodes=5, fixed=False, output_file="neuroshot_demo.mp4"):
        print(f"Running {episodes} episodes (Video Recording: {self.record_video})...")
        print("-" * 60)

        video_writer = None
        
        try:
            for ep in range(episodes):
                obs, info = self.env.reset(options={"fixed": fixed})
                
                # AI Predicts
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Decode for display
                real_force = ((action[0] + 1) / 2) * 80 + 30
                real_angle = ((action[1] + 1) / 2) * 70 + 15

                # Execute Step
                obs, reward, terminated, truncated, info = self.env.step(action)

                # --- VIDEO RECORDING LOGIC ---
                if self.record_video and "frames" in info:
                    frames = info["frames"]
                    if not frames: continue

                    # Initialize Writer once
                    if video_writer is None:
                        height, width, layers = frames[0].shape
                        # OpenCV uses BGR, Pygame uses RGB
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                        video_writer = cv2.VideoWriter(output_file, fourcc, 60, (width, height))

                    # Write frames to video
                    for frame in frames:
                        # Convert RGB (Pygame) to BGR (OpenCV)
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(bgr_frame)

                print(
                    f"Episode {ep+1:02d}: "
                    f"Reward={reward:6.2f} | "
                    f"Error={info.get('error', 0):6.2f} | "
                    f"Force={real_force:6.2f} | "
                    f"Angle={real_angle:6.2f}°"
                )
                
                # If human mode, sleep to watch it
                if not self.record_video:
                    time.sleep(1.0)

        finally:
            self.env.close()
            if video_writer:
                video_writer.release()
                print(f"\n✅ Video saved successfully to: {output_file}")

if __name__ == "__main__":
    # SET record_video=True TO SAVE MP4
    player = NeuroShotPlayer(record_video=True)
    player.play(episodes=10, fixed=False, output_file="neuroshot_demo.mp4")