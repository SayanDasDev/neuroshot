# neuro_v1/part1_viz_all.py
import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C
import os
import time
import cv2 # pip install opencv-python
import numpy as np

def visualize_any():
    env_id = "LunarLanderContinuous-v3"
    
    # --- PATH SETUP ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    neuro_v1_dir = os.path.dirname(script_dir) # Adjust if needed based on your folder structure
    # Try looking in current folder first, then parent
    models_dir = os.path.join(script_dir, "part1_models")
    if not os.path.exists(models_dir):
        models_dir = os.path.join(neuro_v1_dir, "part1_models")
    
    # 1. Ask User which one to load
    print("\nWhich model do you want to see?")
    print("1. SAC (The Best)")
    print("2. PPO (Okay)")
    print("3. A2C (Bad/Crashes)")
    choice = input("Enter 1, 2, or 3: ")
    
    if choice == "1":
        name = "sac"
        algo_class = SAC
    elif choice == "2":
        name = "ppo"
        algo_class = PPO
    elif choice == "3":
        name = "a2c"
        algo_class = A2C
    else:
        print("Invalid choice. Defaulting to SAC.")
        name = "sac"
        algo_class = SAC

    model_path = os.path.join(models_dir, f"{name}_lunar_model.zip")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"   Searched in: {models_dir}")
        print("   Run 'python neuro_v1/part1_train_all.py' first.")
        return

    print(f"\n🎥 Playing {name.upper()} Demo (Recording to video)...")
    
    model = algo_class.load(model_path)
    
    # CRITICAL CHANGE: Use 'rgb_array' to get pixel data
    env = gym.make(env_id, render_mode="rgb_array")

    video_writer = None
    output_filename = f"{name}_lunar_demo.mp4"

    try:
        for ep in range(5):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # --- RECORDING & DISPLAY LOGIC ---
                # 1. Get the frame from Gym
                frame = env.render()
                
                # 2. Convert RGB (Gym) to BGR (OpenCV)
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 3. Initialize Video Writer on first frame
                if video_writer is None:
                    height, width, _ = bgr_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_filename, fourcc, 30, (width, height))

                # 4. Write frame to file
                video_writer.write(bgr_frame)

                # 5. Show frame in a window (Live View)
                cv2.imshow(f"Lunar Lander - {name.upper()}", bgr_frame)
                
                # Handle window closing or 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"✅ Video saved: {output_filename}")

if __name__ == "__main__":
    visualize_any()