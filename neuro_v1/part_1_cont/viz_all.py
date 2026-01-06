# neuro_v1\part_1_cont\viz_all.py
import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C
import os
import time
import cv2  # pip install opencv-python
import numpy as np

def visualize_any():
    # LunarLander is a great "Moderate Difficulty" continuous environment
    env_id = "LunarLanderContinuous-v3"
    
    # --- PATH SETUP ---
    # Automatically find the models folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    neuro_v1_dir = os.path.dirname(script_dir) 
    
    # Check current dir or parent dir for the 'part1_models' folder
    models_dir = os.path.join(script_dir, "part1_models")
    if not os.path.exists(models_dir):
        models_dir = os.path.join(neuro_v1_dir, "part1_models")
    
    # --- 1. USER MENU ---
    print(f"\nEnvironment: {env_id}")
    print("Which model do you want to evaluate?")
    print("1. SAC (Soft Actor-Critic)")
    print("2. PPO (Proximal Policy Optimization)")
    print("3. A2C (Advantage Actor-Critic)")
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

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print(f"   Looking in: {models_dir}")
        print("   Please run 'part1_train_all.py' first to generate the models.")
        return

    print(f"\n🎥 Loading {name.upper()}... (Live Window + Recording)")
    
    # Load the Agent
    model = algo_class.load(model_path)
    
    # CRITICAL: We use 'rgb_array' so we can capture the image for both Video AND Window
    env = gym.make(env_id, render_mode="rgb_array")

    # Video Writer Setup
    video_writer = None
    output_filename = f"{name}_lunar_demo.mp4"
    window_name = f"NeuroShot - {name.upper()} Agent"

    try:
        # Run for 5 Episodes
        for ep in range(5):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            print(f"Starting Episode {ep+1}...")

            while not done:
                # Predict Action
                action, _ = model.predict(obs, deterministic=True)
                
                # Step Environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # --- VISUALIZATION & RECORDING LOGIC ---
                
                # 1. Capture Frame from Gym
                frame = env.render()
                
                # 2. Convert Colors (Gym uses RGB, OpenCV uses BGR)
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 3. Initialize Video Writer (Once)
                if video_writer is None:
                    height, width, _ = bgr_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_filename, fourcc, 30, (width, height))

                # 4. Write to Video File
                video_writer.write(bgr_frame)

                # 5. Show in Live Window
                # Add text to the window for clarity
                cv2.putText(bgr_frame, f"Ep: {ep+1} | Reward: {total_reward:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, bgr_frame)
                
                # 6. Window Management
                # Press 'q' to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            print(f"  -> Finished Episode {ep+1}: Total Reward = {total_reward:.2f}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    
    finally:
        # Cleanup
        env.close()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Visualization Complete.")
        print(f"📁 Video saved to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    visualize_any()