# neuro_v1/part1_viz_all.py
import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C
import os
import time

def visualize_any():
    env_id = "LunarLanderContinuous-v3"
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    neuro_v1_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(neuro_v1_dir, "part1_models")
    # models_dir = "neuro_v1/part1_models"
    
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
        print("   Run 'python neuro_v1/part1_train_all.py' first.")
        return

    print(f"\n🎥 Playing {name.upper()} Demo...")
    model = algo_class.load(model_path)
    env = gym.make(env_id, render_mode="human")

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
                # time.sleep(0.01) # Uncomment to slow down
            
            print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    visualize_any()