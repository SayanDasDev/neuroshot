# C:\Users\raiha\OneDrive\Desktop\neuroshot\neuro_v1\part_1_cont\training_all.py
import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C
import os
import time

def train_all():
    # 1. Setup Environment
    # LunarLanderContinuous is "Moderately Difficult"
    env_id = "LunarLanderContinuous-v3"
    
    # 2. Setup Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "part1_models_cont")
    log_dir = os.path.join(script_dir, "part1_logs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting Training on {env_id}")
    print(f"📂 Models will be saved to: {models_dir}")
    print("-" * 50)

    # --- MODEL 1: SAC (Soft Actor-Critic) ---
    # SAC is usually the Best for continuous control like LunarLander
    print("\n1️⃣  Training SAC (Soft Actor-Critic)...")
    env = gym.make(env_id)
    model_sac = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    start_time = time.time()
    # SAC is sample efficient, 50k steps is usually enough for a good landing
    model_sac.learn(total_timesteps=50_000) 
    print(f"✅ SAC Done in {time.time() - start_time:.2f}s")
    
    sac_path = os.path.join(models_dir, "sac_lunar_model")
    model_sac.save(sac_path)
    print(f"💾 Saved SAC to {sac_path}.zip")

    # --- MODEL 2: PPO (Proximal Policy Optimization) ---
    # PPO is robust but needs more samples than SAC
    print("\n2️⃣  Training PPO (Proximal Policy Optimization)...")
    env = gym.make(env_id)
    model_ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    start_time = time.time()
    # PPO needs more steps to converge nicely
    model_ppo.learn(total_timesteps=80_000)
    print(f"✅ PPO Done in {time.time() - start_time:.2f}s")
    
    ppo_path = os.path.join(models_dir, "ppo_lunar_model")
    model_ppo.save(ppo_path)
    print(f"💾 Saved PPO to {ppo_path}.zip")

    # --- MODEL 3: A2C (Advantage Actor-Critic) ---
    # A2C is older and often performs worse on continuous tasks, good for comparison
    print("\n3️⃣  Training A2C (Advantage Actor-Critic)...")
    env = gym.make(env_id)
    model_a2c = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    start_time = time.time()
    # A2C is fast but unstable
    model_a2c.learn(total_timesteps=80_000)
    print(f"✅ A2C Done in {time.time() - start_time:.2f}s")
    
    a2c_path = os.path.join(models_dir, "a2c_lunar_model")
    model_a2c.save(a2c_path)
    print(f"💾 Saved A2C to {a2c_path}.zip")
    
    print("-" * 50)
    print("🎉 All Models Trained and Saved!")
    print(f"👉 Now run 'python neuro_v1/part1_viz_all.py' to see them.")

if __name__ == "__main__":
    train_all()