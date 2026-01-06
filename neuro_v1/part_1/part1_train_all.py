# neuro_v1/part1_train_all.py
import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C
import os

def train_all_algos():
    env_id = "LunarLanderContinuous-v3"
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    neuro_v1_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(neuro_v1_dir, "part1_models")
    # models_dir = "neuro_v1/part1_models"
    total_timesteps = 150000
    
    # Define the algorithms to train
    algos = {
        "SAC": SAC,
        "PPO": PPO,
        "A2C": A2C
    }

    os.makedirs(models_dir, exist_ok=True)

    print(f"🚀 Starting Mass Training for Part 1 ({env_id})")
    print(f"   Algorithms: {list(algos.keys())}")
    print(f"   Steps per algo: {total_timesteps}\n")

    for name, algo_class in algos.items():
        save_path = os.path.join(models_dir, f"{name.lower()}_lunar_model")
        
        # Check if already exists to avoid overwriting
        if os.path.exists(f"{save_path}.zip"):
            print(f"✅ {name} model already exists. Skipping...")
            continue

        print(f"👉 Training {name}...")
        env = gym.make(env_id)
        
        # Initialize
        model = algo_class("MlpPolicy", env, verbose=0, seed=42)
        
        # Train
        model.learn(total_timesteps=total_timesteps)
        
        # Save
        model.save(save_path)
        print(f"   💾 Saved {name} to {save_path}.zip")
        env.close()

    print("\n🎉 All models trained and saved!")

if __name__ == "__main__":
    train_all_algos()