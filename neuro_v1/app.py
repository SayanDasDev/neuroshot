from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import sys

# --- 1. SETUP PATHS ---
# Add current directory to path so we can import 'env'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import env  # Registers NeuroShot-v1

app = FastAPI(title="NeuroShot Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CONFIGURATION ---
# Match the path used in HumanVsAI.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "neuro_v1_models", "models", "neuroshot_v1_ppo_model_1100000_steps.zip")
STATS_PATH = os.path.join(os.path.dirname(__file__), "neuro_v1_models", "models", "vecnormalize.pkl") # Optional: If you have it

# Global variables
model = None
env_wrapper = None

class Observation(BaseModel):
    # Expects the 7 values from the environment
    data: list[float]

@app.on_event("startup")
async def load_model():
    global model, env_wrapper
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  WARNING: Model not found at {MODEL_PATH}")
        print("Please train the model or update the path in app.py")
        return

    print(f"Loading Model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    
    # Handle Normalization (Crucial for accuracy)
    if os.path.exists(STATS_PATH):
        print(f"Loading Normalization Stats: {STATS_PATH}")
        # Create a dummy env to hold the stats
        dummy_env = DummyVecEnv([lambda: gym.make("NeuroShot-v1")])
        env_wrapper = VecNormalize.load(STATS_PATH, dummy_env)
        env_wrapper.training = False
        env_wrapper.norm_reward = False
    else:
        print("No normalization stats found. Using raw model.")

    print("✅ System Ready.")

@app.get("/")
def read_root():
    return {"status": "online", "message": "NeuroShot V1 API is running."}

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "normalization_active": env_wrapper is not None
    }

@app.post("/predict")
def predict(obs: Observation):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        observation = np.array(obs.data)
        
        # If we have normalization stats, we must normalize the input manually
        # because the API doesn't have a running 'env' to do it for us.
        if env_wrapper is not None:
            # Normalize observation using the loaded moving average
            observation = env_wrapper.normalize_obs(observation)

        action, _ = model.predict(observation, deterministic=True)
        
        # Action is a numpy array, convert to list for JSON
        return {"action": action.tolist()}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)