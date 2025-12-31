from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from neuro_v1.utils.config import ConfigLoader

app = FastAPI(title="NeuroShot Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
model = None
config = ConfigLoader.load()

class Observation(BaseModel):
    # Depending on env version, size varies. V1 is 7 dims.
    # Accepting list of floats
    data: list[float]

@app.on_event("startup")
async def load_model():
    global model
    model_path = os.path.join(config['logging']['model_dir'], config['logging']['model_name'])
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}. Please train first.")
        # Create a dummy or fail? We'll fail softly for now/
        return
    model = PPO.load(model_path)
    print("Model loaded successfully.")

@app.get("/")
def read_root():
    return {"status": "online", "message": "NeuroShot V1 API is running. Use /predict to get actions."}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(obs: Observation):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        observation = np.array(obs.data)
        action, _ = model.predict(observation, deterministic=True)
        return {"action": action.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
