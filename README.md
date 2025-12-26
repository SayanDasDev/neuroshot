# NeuroShot 🎯

A reinforcement learning environment for training AI agents to solve projectile motion challenges with moving targets.

## Overview

NeuroShot is a custom Gymnasium environment where an agent learns to shoot a ball into a moving basket by controlling the launch force and angle. The environment features progressively complex physics including wind forces and non-linear basket movement.

## Environment Versions

### NeuroShot-v0 (Basic Physics)
- **Features**: Simple projectile motion with linear basket movement
- **Observation Space**: 4 dimensions (normalized)
  - Basket X position (0-1)
  - Basket speed (-1 to 0)
  - Ball X position (0-1)
  - Ball Y position (0-1)
- **Complexity**: Easiest - good for initial testing

### NeuroShot-v0.1 (Non-Linear Movement)
- **Features**: Adds oscillating basket movement (sine wave)
- **Observation Space**: 6 dimensions (normalized)
  - Basket X position (0-1)
  - Basket speed (-1 to 0)
  - Oscillation amplitude (0.4-1)
  - Oscillation frequency (0.25-1)
  - Ball X position (0-1)
  - Ball Y position (0-1)
- **Complexity**: Medium - requires predicting oscillation

### NeuroShot-v0.2 (Wind Forces)
- **Features**: Adds horizontal wind acceleration
- **Observation Space**: 5 dimensions (normalized)
  - Basket X position (0-1)
  - Basket speed (-1 to 0)
  - Wind force (-1 to 1)
  - Ball X position (0-1)
  - Ball Y position (0-1)
- **Complexity**: Medium - requires compensating for wind

### NeuroShot-v1 (Full Complexity)
- **Features**: Combines wind + non-linear oscillating movement
- **Observation Space**: 7 dimensions (normalized)
  - Basket X position (0-1)
  - Basket speed (-1 to 0)
  - Wind force (-0.6 to 0.6)
  - Oscillation amplitude (0.4-1)
  - Oscillation frequency (0.25-1)
  - Ball X position (0-1)
  - Ball Y position (0-1)
- **Complexity**: Hardest - full challenge

## Action Space

All versions use the same action space:
- **Dimension**: 2 (continuous)
- **Range**: [-1, 1] for both actions
- **Action 0**: Launch force (mapped to 30-110 units)
- **Action 1**: Launch angle (mapped to 15-85 degrees)

## Reward Structure

All environments use consistent reward shaping:
- **Success**: +100 points if ball lands within 25 pixels of basket
- **Distance penalty**: -0.05 × error distance (provides gradient signal)
- **Episode**: Single-step (terminates after one shot)

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd neuroshot

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training an Agent

```python
# Train on the full complexity environment
cd neuro_v1
python train.py
```

The training script will:
- Train a PPO agent for 1,000,000 steps
- Save the model as `neuroshot_v1_ppo_model.zip`
- Display training progress and statistics

### Visualizing a Trained Agent

```python
# Watch the trained agent play
cd neuro_v1
python viz.py
```

This will render 10 episodes showing the agent's performance.

### Using Environments Directly

```python
import gymnasium as gym
import env  # Import to register environments

# Create any environment version
env = gym.make("NeuroShot-v1", render_mode="human")

# Reset with optional fixed mode for debugging
obs, info = env.reset(options={"fixed": True})  # Deterministic
# obs, info = env.reset()  # Random

# Take an action
action = env.action_space.sample()  # Random action
obs, reward, terminated, truncated, info = env.step(action)

print(f"Reward: {reward:.2f}, Error: {info['error']:.2f}")
env.close()
```

## Project Structure

```
neuroshot/
├── env/                          # Environment definitions
│   ├── __init__.py              # Gymnasium registration
│   ├── neuroshot_v0_env.py      # Basic physics
│   ├── neuroshot_v0_1_env.py    # + Non-linear movement
│   ├── neuroshot_v0_2_env.py    # + Wind forces
│   └── neuroshot_v1_env.py      # Full complexity
├── neuro_v1/                     # Training & visualization
│   ├── train.py                 # PPO training script
│   ├── viz.py                   # Visualization script
│   └── neuroshot_v1_ppo_model.zip  # Trained model (after training)
└── requirements.txt              # Dependencies
```

## Technical Details

### Physics Simulation
- **Gravity**: 9.8 m/s²
- **Time step**: 0.03s (consistent across all versions)
- **Ground level**: Y = 350 pixels
- **World size**: 800×400 pixels

### Basket Dynamics
- **Initial position**: Random between X=400-700
- **Speed**: Random between -10 to -5 (moving left)
- **Oscillation** (v0.1, v1):
  - Amplitude: 20-50 pixels
  - Frequency: 0.5-2.0 Hz
- **Wind** (v0.2, v1):
  - Force: -4 to +4 (horizontal acceleration)

### Observation Normalization
All observations are normalized to approximately [-1, 1] or [0, 1] ranges to improve neural network training stability.

## Training Tips

1. **Start Simple**: Begin with `NeuroShot-v0` to verify your setup
2. **Use Fixed Mode**: Debug with `options={"fixed": True}` for reproducible scenarios
3. **Hyperparameters**: The default PPO settings work well, but you can adjust:
   - `learning_rate`: 3e-4
   - `n_steps`: 2048
   - `batch_size`: 64
   - `gamma`: 0.99

4. **Training Time**: Expect ~200k-1M steps for good performance depending on version complexity

## Rendering

All environments support human rendering with pygame:
- **Dark background**: Different tint per version
- **Ground line**: Gray horizontal line
- **Basket**: Cyan rectangle (60×10 pixels)
- **Ball**: White circle (radius 6)
- **Trajectory**: Colored path trace
- **Wind gauge** (v0.2, v1): Visual indicator at top

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

Built with:
- [Gymnasium](https://gymnasium.farama.org/) - RL environment framework
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [PyGame](https://www.pygame.org/) - Rendering
