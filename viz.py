import time
import gymnasium as gym
import env  # This import triggers the registration in env/__init__.py

# Configuration
RENDER_MODE = "human" 

def main():
    # Use gym.make instead of calling the class directly
    # We pass the render_mode here
    # env = gym.make("NeuroShot-v0", render_mode=RENDER_MODE)
    # env = gym.make("NeuroShot-v0.1", render_mode=RENDER_MODE)
    # env = gym.make("NeuroShot-v0.2", render_mode=RENDER_MODE)

    env = gym.make("NeuroShot-v1", render_mode=RENDER_MODE)
    
    print("Starting NeuroShot Registered Env Test...")
    print("Yellow Line = Trajectory | Cyan Box = Moving Basket")

    try:
        for episode in range(10):
            # Standard Gymnasium reset returns (obs, info)
            obs, info = env.reset()
            
            # Sample a random action from the space
            action = env.action_space.sample() 
            
            # Step returns 5 values (standard for Gymnasium v1.0+)
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Episode {episode + 1}:")
            print(f"  - Action Taken: Force {action[0]:.2f}, Angle {action[1]:.2f}")
            print(f"  - Reward: {reward:.2f}")
            print(f"  - Accuracy (Error): {info.get('error', 0):.2f}m")
            print("-" * 30)
            
            # Small delay to see the result
            if RENDER_MODE == "human":
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()