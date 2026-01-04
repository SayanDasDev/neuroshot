# neuro_v1/play_vs_ai.py
import pygame
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os
import time

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env

# Colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 0)

class HumanVsAI:
    def __init__(self, model_path):
        self.env = gym.make("NeuroShot-v1", render_mode="human")
        print(f"Loading AI Model: {model_path}")
        self.model = PPO.load(model_path)
        
        # Human Input State
        self.human_angle = 45.0  # Degrees
        self.human_force = 70.0  # Force
        
        self.score_human = 0
        self.score_ai = 0

    def get_trajectory_points(self, v0, theta, wind):
        """Calculates the predicted path for drawing the aim line"""
        points = []
        g = 9.8
        dt = 0.1
        t = 0
        vx = v0 * np.cos(np.radians(theta))
        vy = v0 * np.sin(np.radians(theta))
        
        for _ in range(20): # Draw 20 dots
            x = 50.0 + (vx * t) + (0.5 * wind * t**2)
            y = 350.0 - (vy * t - 0.5 * g * t**2)
            if y > 350: break
            points.append((x, y))
            t += dt
        return points

    def run(self):
        print("\n--- HUMAN VS AI MODE ---")
        print("Controls:")
        print("  [LEFT / RIGHT] : Change Angle")
        print("  [UP / DOWN]    : Change Force")
        print("  [SPACE]        : SHOOT!")
        print("  [Q]            : Quit")
        
        obs, info = self.env.reset(options={"fixed": False})
        
        # We need to manually initialize pygame since we are hijacking the loop
        pygame.init()
        screen = pygame.display.set_mode((800, 400))
        pygame.display.set_caption("NeuroShot: Human vs AI Challenge")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)

        running = True
        waiting_for_shot = True
        
        while running:
            # 1. Handle Human Input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            keys = pygame.key.get_pressed()
            
            if waiting_for_shot:
                # Adjust Aim
                if keys[pygame.K_UP]: self.human_force = min(110, self.human_force + 0.5)
                if keys[pygame.K_DOWN]: self.human_force = max(30, self.human_force - 0.5)
                if keys[pygame.K_LEFT]: self.human_angle = min(85, self.human_angle + 0.5)
                if keys[pygame.K_RIGHT]: self.human_angle = max(15, self.human_angle - 0.5)
                
                # Shoot
                if keys[pygame.K_SPACE]:
                    print(f"Human Shoots! Force: {self.human_force:.1f}, Angle: {self.human_angle:.1f}")
                    
                    # --- HUMAN TURN ---
                    # Convert physics values back to Action Space (-1 to 1) for the step() function
                    # Force: 30..110 -> -1..1
                    action_force = ((self.human_force - 30) / 80 * 2) - 1
                    # Angle: 15..85 -> -1..1
                    action_angle = ((self.human_angle - 15) / 70 * 2) - 1
                    
                    obs, reward, terminated, _, info = self.env.step(np.array([action_force, action_angle], dtype=np.float32))
                    
                    if reward > 50: 
                        print(">>> HUMAN SCORES! <<<")
                        self.score_human += 1
                    else:
                        print("Human Missed.")
                        
                    time.sleep(1) # Pause to let result sink in
                    
                    # --- AI TURN ---
                    print("AI is thinking...")
                    # We reset strictly to the SAME position so it's a fair competition
                    # But since step() resets internally, we might need to restore state or just accept next round
                    # For simplicity, AI takes the NEXT shot on a new target.
                    
                    obs, _ = self.env.reset() # New random target
                    ai_action, _ = self.model.predict(obs, deterministic=True)
                    obs, ai_reward, _, _, ai_info = self.env.step(ai_action)
                    
                    if ai_reward > 50:
                        print(">>> AI SCORES! <<<")
                        self.score_ai += 1
                    else:
                        print("AI Missed.")
                    
                    # Reset for next round
                    obs, _ = self.env.reset()
                    waiting_for_shot = True

            # 2. Rendering
            # Clear screen (Dark Background)
            screen.fill((20, 10, 30))
            
            # Draw UI Text
            text_score = font.render(f"Human: {self.score_human}  |  AI: {self.score_ai}", True, WHITE)
            text_controls = font.render(f"Force: {self.human_force:.1f}  Angle: {self.human_angle:.1f}", True, YELLOW)
            screen.blit(text_score, (20, 20))
            screen.blit(text_controls, (20, 60))

            # Draw "Aim Line" (Angry Birds style)
            # We need the current wind to predict correctly
            # Accessing wind from env is tricky without breaking encapsulation, 
            # so we use a default or last known wind.
            # (Note: For perfect visualization, we'd need to expose env.wind_force)
            if hasattr(self.env.unwrapped, 'wind_force'):
                current_wind = self.env.unwrapped.wind_force
                wind_text = font.render(f"Wind: {current_wind:.2f}", True, CYAN)
                screen.blit(wind_text, (600, 20))
                
                # Draw Trajectory
                aim_points = self.get_trajectory_points(self.human_force, self.human_angle, current_wind)
                if len(aim_points) > 1:
                    pygame.draw.lines(screen, (255, 255, 255), False, aim_points, 1)
                    for p in aim_points:
                        pygame.draw.circle(screen, (255, 255, 255), (int(p[0]), int(p[1])), 2)

            pygame.display.flip()
            clock.tick(60)

        self.env.close()
        pygame.quit()

if __name__ == "__main__":
    # Point this to your BEST model
    model_path = os.path.join("neuro_v1_models", "models", "neuroshot_v1_ppo_model_1100000_steps.zip")
    if os.path.exists(model_path):
        game = HumanVsAI(model_path)
        game.run()
    else:
        print("Model not found! Please train or rename your model.")
        