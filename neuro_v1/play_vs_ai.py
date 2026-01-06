# neuro_v1/play_vs_ai.py

import pygame
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os
import time
import random

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env  # registers NeuroShot-v1

# Colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 0)
GRAY = (150, 150, 150)
DARK_BG = (20, 10, 30)
ORANGE = (255, 165, 0)

class HumanVsAI:
    def __init__(self, model_path: str):
        # NOTE: render_mode=None
        self.env = gym.make("NeuroShot-v1", render_mode=None)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading PPO model: {model_path}")
        self.model = PPO.load(model_path)

        self.human_angle = 45.0
        self.human_force = 70.0
        self.score_human = 0
        self.score_ai = 0

    def get_trajectory_points(self, v0, theta_deg, wind):
        points = []
        g = 9.8
        dt = 0.05
        t = 0.0
        theta = np.radians(theta_deg)
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        start_x, start_y = 50.0, 350.0

        for _ in range(120):
            x = start_x + (vx * t) + (0.5 * wind * t * t)
            y = start_y - (vy * t - 0.5 * g * t * t)
            if y > 350:
                points.append((x, 350))
                break
            if 0 <= x <= 800 and -100 <= y <= 400:
                points.append((x, y))
            t += dt
        return points

    def animate_shot(self, screen, v0, theta_deg, wind, env_core, who="HUMAN"):
        """
        Animates the ball AND moves the basket to match the physics.
        """
        path = self.get_trajectory_points(v0, theta_deg, wind)
        
        # --- FIX: Retrieve Correct Variables from Env ---
        A = env_core.amplitude
        f = env_core.frequency
        start_basket_x = env_core.basket_x  # Changed from basket_center_x
        basket_speed = env_core.basket_speed
        
        t_anim = 0.0
        dt = 0.05 

        for (x, y) in path:
            # 1. Update Basket Position (Matches neuroshot_v1_env.py logic)
            # x = start + (velocity * t) + (Amplitude * sin(freq * t))
            oscillation = A * np.sin(f * t_anim)
            linear_move = basket_speed * t_anim
            basket_x = start_basket_x + linear_move + oscillation
            
            # 2. Draw Scene
            screen.fill(DARK_BG)
            pygame.draw.line(screen, GRAY, (0, 350), (800, 350), 2)
            
            # Draw Moving Basket
            pygame.draw.rect(screen, CYAN, (int(basket_x), 340, 60, 10))
            pygame.draw.line(screen, RED, (int(basket_x) + 30, 340), (int(basket_x) + 30, 350), 2)

            # Draw HUD
            font = pygame.font.Font(None, 28)
            screen.blit(font.render(f"Human: {self.score_human}", True, GREEN), (20, 20))
            screen.blit(font.render(f"AI: {self.score_ai}", True, CYAN), (20, 50))
            screen.blit(font.render(f"Shooting: {who}", True, ORANGE), (350, 50))

            # 3. Draw Ball
            pygame.draw.circle(screen, ORANGE, (int(x), int(y)), 8)
            
            pygame.display.flip()
            time.sleep(0.01) # Speed of animation
            
            t_anim += dt # Advance time

    def run(self):
        print("\n--- FAIR MODE: HUMAN vs AI ---")
        pygame.init()
        screen = pygame.display.set_mode((800, 400))
        pygame.display.set_caption("NeuroShot — Man vs Machine")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 28)
        big_font = pygame.font.Font(None, 40)

        running = True
        waiting_for_shot = True

        current_seed = random.randint(0, 100000)
        obs, _ = self.env.reset(seed=current_seed, options={"fixed": False})

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: running = False
                    if event.key == pygame.K_SPACE and waiting_for_shot:
                        waiting_for_shot = False
                        self.handle_round(screen, big_font, current_seed)
                        current_seed = random.randint(0, 100000)
                        obs, _ = self.env.reset(seed=current_seed, options={"fixed": False})
                        waiting_for_shot = True

            keys = pygame.key.get_pressed()
            if waiting_for_shot:
                if keys[pygame.K_UP]: self.human_force = min(110, self.human_force + 0.5)
                if keys[pygame.K_DOWN]: self.human_force = max(30, self.human_force - 0.5)
                if keys[pygame.K_LEFT]: self.human_angle = min(85, self.human_angle + 0.5)
                if keys[pygame.K_RIGHT]: self.human_angle = max(15, self.human_angle - 0.5)

            # --- DRAW AIMING SCREEN ---
            screen.fill(DARK_BG)
            pygame.draw.line(screen, GRAY, (0, 350), (800, 350), 2)

            # Static Basket (Time is paused while aiming)
            env_core = self.env.unwrapped
            basket_x = env_core.basket_x
            pygame.draw.rect(screen, CYAN, (int(basket_x), 340, 60, 10))
            pygame.draw.line(screen, RED, (int(basket_x) + 30, 340), (int(basket_x) + 30, 350), 2)

            # Trajectory
            wind = env_core.wind_force
            if waiting_for_shot:
                points = self.get_trajectory_points(self.human_force, self.human_angle, wind)
                for p in points:
                    pygame.draw.circle(screen, (100, 100, 100), (int(p[0]), int(p[1])), 1)
                if points:
                    lx, ly = points[-1]
                    pygame.draw.circle(screen, RED, (int(lx), int(ly)), 5)

            # Text
            screen.blit(font.render(f"Human: {self.score_human}", True, GREEN), (20, 20))
            screen.blit(font.render(f"AI: {self.score_ai}", True, CYAN), (20, 50))
            screen.blit(font.render(f"Force: {self.human_force:.1f}  Angle: {self.human_angle:.1f}", True, YELLOW), (20, 360))
            wind_color = RED if abs(wind) > 1.5 else GREEN
            screen.blit(font.render(f"Wind: {wind:.2f}", True, wind_color), (650, 20))
            
            # HINT: Show movement info
            if waiting_for_shot:
                screen.blit(font.render("Aim & Press SPACE", True, WHITE), (300, 360))
                # Show oscillation info
                osc_text = f"Target moves by +/- {env_core.amplitude:.0f}"
                screen.blit(font.render(osc_text, True, GRAY), (600, 50))

            pygame.display.flip()
            clock.tick(60)

        self.env.close()
        pygame.quit()

    def handle_round(self, screen, big_font, seed):
        env_core = self.env.unwrapped
        wind = env_core.wind_force
        
        # HUMAN
        # Pass env_core to animate_shot so it can read Amplitude/Freq
        self.animate_shot(screen, self.human_force, self.human_angle, wind, env_core, "HUMAN")
        
        action_force = ((self.human_force - 30) / 80) * 2 - 1
        action_angle = ((self.human_angle - 15) / 70) * 2 - 1
        _, reward, _, _, info = self.env.step(np.array([action_force, action_angle], dtype=np.float32))

        msg = "HUMAN SCORES!" if info.get("error", 999) < 25 else "Human Missed"
        color = GREEN if info.get("error", 999) < 25 else RED
        self.draw_message(screen, big_font, msg, color)

        # AI
        obs, _ = self.env.reset(seed=seed, options={"fixed": False})
        ai_action, _ = self.model.predict(obs, deterministic=True)
        
        ai_force_real = ((ai_action[0] + 1) / 2) * 80 + 30
        ai_angle_real = ((ai_action[1] + 1) / 2) * 70 + 15
        
        # Pass env_core again (it has fresh reset values)
        self.animate_shot(screen, ai_force_real, ai_angle_real, wind, self.env.unwrapped, "AI AGENT")
        
        _, ai_reward, _, _, ai_info = self.env.step(ai_action)
        msg = "AI SCORES!" if ai_info.get("error", 999) < 25 else "AI Missed"
        color = CYAN if ai_info.get("error", 999) < 25 else WHITE
        self.draw_message(screen, big_font, msg, color)

    def draw_message(self, screen, font, text, color):
        screen.fill(DARK_BG)
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(400, 200))
        screen.blit(surf, rect)
        pygame.display.flip()
        time.sleep(1.0)

if __name__ == "__main__":
    # CHECK THIS PATH
    model_path = "neuro_v1_models/models/neuroshot_v1_ppo_model_1100000_steps.zip"
    HumanVsAI(model_path).run()