import pygame
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os
import time
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import env  # Registers NeuroShot-v1 from your env/__init__.py

# --- Constants & Colors ---
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 0)
GRAY = (150, 150, 150)
DARK_BG = (20, 10, 30)
ORANGE = (255, 165, 0)

# Window Dimensions
WIDTH, HEIGHT = 800, 400

class HumanVsAI:
    def __init__(self, model_path: str, stats_path: str = None):
        """
        model_path: Path to the .zip PPO model file.
        stats_path: Path to the .pkl VecNormalize stats file (optional, but recommended if used during training).
        """
        # 1. Initialize Environment
        self.raw_env = gym.make("NeuroShot-v1", render_mode=None)
        
        # 2. Load Model & Normalization Stats
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        print(f"Loading PPO model: {model_path}")
        self.model = PPO.load(model_path)
        
        # If specific normalization stats exist, wrap the env to match training conditions
        if stats_path and os.path.exists(stats_path):
            print(f"Loading normalization stats: {stats_path}")
            self.env = DummyVecEnv([lambda: self.raw_env])
            self.env = VecNormalize.load(stats_path, self.env)
            self.env.training = False # Do not update stats during test
            self.env.norm_reward = False 
        else:
            self.env = self.raw_env

        # 3. Game State
        self.human_angle = 45.0
        self.human_force = 70.0
        self.score_human = 0
        self.score_ai = 0

    def animate_shot(self, screen, v0, theta_deg, wind, env_core, who="HUMAN"):
        """
        Simulates the shot frame-by-frame for visualization.
        CRITICAL: Physics constants must match neuroshot_v1_env.py exactly.
        """
        g = 9.8
        dt = 0.08  # CORRECTED: Matches env/neuroshot_v1_env.py (was 0.03)
        t = 0.0

        theta = np.radians(theta_deg)
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)

        ball_x, ball_y = 50.0, 350.0

        # Snapshot of target parameters
        A = env_core.amplitude
        f = env_core.frequency
        start_basket_x = env_core.basket_x
        basket_speed = env_core.basket_speed

        final_basket_x = start_basket_x
        
        # Loop until ball hits ground or leaves screen
        running = True
        while running:
            # 1. Physics Step (Matches Env)
            t += dt
            
            # Ball: x = x0 + vx*t + 0.5*wind*t^2
            ball_x = 50.0 + (vx * t) + (0.5 * wind * t * t)
            ball_y = 350.0 - (vy * t - 0.5 * g * t * t)

            # Basket: x = x0 + v*t + A*sin(f*t)
            oscillation = A * np.sin(f * t)
            basket_x = start_basket_x + basket_speed * t + oscillation
            final_basket_x = basket_x

            # 2. Rendering
            screen.fill(DARK_BG)
            
            # Ground
            pygame.draw.line(screen, GRAY, (0, 350), (WIDTH, 350), 2)

            # Wind Indicator
            self.draw_wind_gauge(screen, wind)

            # Basket
            pygame.draw.rect(screen, CYAN, (int(basket_x), 340, 60, 10))
            # Basket Center Marker (Hit Box)
            pygame.draw.line(screen, RED, (int(basket_x) + 30, 340), (int(basket_x) + 30, 350), 2)

            # Player Label
            font = pygame.font.Font(None, 28)
            screen.blit(font.render(f"Shooting: {who}", True, ORANGE), (330, 40))

            # Ball
            pygame.draw.circle(screen, ORANGE, (int(ball_x), int(ball_y)), 8)

            pygame.display.flip()
            
            # Smooth animation delay (approx 30 FPS for viewing, independent of physics dt)
            time.sleep(0.02) 

            # Check bounds
            if ball_y > 350 or ball_x > WIDTH:
                running = False

        return ball_x, final_basket_x

    def draw_wind_gauge(self, screen, wind):
        # Draw a simple arrow at the top to show wind
        center = (400, 30)
        end = (400 + int(wind * 20), 30)
        pygame.draw.line(screen, WHITE, (350, 30), (450, 30), 1) # Axis
        if abs(wind) > 0.1:
            color = GREEN if wind > 0 else RED
            pygame.draw.line(screen, color, center, end, 3)
            pygame.draw.circle(screen, color, end, 3)

    def check_score(self, ball_x, basket_x):
        # Hit logic: Distance from ball center to basket center < 25
        basket_center = basket_x + 30
        return abs(ball_x - basket_center) < 25

    def handle_round(self, screen, big_font, seed):
        # Access the unwrapped env to get exact physics parameters
        if isinstance(self.env, DummyVecEnv) or isinstance(self.env, VecNormalize):
            env_core = self.env.envs[0].unwrapped
        else:
            env_core = self.env.unwrapped

        wind = env_core.wind_force

        # ===== HUMAN TURN =====
        ball_x, basket_x = self.animate_shot(
            screen, self.human_force, self.human_angle, wind, env_core, "HUMAN"
        )

        if self.check_score(ball_x, basket_x):
            self.score_human += 1
            msg, color = "HUMAN SCORES!", GREEN
        else:
            msg, color = "Human Missed", RED
        
        self.draw_message(screen, big_font, msg, color)

        # ===== AI TURN =====
        # Reset with SAME seed ensures AI faces exact same wind/target
        obs, _ = self.env.reset(seed=seed)
        
        # Predict Action
        ai_action, _ = self.model.predict(obs, deterministic=True)

        # Decode Action (Normalization from -1..1 to Real Values)
        ai_force = ((ai_action[0] + 1) / 2) * 80 + 30
        ai_angle = ((ai_action[1] + 1) / 2) * 70 + 15

        ball_x, basket_x = self.animate_shot(
            screen, ai_force, ai_angle, wind, env_core, "AI AGENT"
        )

        if self.check_score(ball_x, basket_x):
            self.score_ai += 1
            msg, color = "AI SCORES!", CYAN
        else:
            msg, color = "AI Missed", WHITE

        self.draw_message(screen, big_font, msg, color)

    def draw_message(self, screen, font, text, color):
        box_surf = pygame.Surface((300, 80))
        box_surf.set_alpha(200)
        box_surf.fill(DARK_BG)
        screen.blit(box_surf, (250, 160))
        
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(400, 200))
        screen.blit(surf, rect)
        pygame.display.flip()
        time.sleep(1.5)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NeuroShot — Man vs Machine")

        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 28)
        big_font = pygame.font.Font(None, 40)

        running = True
        waiting_for_shot = True

        # Initial seed for the first round
        current_seed = random.randint(0, 100_000)
        self.env.reset(seed=current_seed)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    
                    # Spacebar triggers the duel
                    if event.key == pygame.K_SPACE and waiting_for_shot:
                        waiting_for_shot = False
                        self.handle_round(screen, big_font, current_seed)
                        
                        # Prepare next round
                        current_seed = random.randint(0, 100_000)
                        self.env.reset(seed=current_seed)
                        waiting_for_shot = True

            # Input Handling (Adjust Human Aim)
            keys = pygame.key.get_pressed()
            if waiting_for_shot:
                if keys[pygame.K_UP]: self.human_force = min(110, self.human_force + 0.5)
                if keys[pygame.K_DOWN]: self.human_force = max(30, self.human_force - 0.5)
                if keys[pygame.K_LEFT]: self.human_angle = min(85, self.human_angle + 0.5)
                if keys[pygame.K_RIGHT]: self.human_angle = max(15, self.human_angle - 0.5)

            # --- IDLE DRAWING ---
            screen.fill(DARK_BG)
            pygame.draw.line(screen, GRAY, (0, 350), (WIDTH, 350), 2)

            # Get current environment state for preview
            if isinstance(self.env, DummyVecEnv) or isinstance(self.env, VecNormalize):
                env_core = self.env.envs[0].unwrapped
            else:
                env_core = self.env.unwrapped

            # Draw Basket (Static Preview)
            basket_x = env_core.basket_x
            pygame.draw.rect(screen, CYAN, (int(basket_x), 340, 60, 10))
            pygame.draw.line(screen, RED, (int(basket_x) + 30, 340), (int(basket_x) + 30, 350), 2)
            
            # Draw Wind
            self.draw_wind_gauge(screen, env_core.wind_force)

            # UI Text
            screen.blit(font.render(f"Human: {self.score_human}", True, GREEN), (20, 20))
            screen.blit(font.render(f"AI: {self.score_ai}", True, CYAN), (20, 50))
            screen.blit(font.render(f"Force: {self.human_force:.1f}", True, YELLOW), (20, 340))
            screen.blit(font.render(f"Angle: {self.human_angle:.1f}", True, YELLOW), (20, 370))
            screen.blit(font.render("Arrow Keys to Aim, SPACE to Fire", True, WHITE), (WIDTH - 350, 370))

            pygame.display.flip()
            clock.tick(60)

        self.env.close()
        pygame.quit()

if __name__ == "__main__":
    # Point this to your trained model file
    # Ensure you have run train.py first to generate this!
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "neuro_v1_models", "models", "neuroshot_v1_ppo_model_1100000_steps.zip") 
    
    # If you haven't trained a model yet, this will crash.
    # Run the training script below first.
    if os.path.exists(MODEL_PATH):
        HumanVsAI(MODEL_PATH).run()
    else:
        print(f"Error: Could not find {MODEL_PATH}. Please train the agent first.")