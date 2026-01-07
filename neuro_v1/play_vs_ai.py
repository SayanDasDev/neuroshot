import pygame
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os
import time
import random

# Fix path to ensure we can import 'env'
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

# STARTING POSITION (Matches env.GROUND_Y)
START_X = 50.0
START_Y = 350.0

def print_manual():
    print("\n" + "="*60)
    print("       🎮 NEUROSHOT: HUMAN VS AI - PLAYER MANUAL")
    print("="*60)
    print(" OBJECTIVE: Hit the moving CYAN BASKET with the ORANGE BALL.")
    print("            You are competing against a trained AI Agent.")
    print("-" * 60)
    print(" 🕹️  CONTROLS:")
    print("    [⬆️  UP ARROW]    : Increase Shot FORCE (Harder)")
    print("    [⬇️  DOWN ARROW]  : Decrease Shot FORCE (Softer)")
    print("    [⬅️  LEFT ARROW]  : Increase Angle (Aim HIGHER)")
    print("    [➡️  RIGHT ARROW] : Decrease Angle (Aim LOWER)")
    print("    [SPACEBAR]       : FIRE! (Ends your turn)")
    print("    [Q]              : Quit Simulation")
    # print("-" * 60)
    # print(" 📊 HUD GUIDE:")
    # print("    > WIND (Text): (+) pushes Right, (-) pushes Left.")
    # print("    > WHITE LINE: Shows your current aim direction.")
    print("="*60 + "\n")


class HumanVsAI:
    def __init__(self, model_path: str, stats_path: str = None):
        self.raw_env = gym.make("NeuroShot-v1", render_mode=None)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        print(f"Loading PPO model: {model_path}")
        self.model = PPO.load(model_path)
        
        if stats_path and os.path.exists(stats_path):
            print(f"Loading normalization stats: {stats_path}")
            self.env = DummyVecEnv([lambda: self.raw_env])
            self.env = VecNormalize.load(stats_path, self.env)
            self.env.training = False 
            self.env.norm_reward = False 
        else:
            self.env = self.raw_env

        self.human_angle = 45.0
        self.human_force = 70.0
        self.score_human = 0
        self.score_ai = 0

    def animate_shot(self, screen, v0, theta_deg, wind, env_core, who="HUMAN", t_start=0.0):
        """
        Simulates the shot frame-by-frame for visualization.
        """
        g = 9.8
        dt = 0.03  # CORRECTED: Matches your new Env file (0.03)
        t = 0.0

        theta = np.radians(theta_deg)
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)

        ball_x, ball_y = float(START_X), float(START_Y)

        A = env_core.amplitude
        f = env_core.frequency
        start_basket_x = env_core.basket_x
        basket_speed = env_core.basket_speed

        final_basket_x = start_basket_x
        running = True
        
        # Create Font
        font = pygame.font.Font(None, 28)

        while running:
            # 1. Physics Step
            t += dt
            ball_x = START_X + (vx * t) + (0.5 * wind * t * t)
            ball_y = START_Y - (vy * t - 0.5 * g * t * t)

            current_t = t + t_start
            
            ball_x = START_X + (vx * t) + (0.5 * wind * t * t)
            ball_y = START_Y - (vy * t - 0.5 * g * t * t)

            oscillation = A * np.sin(f * current_t)
            basket_x = start_basket_x + basket_speed * current_t + oscillation
            final_basket_x = basket_x

            # 2. Rendering
            screen.fill(DARK_BG)
            pygame.draw.line(screen, GRAY, (0, int(START_Y)), (WIDTH, int(START_Y)), 2)
            self.draw_wind_gauge(screen, wind)

            # Basket
            pygame.draw.rect(screen, CYAN, (int(basket_x), int(START_Y) - 10, 60, 10))
            pygame.draw.line(screen, RED, (int(basket_x) + 30, int(START_Y) - 10), (int(basket_x) + 30, int(START_Y)), 2)

            # Player Label
            screen.blit(font.render(f"Shooting: {who}", True, ORANGE), (330, 40))
            
            # Ball
            pygame.draw.circle(screen, ORANGE, (int(ball_x), int(ball_y)), 6)

            # === LOCKED TEXT POSITIONS (Firing Phase) ===
            screen.blit(font.render(f"Wind: {wind:+.2f}", True, YELLOW), (120, 20))
            screen.blit(font.render(f"Force: {v0:.1f}", True, YELLOW), (20, 360))
            screen.blit(font.render(f"Angle: {theta_deg:.1f}", True, YELLOW), (20, 380))
            # ============================================

            pygame.display.flip()
            time.sleep(0.01)

            # Check bounds
            if ball_y > START_Y or ball_x > WIDTH:
                running = False

        return ball_x, final_basket_x

    def draw_wind_gauge(self, screen, wind):
        center = (400, 30)
        end = (400 + int(wind * 20), 30)
        pygame.draw.line(screen, WHITE, (350, 30), (450, 30), 1) # Axis
        if abs(wind) > 0.1:
            color = GREEN if wind > 0 else RED
            pygame.draw.line(screen, color, center, end, 3)
            pygame.draw.circle(screen, color, end, 3)

    def check_score(self, ball_x, basket_x):
        basket_center = basket_x + 30
        return abs(ball_x - basket_center) < 25

    def handle_round(self, screen, big_font, seed, human_t_start=0.0):
        if isinstance(self.env, DummyVecEnv) or isinstance(self.env, VecNormalize):
            env_core = self.env.envs[0].unwrapped
        else:
            env_core = self.env.unwrapped

        wind = env_core.wind_force

        # ===== HUMAN TURN =====
        ball_x, basket_x = self.animate_shot(
            screen, self.human_force, self.human_angle, wind, env_core, "HUMAN", t_start=human_t_start
        )

        if self.check_score(ball_x, basket_x):
            self.score_human += 1
            msg, color = "HUMAN SCORES!", GREEN
        else:
            msg, color = "Human Missed", RED
        
        self.draw_message(screen, big_font, msg, color)

        # ===== AI TURN =====
        obs, _ = self.env.reset(seed=seed)
        ai_action, _ = self.model.predict(obs, deterministic=True)
        ai_force = ((ai_action[0] + 1) / 2) * 80 + 30
        ai_angle = ((ai_action[1] + 1) / 2) * 70 + 15

        ball_x, basket_x = self.animate_shot(
            screen, ai_force, ai_angle, wind, env_core, "AI AGENT", t_start=0.0
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

        current_seed = random.randint(0, 100_000)
        self.env.reset(seed=current_seed)
        
        idle_time = 0.0

        while running:
             # Track time for basket animation
            dt_s = clock.get_time() / 1000.0
            if waiting_for_shot:
                idle_time += dt_s

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_SPACE and waiting_for_shot:
                        waiting_for_shot = False
                        self.handle_round(screen, big_font, current_seed, human_t_start=idle_time)
                        current_seed = random.randint(0, 100_000)
                        self.env.reset(seed=current_seed)
                        waiting_for_shot = True
                        idle_time = 0.0

            # Input Handling
            keys = pygame.key.get_pressed()
            if waiting_for_shot:
                if keys[pygame.K_UP]: self.human_force = min(110, self.human_force + 0.5)
                if keys[pygame.K_DOWN]: self.human_force = max(30, self.human_force - 0.5)
                if keys[pygame.K_LEFT]: self.human_angle = min(85, self.human_angle + 0.5)
                if keys[pygame.K_RIGHT]: self.human_angle = max(15, self.human_angle - 0.5)

            # --- IDLE DRAWING (AIMING PHASE) ---
            screen.fill(DARK_BG)
            pygame.draw.line(screen, GRAY, (0, int(START_Y)), (WIDTH, int(START_Y)), 2) 

            if isinstance(self.env, DummyVecEnv) or isinstance(self.env, VecNormalize):
                env_core = self.env.envs[0].unwrapped
            else:
                env_core = self.env.unwrapped

            # Basket
            # Basket (Animated)
            oscillation = env_core.amplitude * np.sin(env_core.frequency * idle_time)
            curr_basket_x = env_core.basket_x + (env_core.basket_speed * idle_time) + oscillation
            
            pygame.draw.rect(screen, CYAN, (int(curr_basket_x), int(START_Y) - 10, 60, 10))
            pygame.draw.line(screen, RED, (int(curr_basket_x) + 30, int(START_Y) - 10), (int(curr_basket_x) + 30, int(START_Y)), 2)
            
            # Wind Gauge
            self.draw_wind_gauge(screen, env_core.wind_force)

            # Ball & Aim Line
            pygame.draw.circle(screen, ORANGE, (int(START_X), int(START_Y)), 6)
            rad_angle = np.radians(self.human_angle)
            barrel_end_x = START_X + 40 * np.cos(rad_angle)
            barrel_end_y = START_Y - 40 * np.sin(rad_angle)
            pygame.draw.line(screen, WHITE, (START_X, START_Y), (barrel_end_x, barrel_end_y), 2)

            # Scores
            screen.blit(font.render(f"Human: {self.score_human}", True, GREEN), (20, 20))
            screen.blit(font.render(f"AI: {self.score_ai}", True, CYAN), (20, 50))
            
            # === LOCKED TEXT POSITIONS (Aiming Phase) ===
            screen.blit(font.render(f"Wind: {env_core.wind_force:+.2f}", True, YELLOW), (120, 20))
            screen.blit(font.render(f"Force: {self.human_force:.1f}", True, YELLOW), (20, 360))
            screen.blit(font.render(f"Angle: {self.human_angle:.1f}", True, YELLOW), (20, 380))
            # ============================================

            screen.blit(font.render("Arrow Keys to Aim, SPACE to Fire", True, WHITE), (WIDTH - 350, 370))

            pygame.display.flip()
            clock.tick(60)

        self.env.close()
        pygame.quit()

if __name__ == "__main__":
    # ⚠️ UPDATE THIS PATH to your actual model file
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "neuro_v1_models", "models", "neuroshot_v1_ppo_model_1100000_steps.zip") 
    
    print_manual()

    if os.path.exists(MODEL_PATH):
        input("Press Enter to launch simulation...") 
        HumanVsAI(MODEL_PATH).run()
    else:
        print(f"Error: Could not find {MODEL_PATH}.")
        print("Please check the folder path at the bottom of the script!")