# /home/sysadm/Music/neuroshot/env/neuroshot_v1_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional

class NeuroShotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # Actions: [Force, Angle] (Inputs are -1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observations: Normalized [Basket X, Basket Speed, Wind, Amp, Freq, Ball X, Ball Y]
        # We use -inf to inf to avoid warnings, but values will mostly be between -1 and 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # World Constants
        self.WIDTH, self.HEIGHT = 800, 400
        self.GROUND_Y = 350.0

        self.screen = None
        self.clock = None

    def _get_obs(self):
        """
        Returns normalized observations to help the Neural Network learn faster.
        Scales values to approximately 0.0 - 1.0 or -1.0 - 1.0 range.
        """
        return np.array([
            self.basket_x / self.WIDTH,       # 0.0 to 1.0
            self.basket_speed / 20.0,         # approx -1.0 to 0.0
            self.wind_force / 5.0,            # approx -0.6 to 0.6
            self.amplitude / 50.0,            # 0.4 to 1.0
            self.frequency / 2.0,             # 0.25 to 1.0
            self.ball_pos[0] / self.WIDTH,    # 0.0 to 1.0
            self.ball_pos[1] / self.HEIGHT    # 0.0 to 1.0
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # 🔧 Fix: Deterministic Reset for Testing
        # If options={"fixed": True} is passed, we use the same values every time.
        fixed_mode = options and options.get("fixed")

        if fixed_mode:
            self.basket_x = 600.0
            self.basket_speed = -8.0
            self.wind_force = 0.0
            self.amplitude = 30.0
            self.frequency = 1.0
        else:
            self.basket_x = self.np_random.uniform(400, 700)
            self.basket_speed = self.np_random.uniform(-10, -5)
            self.wind_force = self.np_random.uniform(-3, 3)
            self.amplitude = self.np_random.uniform(20, 50)
            self.frequency = self.np_random.uniform(0.5, 2.0)

        self.ball_pos = np.array([50.0, self.GROUND_Y])
        return self._get_obs(), {}

    def step(self, action):
        # 1. Action Decoding (Scale -1..1 to Physics Values)
        # Force (v0): 30 to 110
        v0 = ((action[0] + 1) / 2) * 80 + 30
        # Angle (theta): 15 to 85 degrees
        theta = np.radians(((action[1] + 1) / 2) * 70 + 15)

        # Physics Constants
        g = 9.8
        dt = 0.03  # 🔧 Fix: Smaller time step for smoother physics (was 0.08)
        t = 0.0
        
        # Initial Components
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        
        # 🔧 Fix: Use local variables for calculation to prevent "drifting" glitches
        ball_x, ball_y = 50.0, self.GROUND_Y
        path = []

        # 2. Physics Simulation Loop
        # 🔧 Fix: Better loop condition (check both Y and X bounds)
        while ball_y <= self.GROUND_Y and ball_x <= self.WIDTH:
            t += dt
            
            # Projectile Motion with Wind Acceleration
            # x = x0 + vx*t + 0.5 * wind * t^2
            ball_x = 50.0 + (vx * t) + (0.5 * self.wind_force * t**2)
            
            # y = y0 - (vy*t - 0.5 * g * t^2)  (Minus because Y grows downwards)
            ball_y = self.GROUND_Y - (vy * t - 0.5 * g * t**2)
            
            # Update self.ball_pos ONLY for rendering/obs
            self.ball_pos = np.array([ball_x, ball_y])
            
            # Basket Movement (Linear + Oscillation)
            oscillation = self.amplitude * np.sin(self.frequency * t)
            curr_basket_x = self.basket_x + (self.basket_speed * t) + oscillation
            
            if self.render_mode == "human":
                path.append((int(ball_x), int(ball_y)))
                self._render_frame(curr_basket_x, path)

            # Break early if ball hits the ground
            if ball_y > self.GROUND_Y:
                break

        # 3. Calculate Reward
        # Where is the basket at the exact moment of impact?
        final_basket_x = self.basket_x + (self.basket_speed * t) + (self.amplitude * np.sin(self.frequency * t))
        error = abs(ball_x - final_basket_x)
        
        # 🔧 Fix: Improved Reward Shaping
        # Give a small penalty based on distance to guide the AI
        reward = -error * 0.05
        if error < 25: # Hit threshold
            reward += 100.0
        
        terminated = True
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {"error": error, "wind": self.wind_force}

    def _render_frame(self, basket_x, path):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("NeuroShot v1 (Corrected)")
            self.clock = pygame.time.Clock()

        # Darker background
        self.screen.fill((20, 10, 30)) 
        
        # --- WIND GAUGE ---
        center_x, center_y = 400, 50
        # Draw baseline
        pygame.draw.line(self.screen, (60, 60, 90), (center_x - 80, center_y), (center_x + 80, center_y), 1)
        pygame.draw.circle(self.screen, (200, 200, 200), (center_x, center_y), 2)
        
        # Draw Arrow
        tip_x = center_x + int(self.wind_force * 25)
        if abs(self.wind_force) > 0.1:
            pygame.draw.line(self.screen, (100, 255, 100), (center_x, center_y), (tip_x, center_y), 3)

        # --- ENVIRONMENT ---
        # Ground
        pygame.draw.line(self.screen, (150, 150, 150), (0, int(self.GROUND_Y)), (self.WIDTH, int(self.GROUND_Y)), 2)
        
        # Basket (Cyan)
        pygame.draw.rect(self.screen, (0, 255, 255), (int(basket_x), int(self.GROUND_Y) - 8, 60, 10))
        
        # Path and Ball
        if len(path) > 1:
            pygame.draw.lines(self.screen, (255, 100, 255), False, path, 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), 6)
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen: pygame.quit()