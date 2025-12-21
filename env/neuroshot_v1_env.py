import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
from typing import Optional

class NeuroShotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # Actions: [Force, Angle]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observations: [Basket X, Basket Speed, Wind, Ball X, Ball Y]
        # We added 'Wind' to the state so the AI can learn to compensate.
        self.observation_space = spaces.Box(
            low=np.array([0, -20, -5, 0, 0], dtype=np.float32),
            high=np.array([800, 20, 5, 800, 400], dtype=np.float32),
            dtype=np.float32
        )

        self.width, self.height = 800, 400
        self.screen = None
        self.clock = None

    def _get_obs(self):
        return np.array([
            self.basket_x, 
            self.basket_speed, 
            self.wind_force,
            self.ball_pos[0], 
            self.ball_pos[1]
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Randomized Environment Factors
        self.basket_x = self.np_random.uniform(400, 700)
        self.basket_speed = self.np_random.uniform(-10, -5)
        self.wind_force = self.np_random.uniform(-3, 3) # Random horizontal wind
        self.ball_pos = np.array([50.0, 350.0])
        
        # Oscillation parameters for non-linear movement
        self.amplitude = self.np_random.uniform(20, 50) 
        self.frequency = self.np_random.uniform(0.5, 2.0)

        return self._get_obs(), {}

    def step(self, action):
        # 1. Decode Actions
        v0 = ((action[0] + 1) / 2) * 80 + 30
        theta = np.radians(((action[1] + 1) / 2) * 70 + 15)
        
        g, t, dt = 9.8, 0, 0.08
        path = []
        
        # Initial Velocities
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)

        # 2. Physics Simulation with Wind and Acceleration
        while self.ball_pos[1] <= 350:
            t += dt
            
            # Wind affects horizontal velocity over time (Acceleration)
            # x = x0 + vx*t + 0.5 * a * t^2
            curr_x = 50 + (vx * t) + (0.5 * self.wind_force * t**2)
            curr_y = 350 - (vy * t - 0.5 * g * t**2)
            
            self.ball_pos = np.array([curr_x, curr_y])
            
            # Non-Linear Basket Movement: Linear Speed + Sine Wave Oscillation
            # This makes the basket "wobble" or speed up/slow down unpredictably
            oscillation = self.amplitude * np.sin(self.frequency * t)
            curr_basket_x = self.basket_x + (self.basket_speed * t) + oscillation
            
            if self.render_mode == "human":
                path.append((int(curr_x), int(curr_y)))
                self._render_frame(curr_basket_x, path)

            if curr_x > 800 or curr_y > 400: break

        # 3. Final State and Reward
        final_basket_x = self.basket_x + (self.basket_speed * t) + (self.amplitude * np.sin(self.frequency * t))
        error = abs(self.ball_pos[0] - final_basket_x)
        
        reward = 100.0 if error < 25 else -error * 0.1
        
        return self._get_obs(), reward, True, False, {"error": error, "wind": self.wind_force}

    def _render_frame(self, basket_x, path):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("NeuroShot v1 (Complex)")
            self.clock = pygame.time.Clock()

        # Darker background for better contrast
        self.screen.fill((20, 10, 30)) 
        
        # --- ENHANCED WIND GAUGE ---
        gauge_center_x = 400
        gauge_y = 50
        max_bar_width = 80 # The "baseline" length
        
        # 1. Draw Static Baseline (The "Neutral" reference)
        # This helps you see how far from zero the wind actually is
        pygame.draw.line(self.screen, (60, 60, 90), 
                         (gauge_center_x - max_bar_width, gauge_y), 
                         (gauge_center_x + max_bar_width, gauge_y), 1)
        pygame.draw.circle(self.screen, (200, 200, 200), (gauge_center_x, gauge_y), 2) # Center point
        
        # 2. Calculate Arrow Tip
        # Scaling wind_force (usually -3 to 3) to pixels
        tip_x = gauge_center_x + int(self.wind_force * 25)
        
        # 3. Draw the Wind Arrow and Head
        if abs(self.wind_force) > 0.1:
            wind_color = (100, 255, 100)
            # Arrow Shaft
            pygame.draw.line(self.screen, wind_color, (gauge_center_x, gauge_y), (tip_x, gauge_y), 3)
            
            # Arrow Head (Triangle)
            direction = 1 if self.wind_force > 0 else -1
            head_size = 10
            points = [
                (tip_x, gauge_y), # Tip
                (tip_x - (direction * head_size), gauge_y - 6), # Top wing
                (tip_x - (direction * head_size), gauge_y + 6)  # Bottom wing
            ]
            pygame.draw.polygon(self.screen, wind_color, points)

        # --- ENVIRONMENT RENDERING ---
        # Ground
        pygame.draw.line(self.screen, (150, 150, 150), (0, 350), (800, 350), 2)
        
        # Basket (Cyan)
        pygame.draw.rect(self.screen, (0, 255, 255), (int(basket_x), 342, 60, 10))
        
        # Path and Ball
        if len(path) > 1:
            pygame.draw.lines(self.screen, (255, 100, 255), False, path, 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), 6)
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen: pygame.quit()