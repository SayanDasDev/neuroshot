import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
from typing import Optional, Any

class NeuroShotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # Define Action Space: [Force, Angle] (Normalized to -1, 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Define Observation Space: [Basket X, Basket Speed, Ball X, Ball Y]
        # Using Box space as per standard for continuous values
        self.observation_space = spaces.Box(
            low=np.array([0, -20, 0, 0], dtype=np.float32),
            high=np.array([800, 20, 800, 400], dtype=np.float32),
            dtype=np.float32
        )

        # Environment State Variables
        self.ball_pos = np.array([50.0, 350.0])
        self.basket_x = 0.0
        self.basket_speed = 0.0
        
        # Rendering bridge
        self.width, self.height = 800, 400
        self.screen = None
        self.clock = None

    def _get_obs(self):
        """Standard helper to return the current observation."""
        return np.array([
            self.basket_x, 
            self.basket_speed, 
            self.ball_pos[0], 
            self.ball_pos[1]
        ], dtype=np.float32)

    def _get_info(self):
        """Standard helper to return auxiliary info."""
        return {
            "basket_pos": self.basket_x,
            "ball_pos": self.ball_pos.tolist()
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize state using self.np_random (standard for reproducibility)
        self.basket_x = self.np_random.uniform(400, 750)
        self.basket_speed = self.np_random.uniform(-15, -5)
        self.ball_pos = np.array([50.0, 350.0])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(self.basket_x, [])

        return observation, info

    def step(self, action):
        # 1. Action Scaling
        v0 = ((action[0] + 1) / 2) * 80 + 30
        theta = np.radians(((action[1] + 1) / 2) * 70 + 15)
        
        g, t, dt = 9.8, 0, 0.08
        path = []
        
        # 2. Physics Simulation
        while self.ball_pos[1] <= 350:
            t += dt
            x = 50 + v0 * np.cos(theta) * t
            y = 350 - (v0 * np.sin(theta) * t - 0.5 * g * t**2)
            
            self.ball_pos = np.array([x, y])
            curr_basket_x = self.basket_x + (self.basket_speed * t)
            
            if self.render_mode == "human":
                path.append((int(x), int(y)))
                self._render_frame(curr_basket_x, path)

            if x > 800 or y > 400: break # Out of bounds

        # 3. Termination Logic
        final_basket_x = self.basket_x + (self.basket_speed * t)
        error = abs(self.ball_pos[0] - final_basket_x)
        
        # Reward design: Sparse + Shape
        terminated = True
        reward = 100.0 if error < 25 else -error * 0.1
        
        # Truncation (optional, e.g., for time limits)
        truncated = False 

        return self._get_obs(), reward, terminated, truncated, {"error": error}

    def _render_frame(self, basket_x, path):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("NeuroShot v1.0")
            self.clock = pygame.time.Clock()

        self.screen.fill((15, 15, 25))
        pygame.draw.line(self.screen, (200, 200, 200), (0, 350), (800, 350), 2)
        pygame.draw.rect(self.screen, (0, 255, 255), (int(basket_x), 342, 60, 10))
        
        if len(path) > 1:
            pygame.draw.lines(self.screen, (255, 255, 0), False, path, 1)
        
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), 6)
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None