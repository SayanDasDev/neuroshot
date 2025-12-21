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
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Observation: [Basket X, Basket Speed, Amplitude, Frequency, Ball X, Ball Y]
        self.observation_space = spaces.Box(
            low=np.array([0, -20, 20, 0.5, 0, 0], dtype=np.float32),
            high=np.array([800, 20, 50, 2.0, 800, 400], dtype=np.float32),
            dtype=np.float32
        )

        self.ball_pos = np.array([50.0, 350.0])
        self.width, self.height = 800, 400
        self.screen = None
        self.clock = None

    def _get_obs(self):
        return np.array([
            self.basket_x, self.basket_speed, 
            self.amplitude, self.frequency,
            self.ball_pos[0], self.ball_pos[1]
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.basket_x = self.np_random.uniform(400, 700)
        self.basket_speed = self.np_random.uniform(-10, -5)
        
        # Non-Linear Parameters
        self.amplitude = self.np_random.uniform(20, 50)
        self.frequency = self.np_random.uniform(0.5, 2.0)
        
        self.ball_pos = np.array([50.0, 350.0])
        return self._get_obs(), {}

    def step(self, action):
        v0 = ((action[0] + 1) / 2) * 80 + 30
        theta = np.radians(((action[1] + 1) / 2) * 70 + 15)
        g, t, dt = 9.8, 0, 0.08
        path = []

        while self.ball_pos[1] <= 350:
            t += dt
            x = 50 + v0 * np.cos(theta) * t
            y = 350 - (v0 * np.sin(theta) * t - 0.5 * g * t**2)
            self.ball_pos = np.array([x, y])

            # OSCILLATION LOGIC
            oscillation = self.amplitude * np.sin(self.frequency * t)
            curr_basket_x = self.basket_x + (self.basket_speed * t) + oscillation

            if self.render_mode == "human":
                path.append((int(x), int(y)))
                self._render_frame(curr_basket_x, path)
            if x > 800 or y > 400: break

        final_basket_x = self.basket_x + (self.basket_speed * t) + (self.amplitude * np.sin(self.frequency * t))
        error = abs(self.ball_pos[0] - final_basket_x)
        reward = 100.0 if error < 25 else -error * 0.1
        
        return self._get_obs(), reward, True, False, {"error": error}

    def _render_frame(self, basket_x, path):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("NeuroShot v0.1 (Non-Linear)")
            self.clock = pygame.time.Clock()

        self.screen.fill((25, 15, 15)) # Dark Red Tint
        pygame.draw.line(self.screen, (200, 200, 200), (0, 350), (800, 350), 2)
        pygame.draw.rect(self.screen, (0, 255, 255), (int(basket_x), 342, 60, 10))
        if len(path) > 1:
            pygame.draw.lines(self.screen, (255, 255, 0), False, path, 1)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), 6)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen: pygame.quit()