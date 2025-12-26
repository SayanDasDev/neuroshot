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

        # Observation: [Basket X, Basket Speed, Wind, Ball X, Ball Y]
        # Using -inf to inf to avoid warnings, values will be normalized in _get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.ball_pos = np.array([50.0, 350.0])
        self.width, self.height = 800, 400
        self.screen = None
        self.clock = None

    def _get_obs(self):
        """
        Returns normalized observations to help the Neural Network learn faster.
        Scales values to approximately 0.0 - 1.0 or -1.0 - 1.0 range.
        """
        return np.array([
            self.basket_x / self.width,        # 0.0 to 1.0
            self.basket_speed / 20.0,          # approx -1.0 to 0.0
            self.wind_force / 5.0,             # approx -1.0 to 1.0
            self.ball_pos[0] / self.width,     # 0.0 to 1.0
            self.ball_pos[1] / self.height     # 0.0 to 1.0
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Fixed mode for deterministic testing
        fixed_mode = options and options.get("fixed")
        
        if fixed_mode:
            self.basket_x = 600.0
            self.basket_speed = -8.0
            self.wind_force = 0.0
        else:
            self.basket_x = self.np_random.uniform(400, 700)
            self.basket_speed = self.np_random.uniform(-10, -5)
            self.wind_force = self.np_random.uniform(-4, 4)
        
        self.ball_pos = np.array([50.0, 350.0])
        return self._get_obs(), {}

    def step(self, action):
        # Action Decoding (Scale -1..1 to Physics Values)
        v0 = ((action[0] + 1) / 2) * 80 + 30
        theta = np.radians(((action[1] + 1) / 2) * 70 + 15)
        vx, vy = v0 * np.cos(theta), v0 * np.sin(theta)
        
        # Physics Constants
        g = 9.8
        dt = 0.03  # Smaller time step for smoother physics
        t = 0.0
        path = []
        
        # Use local variables to prevent loop condition bugs
        ball_x, ball_y = 50.0, 350.0

        while ball_y <= 350 and ball_x <= 800:
            t += dt
            # PHYSICS WITH WIND ACCELERATION
            ball_x = 50 + (vx * t) + (0.5 * self.wind_force * t**2)
            ball_y = 350 - (vy * t - 0.5 * g * t**2)
            
            # Update self.ball_pos for rendering/observation
            self.ball_pos = np.array([ball_x, ball_y])

            # LINEAR BASKET MOVEMENT
            curr_basket_x = self.basket_x + (self.basket_speed * t)

            if self.render_mode == "human":
                path.append((int(ball_x), int(ball_y)))
                self._render_frame(curr_basket_x, path)
            
            if ball_y > 350:
                break

        final_basket_x = self.basket_x + (self.basket_speed * t)
        error = abs(ball_x - final_basket_x)
        
        # Improved reward shaping - always provide gradient signal
        reward = -error * 0.05
        if error < 25:  # Hit threshold
            reward += 100.0
        
        return self._get_obs(), reward, True, False, {"error": error, "wind": self.wind_force}

    def _render_frame(self, basket_x, path):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("NeuroShot v0.2 (Wind)")
            self.clock = pygame.time.Clock()

        self.screen.fill((15, 25, 15)) 
        
        # --- WIND GAUGE LOGIC ---
        center_x, center_y = 400, 40
        max_visual_length = 60 # Maximum pixels for max wind
        
        # 1. Draw a dim baseline (Neutral Zone)
        pygame.draw.line(self.screen, (50, 100, 50), (center_x - max_visual_length, center_y), 
                         (center_x + max_visual_length, center_y), 1)
        pygame.draw.circle(self.screen, (200, 200, 200), (center_x, center_y), 2) # Center dot
        
        # 2. Calculate arrow tip position
        # wind_force is typically -4 to 4, so we scale it
        tip_x = center_x + int(self.wind_force * 15) 
        
        # 3. Draw the main Wind Arrow
        if abs(self.wind_force) > 0.1:
            color = (100, 255, 100)
            pygame.draw.line(self.screen, color, (center_x, center_y), (tip_x, center_y), 3)
            
            # 4. Draw the Arrow Head (Triangle)
            direction = 1 if self.wind_force > 0 else -1
            head_size = 8
            points = [
                (tip_x, center_y),
                (tip_x - direction * head_size, center_y - 5),
                (tip_x - direction * head_size, center_y + 5)
            ]
            pygame.draw.polygon(self.screen, color, points)

        # --- REST OF RENDERING ---
        # Ground
        pygame.draw.line(self.screen, (200, 200, 200), (0, 350), (800, 350), 2)
        # Basket
        pygame.draw.rect(self.screen, (0, 255, 255), (int(basket_x), 342, 60, 10))
        # Path and Ball
        if len(path) > 1:
            pygame.draw.lines(self.screen, (100, 255, 100), False, path, 1)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), 6)
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen: pygame.quit()