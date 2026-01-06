import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional

class NeuroShotEnv(gym.Env):
    # 1. ADDED: 'rgb_array' to metadata so the recorder works
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # Actions: [Force, Angle] (Inputs are -1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observations
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # World Constants
        self.WIDTH, self.HEIGHT = 800, 400
        self.GROUND_Y = 350.0

        self.screen = None
        self.clock = None

    def _get_obs(self):
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
        v0 = ((action[0] + 1) / 2) * 80 + 30
        theta = np.radians(((action[1] + 1) / 2) * 70 + 15)

        # Physics Constants (KEPT YOUR VALUES)
        g = 9.8
        dt = 0.03  # Keeping 0.03 as you requested
        t = 0.0
        
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        
        ball_x, ball_y = 50.0, self.GROUND_Y
        path = []
        frames = [] # 2. ADDED: List to store video frames

        while ball_y <= self.GROUND_Y and ball_x <= self.WIDTH:
            t += dt
            
            ball_x = 50.0 + (vx * t) + (0.5 * self.wind_force * t**2)
            ball_y = self.GROUND_Y - (vy * t - 0.5 * g * t**2)
            self.ball_pos = np.array([ball_x, ball_y])
            
            oscillation = self.amplitude * np.sin(self.frequency * t)
            curr_basket_x = self.basket_x + (self.basket_speed * t) + oscillation
            
            # 3. ADDED: Logic to capture frames if recording
            path.append((int(ball_x), int(ball_y)))
            
            if self.render_mode == "human":
                self._render_frame(curr_basket_x, path)
            elif self.render_mode == "rgb_array":
                # Capture frame and add to list
                frame = self._render_frame(curr_basket_x, path, return_rgb=True)
                frames.append(frame)

            if ball_y > self.GROUND_Y:
                break

        final_basket_x = self.basket_x + (self.basket_speed * t) + (self.amplitude * np.sin(self.frequency * t))
        target_center_x = final_basket_x + 30.0  
        error = abs(ball_x - target_center_x)
        
        reward = -error * 0.05
        if error < 25: 
            reward += 100.0
        
        terminated = True
        truncated = False
        
        # 4. ADDED: Return frames in the info dict
        return self._get_obs(), reward, terminated, truncated, {"error": error, "wind": self.wind_force, "frames": frames}

    def _render_frame(self, basket_x, path, return_rgb=False):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            else:
                # 5. ADDED: Create hidden surface for video recording
                self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
            
            if self.render_mode == "human":
                pygame.display.set_caption("NeuroShot v1")
                self.clock = pygame.time.Clock()

        self.screen.fill((20, 10, 30)) 
        
        # Wind Gauge
        center_x, center_y = 400, 50
        pygame.draw.line(self.screen, (60, 60, 90), (center_x - 80, center_y), (center_x + 80, center_y), 1)
        pygame.draw.circle(self.screen, (200, 200, 200), (center_x, center_y), 2)
        tip_x = center_x + int(self.wind_force * 25)
        if abs(self.wind_force) > 0.1:
            pygame.draw.line(self.screen, (100, 255, 100), (center_x, center_y), (tip_x, center_y), 3)

        # Environment
        pygame.draw.line(self.screen, (150, 150, 150), (0, int(self.GROUND_Y)), (self.WIDTH, int(self.GROUND_Y)), 2)
        pygame.draw.rect(self.screen, (0, 255, 255), (int(basket_x), int(self.GROUND_Y) - 8, 60, 10))
        
        if len(path) > 1:
            pygame.draw.lines(self.screen, (255, 100, 255), False, path, 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), 6)
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)
        
        # 6. ADDED: Return RGB array for video processing
        if return_rgb:
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        if self.screen: pygame.quit()