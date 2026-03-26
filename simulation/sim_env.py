import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from simulation.car import Car
from simulation.map import Map
from simulation.rendering import Renderer

class SimEnv(gym.Env):
    def __init__(self, human=False, screen_size=(600, 600), max_episode_steps=20000):
        super().__init__()

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),          # speed, path, distance
            high=np.array([1000.0, 1.0, 10000.0], dtype=np.float32),  # reasonable caps
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),   # throttle, brake, steering
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.map = Map.map1()
        self.car = Car(1.0, self.map.start, heading=self.map.direction)
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.last_progress = 0.0
        self.no_progress_steps = 0

        self.human = human
        self.renderer = None
        # Initialize Pygame
        if human:
            self.renderer = Renderer([self.car], self.map, screen_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car = Car(1.0, self.map.start, heading=self.map.direction)
        self.episode_steps = 0
        self.last_progress = self.car.path_progress(self.map)
        self.no_progress_steps = 0

        if self.renderer is not None:
            self.renderer.cars = [self.car]

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        speed = self.car.speed
        path_covered = self.car.path_progress(self.map)
        (_, _, forward) = self.car.detect_obstacles(self.map, self.renderer)
        forward = np.clip(np.nan_to_num(forward, posinf=10000.0), 0.0, 10000.0)

        if self.renderer is not None:
            self.renderer.rays = self.car.rays

        return np.array([speed, path_covered, forward], dtype=np.float32)

    def step(self, action):
        throttle, brake, steering = np.asarray(action, dtype=np.float32)

        # Clamp just in case
        throttle = np.clip(throttle, 0, 1)
        brake = np.clip(brake, 0, 1)
        steering = np.clip(steering, -1, 1)

        self.car.update(throttle, brake, steering, 0.5)

        obs = self._get_observation()
        speed, path_covered, forward = obs

        self.episode_steps += 1
        progress_delta = path_covered - self.last_progress
        self.last_progress = path_covered

        reward = (progress_delta * 200.0) + (speed * 0.005)
        if progress_delta < 0:
            reward += progress_delta * 50.0

        # Penalize conflicting longitudinal controls.
        if throttle > 0.2 and brake > 0.2:
            reward -= 0.1 * float(min(throttle, brake))

        # Penalize getting stuck with little movement/progress for a long time.
        if abs(progress_delta) < 1e-4 and speed < 0.2:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        if self.no_progress_steps > 50:
            reward -= 0.2

        collided = forward < (self.car.size[0] / 2.0)
        finished_lap = path_covered >= 0.999
        terminated = collided or finished_lap
        truncated = self.episode_steps >= self.max_episode_steps

        if collided:
            reward -= 100.0
        if finished_lap:
            reward += 300.0

        if self.renderer is not None:
            self.renderer.render()

        info = {
            "progress": float(path_covered),
            "speed": float(speed),
            "finished_lap": bool(finished_lap),
            "collided": bool(collided),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.renderer is not None:
            self.renderer.render()

    def close(self):
        if self.renderer is not None:
            pygame.quit()
