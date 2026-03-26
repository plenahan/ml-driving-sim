import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation.car import Car
from simulation.map import Map
from simulation.rendering import Renderer

class SimEnv(gym.Env):
    def __init__(self, human = True, screen_size=(600, 600), max_episode_steps=1000):
        super().__init__()

        self.max_speed_observation = 10.0
        self.max_sensor_distance = 1000.0
        self.stuck_speed_threshold = 0.05
        self.stuck_steps_limit = 40
        self.stuck_grace_steps = 20

        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),   # acceleration intent, steering
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.map = Map.map1()
        self.car = Car(1.0, self.map.start, heading=self.map.direction)
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.prev_path_covered = 0.0
        self.stagnant_steps = 0

        self.human = human
        self.renderer = None
        # Initialize Pygame
        if human:
            self.renderer = Renderer([self.car], self.map, screen_size)

    def reset(self, seed=None, options=None):
        self.car = Car(1.0, self.map.start, heading=self.map.direction)
        self.episode_steps = 0
        self.prev_path_covered = 0.0
        self.stagnant_steps = 0

        if self.renderer is not None:
            self.renderer.cars = [self.car]

        obs = self.get_observation()
        return obs


    def _get_raw_metrics(self):
        speed = self.car.speed
        path_covered = self.car.path_progress(self.map)
        (left, forward, right) = self.car.detect_obstacles(self.map, self.renderer)
        left = np.clip(np.nan_to_num(left, posinf=self.max_sensor_distance), 0.0, self.max_sensor_distance)
        forward = np.clip(np.nan_to_num(forward, posinf=self.max_sensor_distance), 0.0, self.max_sensor_distance)
        right = np.clip(np.nan_to_num(right, posinf=self.max_sensor_distance), 0.0, self.max_sensor_distance)

        return speed, path_covered, left, forward, right

    def get_observation(self, raw_metrics=None):
        if raw_metrics is None:
            raw_metrics = self._get_raw_metrics()

        speed, path_covered, left, forward, right = raw_metrics

        if self.renderer is not None:
            self.renderer.rays = self.car.rays

        normalized_speed = np.clip(speed / self.max_speed_observation, -1.0, 1.0)
        normalized_left = left / self.max_sensor_distance
        normalized_forward = forward / self.max_sensor_distance
        normalized_right = right / self.max_sensor_distance

        return np.array(
            [normalized_speed, path_covered, normalized_left, normalized_forward, normalized_right],
            dtype=np.float32,
        )
    
    def step(self, action):
        if self.human:
            self.renderer.render()

        acceleration_intent, steering = action
    
        # Clamp just in case
        acceleration_intent = np.clip(acceleration_intent, -1, 1)
        steering = np.clip(steering, -1, 1)
        throttle = max(acceleration_intent, 0.0)
        brake = max(-acceleration_intent, 0.0)

        self.car.update(throttle, brake, steering, 0.5)

        speed, path_covered, left, forward, right = self._get_raw_metrics()
        obs = self.get_observation((speed, path_covered, left, forward, right))

        self.episode_steps += 1
        delta_path_covered = path_covered - self.prev_path_covered
        self.prev_path_covered = path_covered

        # Reward progress and sustained movement, while discouraging inactivity.
        reward = delta_path_covered * 500.0
        if abs(speed) < self.stuck_speed_threshold:
            self.stagnant_steps += 1
            reward -= 0.2
        else:
            self.stagnant_steps = 0

        collided = forward < (self.car.size[0] / 2.0)
        finished_lap = path_covered >= 0.999
        terminated = collided or finished_lap
        truncated = self.episode_steps >= self.max_episode_steps

        if collided:
            reward -= 200

        if self.renderer is not None:
            self.renderer.render()

        return obs, float(reward), terminated, truncated, {}

    def render(self):
        if self.renderer is not None:
            self.renderer.render()
