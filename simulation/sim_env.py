from asyncio import wait

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from simulation.car import Car
from simulation.map import Map
from simulation.rendering import Renderer

class SimEnv(gym.Env):
    def __init__(self, human = True, screen_size=(600, 600)):
        super().__init__()

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),          # speed, path, distance
            high=np.array([1000.0, 1.0, 10000.0]),# reasonable caps
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),   # throttle, brake, steering
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.map = Map.map1()
        self.car = Car(1.0, self.map.start, heading=self.map.direction)

        self.human = human
        # Initialize Pygame
        if human:
            self.renderer = Renderer([self.car], self.map, screen_size)
        while human:
            self.renderer.render()
            throttle = 0.0
            brake = 0.0
            steering = 0.0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                throttle = 1.0
            if keys[pygame.K_DOWN]:
                brake = 1.0
            if keys[pygame.K_LEFT]:
                steering -= 1.0
            if keys[pygame.K_RIGHT]:
                steering += 1.0
            self.step((throttle, brake, steering))

        self.state = None

    def reset(self, seed=None, options=None):
        self.state = np.array([0, 0, 0, 0], dtype=np.float64)
        return self.state, {}
    
    def step(self, action):
        if self.human:
            self.renderer.render()

        throttle, brake, steering = action
    
        # Clamp just in case
        throttle = np.clip(throttle, 0, 1)
        brake = np.clip(brake, 0, 1)
        steering = np.clip(steering, -1, 1)

        self.car.update(throttle, brake, steering, 0.1)

        # Observations
        speed = self.car.speed
        path_covered = self.car.path_progress(self.map)
        (left, right, forward) = self.car.detect_obstacles(self.map, self.renderer)
        self.renderer.rays = self.car.rays

        obs = np.array([speed, path_covered, forward], dtype=np.float32)

        # Reward
        reward = speed * 0.1

        done = forward < self.car.size[0] / 2 # over-simplified collision detection

        return obs, reward, done, False, {}
