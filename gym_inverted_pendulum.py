import pymunk
import pygame
import pymunk.pygame_util

import gym
from gym import spaces
import numpy as np


class InvertedPendulumEnv(gym.Env):
    def __init__(self):
        super(InvertedPendulumEnv, self).__init__()
        
        # Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, 98.1)
        
        # Environment constants
        self.dt = 1/60.0  # Simulation timestep
        self.force_mag = 500  # Magnitude of the applied force
        self.max_angle = np.pi / 2  # Terminate if angle exceeds this
        self.max_position = 400  # Terminate if base exceeds bounds
        
        # Create base
        base_size = (20, 20)
        self.base_body, self.base_shape = self.create_rect_obj(
            mass=5,
            size=base_size,
            pos=(400, 300),
            color=(0, 102, 153, 255)
        )
        self.space.add(self.base_body, self.base_shape)
        
        # Create prismatic joint for base movement
        static_body = self.space.static_body
        groove_start = (100, 300)
        groove_end = (700, 300)
        groove_joint = pymunk.GrooveJoint(static_body, self.base_body, groove_start, groove_end, (0, 0))
        self.space.add(groove_joint)
        
        # Create pendulum link
        link_size = (4, 200)
        self.link_body, self.link_shape = self.create_rect_obj(
            mass=2,
            size=link_size,
            pos=(400, 400),
            color=(255, 153, 51, 255)
        )
        self.space.add(self.link_body, self.link_shape)
        
        # Create revolute joint for pendulum rotation
        pivot_point = self.base_body.position
        revolute_joint = pymunk.PivotJoint(self.base_body, self.link_body, pivot_point)
        self.space.add(revolute_joint)
        
        # Observation space: [x, x_dot, theta, theta_dot]
        high = np.array([self.max_position, np.finfo(np.float32).max, self.max_angle, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Action space: [-1, 0, 1] for left, no force, right
        self.action_space = spaces.Discrete(3)

    def create_rect_obj(self, mass, size, pos, color=(0, 102, 153, 255)):
        width, height = size
        inertia = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, inertia)
        body.position = pos
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.color = color
        return body, shape

    def step(self, action):
        # Apply force based on the action
        if action == 0:  # Left
            self.base_body.apply_force_at_local_point((-self.force_mag, 0))
        elif action == 2:  # Right
            self.base_body.apply_force_at_local_point((self.force_mag, 0))
        
        # Step the simulation
        self.space.step(self.dt)
        
        # Get observations
        x = self.base_body.position.x
        x_dot = self.base_body.velocity.x
        theta = self.link_body.angle
        theta_dot = self.link_body.angular_velocity
        
        obs = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        
        # Check termination conditions
        done = bool(
            x < -self.max_position or x > self.max_position or
            abs(theta) > self.max_angle
        )
        
        # Reward is higher for keeping the pendulum upright and centered
        reward = 1.0 - (abs(theta) / self.max_angle)
        if done:
            reward -= 10.0
        
        return obs, reward, done, {}

    def reset(self):
        # Reset positions and velocities
        self.base_body.position = (400, 300)
        self.base_body.velocity = (0, 0)
        self.link_body.position = (400, 400)
        self.link_body.angle = 0
        self.link_body.angular_velocity = 0
        self.link_body.velocity = (0, 0)
        
        # Return initial observation
        x = self.base_body.position.x
        x_dot = self.base_body.velocity.x
        theta = self.link_body.angle
        theta_dot = self.link_body.angular_velocity
        
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def render(self, mode="human"):
        pass  # Implement visualization if necessary

    def close(self):
        pass  # Clean up resources if needed

if __name__ == "__main__":
    env = InvertedPendulumEnv()
    obs = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # Replace with a trained policy for better control
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    
