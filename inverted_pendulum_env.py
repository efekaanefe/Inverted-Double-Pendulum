import gym
from gym import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame


class InvertedPendulumEnv(gym.Env):
    def __init__(self, 
                 gravity=98.1, 
                 dt=1/60.0, 
                 force_mag=500, 
                 base_size=(20, 20),
                 base_mass=5, 
                 link_size=(4, 200), 
                 link_mass = 2,
                 groove_length = 600,
                 initial_angle=0,
                 max_angle=np.pi/2,
                 max_position=400,
                 max_steps = 1000):
       
        super(InvertedPendulumEnv, self).__init__()
        
        # Store parameters
        self.gravity = gravity
        self.dt = dt
        self.force_mag = force_mag
        self.base_size = base_size
        self.base_mass = base_mass
        self.link_size = link_size
        self.link_mass = link_mass
        self.groove_length = groove_length
        self.initial_angle = initial_angle
        self.max_angle = max_angle
        self.max_position = max_position
        self.max_steps = max_steps

        self.groove_y = 0

        # Pygame setup
        self.screen = None
        self.clock = None
        self.window_width = 800
        self.window_height = 600
        self.render_scale = 1.0  # Scale for rendering
        
        self.create_pymunk_env()
        
        # Observation space: [x, x_dot, theta, theta_dot]
        high = np.array([self.max_position, np.finfo(np.float32).max, self.max_angle, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Action space: [-1, 0, 1] for left, no force, right
        self.action_space = spaces.Discrete(3)

    def create_pymunk_env(self):
        self.steps = 0

        # Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity)

        base_collision_type = 1
        rotating_collision_type = 2
        
        # Create base
        self.base_body, self.base_shape = self.create_rect_obj(
            mass=self.base_mass,
            size=self.base_size,
            pos=(self.groove_length/2, self.groove_y),
            color=(0, 102, 153, 255)
        )
        self.base_shape.collision_type = base_collision_type

        self.space.add(self.base_body, self.base_shape)

        # Create prismatic joint for base movement
        static_body = self.space.static_body
        groove_start = (0, self.groove_y)
        groove_end = (self.groove_length, self.groove_y)
        groove_joint = pymunk.GrooveJoint(static_body, self.base_body, groove_start, groove_end, (0, 0))
        self.space.add(groove_joint)
        
        # Create pendulum link
        self.link_body, self.link_shape = self.create_rect_obj(
            mass=self.link_mass,
            size=self.link_size,
            pos=(self.groove_length/2, self.groove_y - self.link_size[1]/2),
            color=(255, 153, 51, 255)
        )
        # self.link_body.angle = self.initial_angle

        self.link_shape.collision_type = rotating_collision_type

        self.space.add(self.link_body, self.link_shape)
        
        # Create revolute joint for pendulum rotation
        pivot_point = self.base_body.position
        revolute_joint = pymunk.PivotJoint(self.base_body, self.link_body, pivot_point)
        self.space.add(revolute_joint)

        # Collision handler
        handler = self.space.add_collision_handler(base_collision_type, rotating_collision_type)
        handler.begin = lambda arbiter, space, data: False

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
        
        #  # Calculate reward
        # base_center_dist = abs(x - (self.groove_length / 2)) / (self.groove_length / 2)
        # link_perpendicularity = 1 - abs(theta - np.pi)/(2*np.pi)
        # velocity_penalty = (abs(x_dot) + abs(theta_dot)) / 10.0
        # reward = (1.0 * (1 - base_center_dist) + 
        #         1.0 * link_perpendicularity - 
        #         0.5 * velocity_penalty)
        # # Clip reward to avoid extreme values
        # reward = max(reward, -10.0)       
        
        margin = np.deg2rad(5); reward = 0
        if  np.pi - margin <= theta < np.pi + margin:
            reward = 1

        self.steps += 1
        done = bool(
            self.steps >= self.max_steps
        )
        
        return obs, reward, done, {}

    def reset(self):
        # Reset positions and velocities
        self.create_pymunk_env()
        
        # Return initial observation
        x = self.base_body.position.x
        x_dot = self.base_body.velocity.x
        theta = self.link_body.angle
        theta_dot = self.link_body.angular_velocity
        
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Inverted Pendulum")
            self.clock = pygame.time.Clock()

        # keyboard handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                force_vector = (-self.force_mag, 0)
                self.base_body.apply_force_at_local_point(force_vector, (0, 0))
            elif keys[pygame.K_d]:
                force_vector = (self.force_mag, 0)
                self.base_body.apply_force_at_local_point(force_vector, (0, 0))
        
         # Offsets for centering the environment
        render_x = self.window_width / 2 - self.groove_length / 2
        render_y = self.window_height / 2

        self.screen.fill((255, 255, 255))  # Clear screen with white
        
        # Draw groove (centered using offsets)
        groove_start_x = render_x
        groove_end_x = render_x + self.groove_length
        pygame.draw.line(self.screen, (0, 0, 0), 
                         (groove_start_x, render_y), 
                         (groove_end_x, render_y), 2)
        
        # Draw base (centered using offsets)
        base_pos = self.base_body.position
        base_size = self.base_size
        base_rect = pygame.Rect(
            render_x + base_pos.x - base_size[0] / 2,
            render_y - base_pos.y - base_size[1] / 2,
            base_size[0],
            base_size[1]
        )
        pygame.draw.rect(self.screen, (0, 102, 153), base_rect)
        
        # Draw pendulum link (centered using offsets)
        pivot_pos = (render_x + base_pos.x, render_y - base_pos.y)
        link_pos = (render_x + self.link_body.position.x, render_y - self.link_body.position.y)
        pygame.draw.line(self.screen, (255, 153, 51), pivot_pos, link_pos, 4)
        
        pygame.display.flip()
        self.clock.tick(1/self.dt)  

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

