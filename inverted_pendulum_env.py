import gymnasium as gym
from gymnasium  import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame


class InvertedPendulumEnv(gym.Env):
    def __init__(self, 
                 gravity=98.1, 
                 dt=1/60.0, 
                 base_size=(20, 20),
                 base_mass=5, 
                 link_size=(4, 200), 
                 link_mass = 2,
                 groove_length = 600,
                 initial_angle=270,
                 max_steps = 1000,
                 actuation_max = 300,
                 margin = 1,
                 render_mode="human",
                 input_mode = "agent",
                 control_type = "swing-up"

                 ):
       
        super().__init__()
        
        # Store parameters
        self.gravity = gravity
        self.dt = dt
        self.base_size = base_size
        self.base_mass = base_mass
        self.link_size = link_size
        self.link_mass = link_mass
        self.groove_length = groove_length
        self.initial_angle = initial_angle
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.input_mode = input_mode
        self.control_type = control_type
        
        self.actuation_max = actuation_max
        self.margin = margin # for reward

        self.groove_y = 0

        # Pygame setup
        self.screen = None
        self.clock = None
        self.window_width = 800 * 2
        self.window_height = 600
        
        self.create_pymunk_env()
        
        self.max_angle=360
        self.max_position=self.groove_length

        # print(max_position, max_angle, np.finfo(np.float32).max)
        # Observation space: [x, x_dot, theta, theta_dot]
        high = np.array([self.max_position, np.finfo(np.float32).max, self.max_angle, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.action_space = spaces.Discrete(self.actuation_max*2+1) # force or speed applied to the base
        self.reward = 0
        self.steps = 0
        self.consecutive_success_count = 0

    def create_pymunk_env(self):
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
        if self.control_type == "swing-up":
            pos = (self.groove_length/2, self.groove_y - self.link_size[1]/2)
        elif self.control_type == "stabilization":
            pos = (self.groove_length/2, self.groove_y + self.link_size[1]/2)

        self.link_body, self.link_shape = self.create_rect_obj(
            mass=self.link_mass,
            size=self.link_size,
            pos=pos,
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
        info = {}
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
                pygame.display.set_caption("Inverted Pendulum")
                self.clock = pygame.time.Clock()

        if self.input_mode == "human":
            mouse_x, mouse_y = pygame.mouse.get_pos()
            center_x = self.window_width // 2
            if mouse_x == center_x:
                speed = 0
            elif mouse_x > center_x:
                speed = self.actuation_max * (mouse_x - center_x) / (self.window_width // 2)
            else:
                speed = -self.actuation_max * (center_x - mouse_x) / (self.window_width // 2)        
            if 0 < self.base_body.position.x < self.groove_length: # while inside the groove, 
                force = speed
                self.base_body.apply_force_at_local_point((force,0))
                info["force"] = force
                #self.base_body.velocity = (speed, 0)
                #print(speed)

        elif self.input_mode == "agent":
            # force = -self.actuation_max + action
            force = action
            self.base_body.apply_force_at_local_point((force,0))

            #speed = action
            #self.base_body.velocity = (speed,0)

        # Step the simulation
        self.space.step(self.dt)
        
        # Get observations
        # TODO: should I normalize obs between [0, 1] ?
        x = self.base_body.position.x #/ self.max_position
        x_dot = self.base_body.velocity.x
        # theta = self.link_body.angle
        if self.control_type == "swing-up":
            theta = np.rad2deg(np.mod(self.link_body.angle + np.deg2rad(270), 2 * np.pi)) #/ 360
        elif self.control_type == "stabilization":
            theta = np.rad2deg(np.mod(self.link_body.angle + np.deg2rad(90), 2 * np.pi)) #/ 360

        theta_dot = self.link_body.angular_velocity
        
        obs = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        
        #  # Calculate reward
        # self.reward = 0;
        # if np.abs(theta_dot) < 2:
        #     if  90 - self.margin <= theta < 90 + self.margin:
        #         self.reward = self.dt
        # # TODO: divide by theta_dot
        # # TODO: add centering effect

        rewards = {
            "theta_dot":-0.2, # Reward for small theta_dot
            "theta":1,        # Reward for being close to the target angle
        }
        reward = 0

        # Penalize high angular velocity
        if np.abs(theta_dot) > 3: reward += rewards["theta_dot"]
        else:
            #if np.abs(x_dot)<200:
            if 90 - self.margin <= theta <= 90 + self.margin:
                target_x = self.groove_length / 2  
                position_deviation = np.abs(x - target_x) / (self.groove_length / 2)  # Normalize deviation
                if position_deviation < 0.3:
                    self.consecutive_success_count += 0.5
                    reward += rewards["theta"] + self.consecutive_success_count
            else:
                self.consecutive_success_count = 0

        self.reward = (reward * self.dt) * 10

        # rewards = {
        #     "theta_dot": -0.1,  # Penalize high angular velocity
        #     "theta_proximity": 1,  # Reward for being close to 90 degrees
        #     "position_proximity": 0.5,  # Reward for being near the center of the groove
        #     "success_bonus": 2,  # Bonus for achieving the target state
        # }
        # reward = 0
        
        # # Penalize high angular velocity
        # if np.abs(theta_dot) > 3:
        #     reward += rewards["theta_dot"]
        
        # # Check if angle is near the target (90 degrees within margin)
        # if 90 - self.margin <= theta <= 90 + self.margin:
        #     # Calculate normalized deviation from the center of the groove
        #     target_x = self.groove_length / 2
        #     position_deviation = np.abs(x - target_x) / (self.groove_length / 2)
            
        #     # Reward proximity to the center of the groove
        #     if position_deviation < 0.5:
        #         reward += rewards["theta_proximity"] + (1 - position_deviation) * rewards["position_proximity"]
                
        #         # Add success bonus for consecutive stability
        #         self.consecutive_success_count += 1
        #         reward += self.consecutive_success_count * rewards["success_bonus"]
        #     else:
        #         self.consecutive_success_count = 0
        # else:
        #     self.consecutive_success_count = 0
        
        ## Scale the reward for time-step and adjust weight
        #self.reward = reward * self.dt

        self.steps += 1
        done = self.steps >= self.max_steps
        
        return obs, self.reward, done, False, info # add info for easier print

    def reset(self, seed = None):
        # Reset positions and velocities
        super().reset(seed=seed)
        self.steps = 0

        self.create_pymunk_env()
        # Return initial observation
        x = self.base_body.position.x
        x_dot = self.base_body.velocity.x
        theta = self.link_body.angle
        theta_dot = self.link_body.angular_velocity
        
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32), {}

    def render(self, manual_test = False):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
                pygame.display.set_caption("Inverted Pendulum")
                self.clock = pygame.time.Clock()

            # keyboard handler
            for event in pygame.event.get():
                if event.type == pygame.QUIT :
                    break

     #        mouse_x, mouse_y = pygame.mouse.get_pos()
     #        center_x = self.window_width // 2
     #        if mouse_x == center_x:
     #            speed = 0
     #        elif mouse_x > center_x:
     #            speed = self.actuation_max * (mouse_x - center_x) / (self.window_width // 2)
     #        else:
                # speed = -self.actuation_max * (center_x - mouse_x) / (self.window_width // 2)        
     #        if 0 < self.base_body.position.x < self.groove_length:
     #            force = speed
     #            self.base_body.apply_force_at_local_point((force,0))
     #        self.space.step(self.dt)

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
            pygame.draw.line(self.screen, (255, 153, 51), pivot_pos, link_pos, self.link_size[0])

            if manual_test:
                center_x = self.window_width // 2  
                pygame.draw.line(self.screen, (0,0,0), (center_x, 0), (center_x, self.window_height), 2)  # Draw the vertical line
            
            pygame.display.flip()
            self.clock.tick(1/self.dt)  

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

