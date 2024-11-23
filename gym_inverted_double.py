import pymunk
import pygame
import pymunk.pygame_util


def create_rect_obj(space, mass, size, pos, color = (0, 102, 153,255)):
    width, height = size
    inertia = pymunk.moment_for_box(mass, (width, height))
    body = pymunk.Body(mass, inertia)
    body.position = pos
    shape = pymunk.Poly.create_box(body, (width, height))
    shape.color = color
    space.add(body, shape)
    return body, shape # shape is probably unimportant


######################## BODIES AND JOINTS ########################
space = pymunk.Space()    
space.gravity = (0, 98.1)

base_collision_type = 1
rotating_collision_type = 2
# pendulum base
base_size = (20, 20)

base_body, base_shape = pendulum_base = create_rect_obj(
    space, 
    mass = 5, 
    size = base_size, 
    pos = (WIDTH/2,HEIGHT/2),
    color = (0, 102, 153,255))

base_shape.collision_type = base_collision_type  # Set collision type

# prismatic joint
static_body = space.static_body
groove_start = (0.1*WIDTH, HEIGHT/2); groove_end = (0.9*WIDTH, HEIGHT/2)   
groove_joint = pymunk.GrooveJoint(static_body, base_body, groove_start, groove_end, (0, 0))
space.add(groove_joint)
gear_joint = pymunk.GearJoint(static_body, base_body, 0, 1) 
space.add(gear_joint)

# pendulum link
link_size = (4, 200)
link_body, link_shape = create_rect_obj(
    space, 
    mass = 2, 
    size = link_size, 
    pos = (WIDTH/2,HEIGHT/2+link_size[1]/2),
    color = (255, 153, 51, 255))

link_shape.collision_type = rotating_collision_type  # Set collision type

# revolute joint
pivot_point = base_body.position
revolute_joint = pymunk.PivotJoint(base_body, link_body, pivot_point)
space.add(revolute_joint)

# Ignore collisions
handler = space.add_collision_handler(base_collision_type, rotating_collision_type)
handler.begin = lambda arbiter, space, data: False 

force_magnitude = 1500


