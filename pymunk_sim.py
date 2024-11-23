import pymunk
import pygame
import pymunk.pygame_util

pygame.init()

WIDTH = 800
HEIGHT = 600

window = pygame.display.set_mode((WIDTH, HEIGHT))


def draw(space, window, draw_options, debug=False):
    window.fill("gray")

    if debug:
        font = pygame.font.Font(None, 36)
        body = space.bodies[0]
        velocity = body.velocity  
        velocity_x = round(velocity.x, 2)
        velocity_text = font.render(f"Vx: ({velocity_x})", True, (0, 0, 0))

    window.blit(velocity_text, (10, 10))
    space.debug_draw(draw_options)

def create_rect_obj(space, mass, size, pos, color = (0, 102, 153,255)):
    width, height = size
    inertia = pymunk.moment_for_box(mass, (width, height))
    body = pymunk.Body(mass, inertia)
    body.position = pos
    shape = pymunk.Poly.create_box(body, (width, height))
    shape.color = color
    space.add(body, shape)
    return body, shape # shape is probably unimportant

def run(window, width, height):
    run = True
    clock = pygame.time.Clock()
    FPS = 75
    dt = 1/FPS

    space = pymunk.Space()    
    space.gravity = (0, 98.1)

    draw_options = pymunk.pygame_util.DrawOptions(window)

    ######################## BODIES AND JOINTS ########################
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

    ############################# MAINLOOP #############################
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                run = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                force_vector = (-force_magnitude, 0)
                base_body.apply_force_at_local_point(force_vector, (0, 0))
            elif keys[pygame.K_d]:
                force_vector = (force_magnitude, 0)
                base_body.apply_force_at_local_point(force_vector, (0, 0))

        draw(space, window, draw_options, debug=True)
        space.step(dt)

        clock.tick(FPS)
        pygame.display.update()
    
    pygame.quit()


if __name__ == "__main__":
    run(window, WIDTH, HEIGHT)


