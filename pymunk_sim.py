import pymunk
import pygame
import pymunk.pygame_util

pygame.init()

WIDTH = 800
HEIGHT = 600

window = pygame.display.set_mode((WIDTH, HEIGHT))

def draw(space, window, draw_options):
    window.fill("white")
    space.debug_draw(draw_options)

def run(window, width, height):
    run = True
    clock = pygame.time.Clock()
    FPS = 75
    dt = 1/FPS

    space = pymunk.Space() # where we put objects    
    space.gravity = (0, 9.81)

    draw_options = pymunk.pygame_util.DrawOptions(window)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        draw(space, window, draw_options)
        space.step(dt)

        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":

    run(window, WIDTH, HEIGHT)


