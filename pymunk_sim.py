import pymunk
import pygame

pygame.init()

WIDTH = 800
HEIGHT = 600

window = pygame.display.set_mode((WIDTH, HEIGHT))

def run(window, width, height):
    run = True
    clock = pygame.time.Clock()
    FPS = 60

    space = pymunk.Space() # where we put objects    
    space.gravity = (0, -9.81)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":

    run(window, WIDTH, HEIGHT)


