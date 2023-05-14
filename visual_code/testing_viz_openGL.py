import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

# Let's create some 3D points
num_points = 1000
point_cloud = np.random.rand(num_points, 3) * 2 - 1  # points between -1 and 1

def PointCloud():
    glBegin(GL_POINTS)
    for point in point_cloud:
        glVertex3fv(point)
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        PointCloud()
        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    main()
